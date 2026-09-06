"""
Loads V-JEPA 2-AC and exposes the two operations everything else needs:
`encode()` and `predict_next()`.

Why this does not use torch.hub
-------------------------------
Two independent upstream problems:

1. `facebookresearch/vjepa2` ships src/hub/backbones.py with a debug value left
   in -- `VJEPA_BASE_URL = "http://localhost:8300"` -- so nothing downloads.
2. The real checkpoint, dl.fbaipublicfiles.com/vjepa2/vjepa2-ac-vitg.pt, is
   11.76 GB (it carries optimizer state).

So we vendor the model source in ./vjepa2_src and pull the stripped 5.27 GB
inference checkpoint from the Hugging Face mirror `facebook/jepa-wms` instead.

Memory
------
The checkpoint is fp32 and the encoder is ViT-giant. Building the model
normally would need ~4 GB of CPU RAM for fp32 weights on top of the ~5.3 GB
checkpoint. Instead the modules are built on the meta device and the weights
are cast to fp16 tensor-by-tensor and installed with assign=True, so peak host
RAM stays near the size of the checkpoint alone.

Conventions
-----------
`states` and `actions` here are always DROID-convention (euler + gripper
closedness, metric deltas). Use adapter.py to get them from ManiSkill.
"""

import os
import sys

import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
VJEPA_SRC = os.path.join(HERE, "vjepa2_src")
if VJEPA_SRC not in sys.path:
    sys.path.insert(0, VJEPA_SRC)

HF_REPO = "facebook/jepa-wms"
HF_FILE = "vjepa2_ac_oss.pth.tar"

IMG_SIZE = 256
PATCH_SIZE = 16
TUBELET = 2
NUM_FRAMES = 64
EMBED_DIM = 1408          # ViT-giant
TOKENS_PER_FRAME = (IMG_SIZE // PATCH_SIZE) ** 2   # 256

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def checkpoint_path():
    from huggingface_hub import hf_hub_download

    return hf_hub_download(repo_id=HF_REPO, filename=HF_FILE)


def _strip(state_dict):
    out = {}
    for k, v in state_dict.items():
        k = k.replace("module.", "").replace("backbone.", "")
        out[k] = v
    return out


def load_ac_model(device="cuda", dtype=torch.float16, ckpt=None):
    """Returns (encoder, predictor), both eval() and on `device`."""
    from src.models import ac_predictor as ac_mod, vision_transformer as vit_mod

    enc_kwargs = dict(
        patch_size=PATCH_SIZE,
        img_size=(IMG_SIZE, IMG_SIZE),
        num_frames=NUM_FRAMES,
        tubelet_size=TUBELET,
        use_sdpa=True,
        use_SiLU=False,
        wide_SiLU=True,
        uniform_power=False,
        use_rope=True,
    )
    # Build directly in `dtype` rather than fp32. The meta device would be the
    # obvious way to avoid the allocation entirely, but VisionTransformer's
    # __init__ calls .item() on a tensor, which meta tensors do not support.
    prev_dtype = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        encoder = vit_mod.__dict__["vit_giant_xformers"](**enc_kwargs)
        predictor = ac_mod.vit_ac_predictor(
            img_size=(IMG_SIZE, IMG_SIZE),
            patch_size=PATCH_SIZE,
            num_frames=NUM_FRAMES,
            tubelet_size=TUBELET,
            embed_dim=EMBED_DIM,
        )
    finally:
        torch.set_default_dtype(prev_dtype)

    path = ckpt or checkpoint_path()
    try:
        sd = torch.load(path, map_location="cpu", mmap=True, weights_only=False)
    except (RuntimeError, TypeError):
        sd = torch.load(path, map_location="cpu", weights_only=False)

    enc_sd = {k: v.to(dtype) for k, v in _strip(sd["encoder"]).items()}
    missing, unexpected = encoder.load_state_dict(enc_sd, strict=False, assign=True)
    if missing:
        print(f"encoder: {len(missing)} missing keys (first: {missing[:3]})")
    del enc_sd

    pred_sd = {k: v.to(dtype) for k, v in _strip(sd["predictor"]).items()}
    predictor.load_state_dict(pred_sd, strict=True, assign=True)
    del pred_sd, sd

    # attn_mask is a buffer built in __init__, so it came out of the meta build
    # empty. Rebuild it on the real device.
    from src.models.utils.modules import build_action_block_causal_attention_mask

    predictor.attn_mask = build_action_block_causal_attention_mask(
        NUM_FRAMES // TUBELET, IMG_SIZE // PATCH_SIZE, IMG_SIZE // PATCH_SIZE,
        add_tokens=2,
    )

    encoder = encoder.to(device=device).eval()
    predictor = predictor.to(device=device).eval()
    for p in list(encoder.parameters()) + list(predictor.parameters()):
        p.requires_grad_(False)
    return encoder, predictor


def _autocast(device, dtype):
    """RoPE builds its angles in fp32 while values stay fp16, which makes
    F.scaled_dot_product_attention reject the mismatched dtypes. Autocast
    reconciles them at the SDPA call and keeps the rotation itself in fp32,
    which is also the more accurate order of operations."""
    enabled = device != "cpu" and dtype in (torch.float16, torch.bfloat16)
    return torch.autocast("cuda", dtype=dtype, enabled=enabled)


def load_predictor(device="cuda", dtype=torch.float32, ckpt=None):
    """Just the 305M AC predictor, for fine-tuning on cached latents.

    Training never needs the encoder: it is frozen, so collect_data.py runs it
    once and caches its output. Skipping it here frees ~2 GB of VRAM and most of
    the load time.
    """
    from src.models import ac_predictor as ac_mod
    from src.models.utils.modules import build_action_block_causal_attention_mask

    prev = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        predictor = ac_mod.vit_ac_predictor(
            img_size=(IMG_SIZE, IMG_SIZE), patch_size=PATCH_SIZE,
            num_frames=NUM_FRAMES, tubelet_size=TUBELET, embed_dim=EMBED_DIM)
    finally:
        torch.set_default_dtype(prev)

    path = ckpt or checkpoint_path()
    try:
        sd = torch.load(path, map_location="cpu", mmap=True, weights_only=False)
    except (RuntimeError, TypeError):
        sd = torch.load(path, map_location="cpu", weights_only=False)
    pred_sd = {k: v.to(dtype) for k, v in _strip(sd["predictor"]).items()}
    predictor.load_state_dict(pred_sd, strict=True, assign=True)
    del pred_sd, sd

    predictor.attn_mask = build_action_block_causal_attention_mask(
        NUM_FRAMES // TUBELET, IMG_SIZE // PATCH_SIZE, IMG_SIZE // PATCH_SIZE,
        add_tokens=2)
    return predictor.to(device=device)


def preprocess(frames_uint8, device, dtype):
    """(T,H,W,3) uint8 -> (T,3,H,W) normalized, matching app/vjepa_droid/transforms."""
    x = torch.as_tensor(frames_uint8, device=device).permute(0, 3, 1, 2).float() / 255.0
    if x.shape[-1] != IMG_SIZE or x.shape[-2] != IMG_SIZE:
        x = F.interpolate(x, size=(IMG_SIZE, IMG_SIZE), mode="bilinear",
                          align_corners=False)
    mean = torch.tensor(IMAGENET_MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=device).view(1, 3, 1, 1)
    return ((x - mean) / std).to(dtype)


@torch.no_grad()
def encode(encoder, frames_uint8, device, dtype, normalize=True):
    """(T,H,W,3) uint8 -> (T, TOKENS_PER_FRAME, D) layer-normed reps.

    Each frame is encoded as its own 2-frame clip (the frame repeated), which is
    how the upstream WorldModel wrapper turns single images into one temporal
    token under tubelet_size=2.
    """
    imgs = preprocess(frames_uint8, device, dtype)          # (T,3,H,W)
    clip = imgs.unsqueeze(2).repeat(1, 1, TUBELET, 1, 1)     # (T,3,2,H,W)
    with _autocast(device, dtype):
        h = encoder(clip)                                    # (T, tokens, D)
    if normalize:
        h = F.layer_norm(h.float(), (h.size(-1),)).to(dtype)
    return h


@torch.no_grad()
def predict_next(predictor, z, actions, states, normalize=True):
    """One step of imagined dynamics.

    z       : (B, T*TOKENS_PER_FRAME, D)  context reps, oldest frame first
    actions : (B, T, 7) metric DROID actions
    states  : (B, T, 7) DROID poses
    returns : (B, TOKENS_PER_FRAME, D) prediction for the frame after the last

    The predictor is frame-causal and interleaves one action and one state
    token before each frame's patch tokens, so it returns a prediction for
    every frame position; only the last is the new frame.
    """
    with _autocast(z.device.type, z.dtype):
        out = predictor(z, actions, states)[:, -TOKENS_PER_FRAME:]
    if normalize:
        out = F.layer_norm(out.float(), (out.size(-1),)).to(out.dtype)
    return out
