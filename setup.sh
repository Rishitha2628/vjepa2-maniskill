#!/bin/bash
# Fetch the upstream V-JEPA 2 source this project builds its model from.
# Not vendored into git: it is Meta's code under its own license.
set -e
if [ -d vjepa2_src ]; then
  echo "vjepa2_src already present"
else
  git clone --depth 1 https://github.com/facebookresearch/vjepa2.git vjepa2_src
  rm -rf vjepa2_src/.git
  echo "cloned vjepa2_src"
fi
echo
echo "Weights download automatically on first use (5.27 GB from facebook/jepa-wms)."
echo "Python deps: torch torchvision mani-skill gymnasium huggingface_hub scipy timm einops"
