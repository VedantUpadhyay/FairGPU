#!/bin/bash
# Apply FairGPU value_greedy patch to vLLM submodule
# Run from FairGPU root: bash patches/apply_vllm_patch.sh

set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
VLLM_DIR="$SCRIPT_DIR/../vllm"
PATCH_FILE="$SCRIPT_DIR/vllm_value_greedy.patch"

echo "Applying FairGPU patch to vLLM..."
cd "$VLLM_DIR"
git apply "$PATCH_FILE"
echo "Patch applied successfully."
echo "Modified files:"
git diff --name-only
