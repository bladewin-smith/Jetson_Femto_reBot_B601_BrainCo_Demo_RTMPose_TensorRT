#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd -- "$SCRIPT_DIR/.." && pwd)"
MODEL_DIR="$REPO/models/tensorrt"
MODEL_PATHS=(
  "models/tensorrt/rtmdet_nano.engine"
  "models/tensorrt/rtmpose_body.engine"
  "models/tensorrt/rtmpose_hand.engine"
)
MANIFEST="$MODEL_DIR/verified_engines.sha256"

cd "$REPO"
command -v git-lfs >/dev/null 2>&1 || {
  echo "git-lfs is not installed. Install it with: sudo apt install git-lfs" >&2
  exit 2
}
git lfs install --local

for relative_path in "${MODEL_PATHS[@]}"; do
  file="$REPO/$relative_path"
  if [[ ! -s "$file" ]]; then
    echo "Missing or empty model: $file" >&2
    exit 2
  fi
  if [[ "$(git check-attr filter -- "$relative_path" | awk '{print $3}')" != "lfs" ]]; then
    echo "Git LFS is not configured for: $file" >&2
    exit 2
  fi
done
if [[ ! -s "$MANIFEST" ]]; then
  echo "Missing checksum manifest: $MANIFEST" >&2
  exit 2
fi
(cd "$MODEL_DIR" && sha256sum -c verified_engines.sha256)

git add -- .gitattributes \
  models/tensorrt/rtmdet_nano.engine \
  models/tensorrt/rtmpose_body.engine \
  models/tensorrt/rtmpose_hand.engine \
  models/tensorrt/verified_engines.sha256

echo
echo "Staged verified model bundle through Git LFS:"
git lfs ls-files
echo
echo "Review with: git status --short"
echo "Then upload with:"
echo "  git commit -m 'add verified B601 TensorRT engines'"
echo "  git push origin HEAD"
