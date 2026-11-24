#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd -- "$SCRIPT_DIR/.." && pwd)"
USER_ENV="${B601_DESKTOP_ENV_FILE:-$HOME/.config/rebot_b601_demo.env}"

if [[ -f "$USER_ENV" ]]; then
  set -a
  set +u
  # shellcheck disable=SC1090
  source "$USER_ENV"
  set -u
  set +a
fi

VENV_DIR="${B601_VENV_DIR:-$REPO/.venv}"
if [[ ! -x "$VENV_DIR/bin/python" ]]; then
  echo "B601 Python virtual environment is missing: $VENV_DIR" >&2
  echo "Create the repository .venv first, then run the desktop installer again." >&2
  read -r -p "Press Enter to close..." _
  exit 2
fi

# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"
export PYORBBEC_ROOT="${PYORBBEC_ROOT:-$HOME/ws/pyorbbecsdk-2-main}"
export BRAINCO_HAND_SDK_PATH="${BRAINCO_HAND_SDK_PATH:-$HOME/ws/brainco-hand-sdk-main/python}"
export B601_LOCAL_ENGINE_DIR="${B601_LOCAL_ENGINE_DIR:-$REPO/models/tensorrt}"
export PYTHONPATH="$PYORBBEC_ROOT:$BRAINCO_HAND_SDK_PATH:$REPO:$REPO/scripts:$REPO/third_party/reBotArm_control_py:${PYTHONPATH:-}"

cd "$REPO"
exec python "$REPO/scripts/b601_desktop_app.py" \
  --channels "${B601_DESKTOP_CHANNELS:-[1.0,1.0,0.0,0.0,1.0,1.0]}"
