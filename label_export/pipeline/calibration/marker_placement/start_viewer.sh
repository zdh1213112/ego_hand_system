#!/usr/bin/env bash
set -euo pipefail
TOOL_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd -- "$TOOL_DIR/../../../.." && pwd)"
VIEWER_DIR="${1:-$ROOT_DIR/output/native_kb_training_20261111/baseline_mesh_viewer}"
if [[ ! -f "$VIEWER_DIR/index.html" ]]; then
  echo "查看器尚未生成：$VIEWER_DIR/index.html" >&2
  exit 1
fi
echo 'Mesh 检查：http://127.0.0.1:8766/'
echo 'Marker 贴点：http://127.0.0.1:8766/marker_placement/'
echo '保持终端运行，Ctrl+C 退出。远程电脑访问时请转发 SSH 端口 8766。'
exec python3 -m http.server 8766 --bind 127.0.0.1 --directory "$VIEWER_DIR"
