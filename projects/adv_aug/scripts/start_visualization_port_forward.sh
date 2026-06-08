#!/usr/bin/env bash
set -euo pipefail

PORT="${PORT:-8000}"
VIS_DIR="${VIS_DIR:-/beegfs/jung/mmdet3d_legacy/projects/adv_aug/runs/adhoc/visualizations}"

usage() {
  cat <<EOF
Usage:
  $0

Environment overrides:
  PORT      Port used by the HTTP server. Default: ${PORT}
  VIS_DIR   Directory served by HTTP server. Default: ${VIS_DIR}

Run this on the remote machine, then use VS Code's port forwarding for PORT.
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" || "${1:-}" == "help" ]]; then
  usage
  exit 0
fi

if [[ $# -gt 0 ]]; then
  echo "Unexpected argument: $1" >&2
  usage >&2
  exit 2
fi

if [[ ! -d "${VIS_DIR}" ]]; then
  echo "Visualization directory does not exist: ${VIS_DIR}" >&2
  exit 1
fi

cat <<EOF
Serving visualizations from:
  ${VIS_DIR}

Starting:
  cd "${VIS_DIR}"
  python3 -m http.server ${PORT}

Forward port ${PORT} in VS Code, then open the forwarded local URL.

Press Ctrl-C here to stop the HTTP server.
EOF
cd "${VIS_DIR}"
exec python3 -m http.server "${PORT}"
