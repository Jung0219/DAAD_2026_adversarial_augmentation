#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-serve}"
if [[ $# -gt 0 ]]; then
  shift
fi

REMOTE_HOST="${REMOTE_HOST:-jung@fugg2.pleiades.uni-wuppertal.de}"
REMOTE_PORT="${REMOTE_PORT:-8000}"
LOCAL_PORT="${LOCAL_PORT:-8000}"
VIS_DIR="${VIS_DIR:-/beegfs/jung/mmdet3d_legacy/projects/adv_aug/runs/adhoc/visualizations}"
BIND_HOST="${BIND_HOST:-127.0.0.1}"

usage() {
  cat <<EOF
Usage:
  $0 serve
  $0 tunnel

Environment overrides:
  REMOTE_HOST   SSH target for tunnel mode. Default: ${REMOTE_HOST}
  REMOTE_PORT   Port used by the remote HTTP server. Default: ${REMOTE_PORT}
  LOCAL_PORT    Local browser port forwarded to REMOTE_PORT. Default: ${LOCAL_PORT}
  VIS_DIR       Directory served by HTTP server. Default: ${VIS_DIR}
  BIND_HOST     Address for the remote HTTP server. Default: ${BIND_HOST}

Typical workflow:
  1. On the remote machine:
       $0 serve

  2. On your laptop:
       REMOTE_HOST=${REMOTE_HOST} $0 tunnel

  3. Open:
       http://127.0.0.1:${LOCAL_PORT}/
EOF
}

case "${MODE}" in
  serve)
    if [[ ! -d "${VIS_DIR}" ]]; then
      echo "Visualization directory does not exist: ${VIS_DIR}" >&2
      exit 1
    fi

    cat <<EOF
Serving visualizations from:
  ${VIS_DIR}

From your laptop, start the tunnel with:
  ssh -N -L ${LOCAL_PORT}:${BIND_HOST}:${REMOTE_PORT} ${REMOTE_HOST}

Then open:
  http://127.0.0.1:${LOCAL_PORT}/

Press Ctrl-C here to stop the HTTP server.
EOF
    exec python3 -m http.server "${REMOTE_PORT}" --bind "${BIND_HOST}" --directory "${VIS_DIR}"
    ;;

  tunnel)
    echo "Forwarding http://127.0.0.1:${LOCAL_PORT}/ to ${REMOTE_HOST}:${REMOTE_PORT}"
    echo "Press Ctrl-C to stop the tunnel."
    exec ssh -N -L "${LOCAL_PORT}:${BIND_HOST}:${REMOTE_PORT}" "${REMOTE_HOST}"
    ;;

  -h|--help|help)
    usage
    ;;

  *)
    echo "Unknown mode: ${MODE}" >&2
    usage >&2
    exit 2
    ;;
esac
