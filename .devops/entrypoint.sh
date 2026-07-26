#!/usr/bin/env bash
set -e

# Activate the Intel oneAPI environment so icx/icpx, oneDNN and oneMKL are on
# PATH / LD_LIBRARY_PATH / CMAKE_PREFIX_PATH for whatever command runs next.
if [ -f /opt/intel/oneapi/setvars.sh ]; then
    # shellcheck disable=SC1091
    source /opt/intel/oneapi/setvars.sh --force >/dev/null 2>&1 || true
fi

# setvars.sh prepends the base image's own oneDNN to LD_LIBRARY_PATH, which
# shadows the source build in /opt/onednn that the headers come from. The
# mismatch is silent -- the loader resolves libdnnl.so.3 to whichever copy is
# first -- and surfaces much later as a primitive that the newer headers
# advertise but the older library does not implement. Put the source build back
# in front so the runtime matches what was compiled against.
if [ -d /opt/onednn/lib ]; then
    export LD_LIBRARY_PATH="/opt/onednn/lib:${LD_LIBRARY_PATH}"
fi

exec "$@"
