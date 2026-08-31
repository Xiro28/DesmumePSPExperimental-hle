#!/bin/bash
# Build helper: run via `wsl ./wslbuild.sh` from the build directory.
export PSPDEV=/usr/local/pspdev
export PSPSDK=$PSPDEV/psp/sdk
export PATH=$PSPDEV/bin:$PSPSDK/bin:/usr/local/bin:/usr/bin:/bin
exec make "$@"
