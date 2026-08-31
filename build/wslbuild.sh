#!/bin/bash
# Build helper: run via `wsl ./wslbuild.sh` from the build directory.
# Uses the same SDK as ~/.bashrc (/mnt/c/pspdev, GCC 14.1).
export PSPDEV=/mnt/c/pspdev
export PSPSDK=$PSPDEV/psp/sdk
export PATH=$PSPDEV/bin:$PSPSDK/bin:/usr/local/bin:/usr/bin:/bin
exec make "$@"
