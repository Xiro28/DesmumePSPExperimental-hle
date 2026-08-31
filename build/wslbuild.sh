#!/bin/bash
# Build helper: run via `wsl ./wslbuild.sh` from the build directory.
#
# Uses the same SDK as ~/.bashrc (/mnt/c/pspdev, GCC 14.1). There is also an
# older /usr/local/pspdev (GCC 11.2) on this machine -- do not mix them: GCC 14
# emits calls to std::string's _M_replace_cold, which GCC 11's libstdc++ does
# not export, so objects built by one fail to link against the other.
export PSPDEV=/mnt/c/pspdev
export PSPSDK=$PSPDEV/psp/sdk
export PATH=$PSPDEV/bin:$PSPSDK/bin:/usr/local/bin:/usr/bin:/bin
exec make "$@"
