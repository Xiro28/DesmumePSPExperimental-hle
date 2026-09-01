#!/bin/bash
export PATH=/mnt/c/pspdev/bin:/usr/bin:/bin
psp-nm -C DesmumePSP.elf 2>/dev/null | grep -iE " T (exec_hw|exec_cpu|execHardware_interrupts)\(\)?$" | sort
