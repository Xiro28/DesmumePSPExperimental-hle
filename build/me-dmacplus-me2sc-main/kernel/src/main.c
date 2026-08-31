// m-c/d 2025
#include "kcall.h"
#include <pspsdk.h>
#include <psppower.h>

PSP_MODULE_INFO("kcall", 0x1006, 1, 1);
PSP_NO_CREATE_MAIN_THREAD();

int kcall(FCall const f) {
  return f();
}

int module_start(SceSize args, void *argp){
  scePowerLock(0);
  return 0;
}

int module_stop() {
  return 0;
}
