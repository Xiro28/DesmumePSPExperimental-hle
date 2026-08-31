// m-c/d 2025
#include "main.h"

PSP_MODULE_INFO("me2sc", 0, 1, 1);
PSP_HEAP_SIZE_KB(-1024);
PSP_MAIN_THREAD_ATTR(PSP_THREAD_ATTR_VFPU | PSP_THREAD_ATTR_USER);

static volatile u32* mem = nullptr;
#define meCounter         (mem[0])
#define meExit            (mem[1])

constexpr size_t buffer_size = 192*256*4;

__attribute__((noinline, aligned(4)))
static void meLoop() {
  do {
    meDCacheWritebackInvalidAll();
  } while(!mem);
  
  u32 offset = 0;
  u32 color = 0xffffffff;
  do {
    hw((CACHED_KERNEL_MASK | ME_EDRAM_BASE) + offset) = color;
    offset += 4;
    if (!((0xff & offset) / 2)) {
      color = 0xff000000 | (randInRange(0xff) << 16) |
        (randInRange(0xff) << 8) | randInRange(0xff);
    }
  } while(offset < buffer_size);
  meDCacheWritebackInvalidAll();
  
  do {
    meCounter++;
  } while(meExit == 0);
  meExit = 2;
  meHalt();
}

extern char __start__me_section;
extern char __stop__me_section;
__attribute__((section("_me_section")))
void meHandler() {
  hw(0xbc100050) = 0x0f;       // 0b1111, enable me and AW buses
  hw(0xbc100004) = 0xffffffff; // enable NMI interrupts in the global hardware context
  hw(0xbc100040) = 0x02;       // allow 64MB ram
  asm("sync");
  
  asm volatile(
    "li          $k0, 0x30000000\n"
    "mtc0        $k0, $12\n"
    "sync\n"
    "la          $k0, %0\n"
    "li          $k1, 0x80000000\n"
    "or          $k0, $k0, $k1\n"
    "jr          $k0\n"
    "nop\n"
    :
    : "i" (meLoop)
    : "k0"
  );
}

static int initMe() {
  #define me_section_size (&__stop__me_section - &__start__me_section)
  memcpy((void *)ME_HANDLER_BASE, (void*)&__start__me_section, me_section_size);
  meDCacheWritebackInvalidAll();
  // reset me
  hw(0xbc10004c) = 0x04;
  hw(0xbc10004c) = 0x0;
  asm volatile("sync");
  return 0;
}

static volatile DMADescriptor* lli = nullptr;

static int me2Sc() {
  cleanChannel();
  dmacplusLLIFromMe(lli);
  waitChannel();
  return 0;
}

void exitSample(const char* const str) {
  pspDebugScreenInit();
  pspDebugScreenClear();
  pspDebugScreenSetXY(0, 1);
  pspDebugScreenPrintf(str);
  sceKernelDelayThread(100000);
  sceKernelExitGame();
}

int main() {
  scePowerSetClockFrequency(333, 333, 166);

  if (pspSdkLoadStartModule("ms0:/PSP/GAME/me/kcall.prx", PSP_MEMORY_PARTITION_KERNEL) < 0){
    exitSample("Can't load the PRX, exiting...");
    return 0;
  }

  mem = meGetUncached32(4);
  kcall(&initMe);
  
  sceDisplaySetFrameBuf((void*)(UNCACHED_USER_MASK |
    GE_EDRAM_BASE), 512, 3, PSP_DISPLAY_SETBUF_NEXTFRAME);
  pspDebugScreenInitEx(0, PSP_DISPLAY_PIXEL_FORMAT_8888, 0);
  pspDebugScreenSetOffset(0);

  lli = dmacplusInitMe2ScLLI(ME_EDRAM_BASE, GE_EDRAM_BASE, buffer_size);

  SceCtrlData ctl;
  do {
    kcall(&me2Sc);
    sceCtrlPeekBufferPositive(&ctl, 1);
    pspDebugScreenSetXY(0, 17);
    pspDebugScreenPrintf("%x          ", meCounter);
    sceDisplayWaitVblankStart();
  } while(!(ctl.Buttons & PSP_CTRL_HOME));
  
  meExit = 1;
  do {
    asm volatile("sync");
  } while(meExit < 2);
  
  free((void*)lli);
  meGetUncached32(0);
  exitSample("Exiting...");
  return 0;
}
