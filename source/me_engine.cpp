// Code taken from https://github.com/mcidclan/me-sample-interrupt/
#include "me_engine.h"
#include "MMU.h"
#include <pspdmac.h>

__attribute__((noinline, aligned(64))) bool isEmu = true;

#define ME_EDRAM_BASE 0
#define ME_EDRAM_DISPLAY ME_EDRAM_BASE
#define ME_EDRAM_MAIN_GPU ((ME_EDRAM_DISPLAY + (192 * 256 * 5) + 15) & ~15)
#define ME_EDRAM_SUB_GPU (((ME_EDRAM_MAIN_GPU + sizeof(GPU)) + 15) & ~15)
#define GE_EDRAM_BASE 0x04100000
#define UNCACHED_USER_MASK 0x40000000
#define CACHED_KERNEL_MASK 0x80000000
#define UNCACHED_KERNEL_MASK 0xA0000000

#define DMA_CONTROL_SC2SC(width, size) DMA_CONTROL_CONFIG(1, 1, 0, 0, (width), (size))
#define DMA_CONTROL_SC2ME(width, size) DMA_CONTROL_CONFIG(0, 0, 1, 0, (width), (size))
#define DMA_CONTROL_ME2SC(width, size) DMA_CONTROL_CONFIG(0, 0, 0, 1, (width), (size))

#define DMA_CONTROL_CONFIG(idst, isrc, mdst, msrc, width, size) (         \
    (0U << 31) |                /* terminal count interrupt enable bit */ \
    ((unsigned)(idst) << 27) |  /* dst addr increment ?                */ \
    ((unsigned)(isrc) << 26) |  /* src addr increment ?                */ \
    ((unsigned)(mdst) << 25) |  /* dst AHB master select               */ \
    ((unsigned)(msrc) << 24) |  /* src AHB master select               */ \
    ((unsigned)(width) << 21) | /* dst transfer width                  */ \
    ((unsigned)(width) << 18) | /* src transfer width                  */ \
    (1U << 15) |                /* dst burst size or dst step ?        */ \
    (1U << 12) |                /* src burst size or src step ?        */ \
    ((unsigned)(size)))         /* transfer size                       */

static volatile u32 *mem = nullptr;
#define start_draw (mem[0])
// used to sync drawing
#define syncDrawSc (mem[1])
#define syncDrawMe (mem[2])
#define syncTrue (mem[3])
// Closed-loop sync: CPU bumps framesSubmitted on each draw request,
// ME bumps framesCompleted when it finishes. lag = submitted - completed.
#define framesSubmitted (mem[4])
#define framesCompleted (mem[5])

struct DMADescriptor
{
  u32 src;
  u32 dst;
  u32 next;
  u32 ctrl;
  u32 status;
} __attribute__((aligned(16)));

__attribute__((noinline, aligned(4))) static int meLoop()
{
  meDCacheWritebackInvalidAll(); // making sure mem is available

  bool skip = true;

  memset((void *)ME_EDRAM_BASE, 0x7FFF, sizeof(GPU_Screen));

  NDS_Screen main_gpu{.gpu = (GPU *)ME_EDRAM_MAIN_GPU, .offset = 0};
  NDS_Screen sub_gpu{.gpu = (GPU *)ME_EDRAM_SUB_GPU, .offset = 512};

  do
  {
    if (start_draw == 1) // if (vrg(0xA0000000) > 0)
    {
      start_draw = 0;
      syncDrawSc = 0;

      asm volatile("sync");

      memcpy((GPU *)ME_EDRAM_MAIN_GPU, MainScreen.gpu, sizeof(GPU));
      memcpy((GPU *)ME_EDRAM_SUB_GPU, SubScreen.gpu, sizeof(GPU));

      main_gpu.offset = MainScreen.offset;
      sub_gpu.offset = SubScreen.offset;

      for (int i = 0; i < 192; ++i)
      {
        GPU_RenderLine(&main_gpu, i, false);
        GPU_RenderLine(&sub_gpu, i, false);
      }

      skip = !skip;

      // Only the rendered display buffer needs to reach RAM for the DMA gather,
      // not the whole cache. Counters live in uncached mem, already coherent.
      //meDCacheWritebackRange(ME_EDRAM_DISPLAY, (256 * 192 * 4));
      meDCacheWritebackInvalidAll();
      
      framesCompleted = framesCompleted + 1;
      asm volatile("sync");
      vrg(0xBC100044) = 1;
    }
  } while (!uncached(_meExit));
  return uncached(_meExit);
}

extern char __start__me_section;
extern char __stop__me_section;
__attribute__((section("_me_section"), noinline, aligned(4))) int meHandler()
{
  vrg(0xbc100050) = 0x0f;       // enable clocks: ME, AW bus RegA, RegB & Edram, DMACPlus, DMAC
  vrg(0xbc100004) = 0xffffffff; // clear NMI
  vrg(0xbc100040) = 0x02;       // allow 64MB ram

    // disable the vme processing
    vrg(0xBCC00000) = -1;
    vrg(0xBCC00030) = 1;
    vrg(0xBCC00040) = 1;
    
  asm volatile("sync");
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
      : "i"(meLoop)
      : "k0");
  return 0;
}

inline void cleanChannels()
{
  vrg(0xbc800190) = 0;
  vrg(0xbc8001b0) = 0;
  vrg(0xbc8001d0) = 0;
  asm volatile("sync");
}

int waitChannels()
{
  while (vrg(0xbc800190) ||
         vrg(0xbc8001b0) ||
         vrg(0xbc8001d0))
  {
    asm volatile("sync");
    sceKernelDelayThread(10);
  };

  return 0;
}

static int initMe()
{
#define ME_HANDLER_BASE 0xbfc00000
#define me_section_size (&__stop__me_section - &__start__me_section)
  memcpy((void *)ME_HANDLER_BASE, (void *)&__start__me_section, me_section_size);
  meDCacheWritebackInvalidAll();


  // reset and start me and vme
  vrg(0xBC10004C) = 0x4;
  vrg(0xBC10004C) = 0x0;
  asm volatile("sync");
  return 0;
}

int kcpuSendInterrupt()
{
  vrg(0xBC100044) = 1;
  asm("sync");
  return 0;
}
inline void dmacplusLLIFromSc(volatile const DMADescriptor *const lli)
{
  vrg(0xBC800180) = lli->src;    // src
  vrg(0xBC800184) = lli->dst;    // dest
  vrg(0xBC800188) = lli->next;   // addr of the next LLI
  vrg(0xBC80018c) = lli->ctrl;   // control attr
  vrg(0xBC800190) = lli->status; // status
  asm volatile("sync");
}

inline void dmacplusLLIFromMe(volatile const DMADescriptor *const lli)
{
  vrg(0xbc8001a0) = lli->src;
  vrg(0xbc8001a4) = lli->dst;
  vrg(0xbc8001a8) = lli->next;
  vrg(0xbc8001ac) = lli->ctrl;
  vrg(0xbc8001b0) = lli->status;
  asm volatile("sync");
}

static volatile DMADescriptor *lli_to_me_main_gpu;
static volatile DMADescriptor *lli_to_me_sub_gpu;
static volatile DMADescriptor *lli_to_sc;

typedef u32 (*DmaControl)(const u32, const u32);
inline u32 dmaControlSc2Me(const u32 width, const u32 size)
{
  return DMA_CONTROL_SC2ME(width, size);
}
inline u32 dmaControlMe2Sc(const u32 width, const u32 size)
{
  return DMA_CONTROL_ME2SC(width, size);
}

inline DMADescriptor *dmacplusInitLLIs(const DmaControl dc, const u32 src,
                                       const u32 dst, const u32 size)
{

  constexpr u32 MAX_TRANSFER_SIZE = 4095;
  constexpr struct
  {
    u32 width;
    u32 widthBit;
  } modes[] = {
      {16, 4}, {8, 3}, {4, 2}, {2, 1}, {1, 0}};

  u32 lliCount = 0;
  u32 remaining = size;

  for (int w = 0; w < 4 && remaining > 0; w++)
  {
    u32 block = modes[w].width * MAX_TRANSFER_SIZE;
    lliCount += remaining / block;
    remaining %= block;
  }
  lliCount += (remaining > 0);

  const u32 byteCount = sizeof(DMADescriptor) * lliCount;
  DMADescriptor *const lli = (DMADescriptor *)memalign(64, (byteCount + 63) & ~63);

  u32 i = 0;
  u32 offset = 0;
  remaining = size;

  for (int w = 0; w < 5 && remaining > 0; w++)
  {
    u32 width = modes[w].width;
    u32 widthBit = modes[w].widthBit;
    u32 block = width * MAX_TRANSFER_SIZE;

    if (w < 4 && remaining >= block)
    {
      u32 blockCount = remaining / block;
      for (u32 j = 0; j < blockCount; j++)
      {
        lli[i].src = src + offset;
        lli[i].dst = dst + offset;
        lli[i].ctrl = dc(widthBit, MAX_TRANSFER_SIZE);
        lli[i].status = 1;
        lli[i].next = (i < (lliCount - 1)) ? ((u32)&lli[i + 1]) : 0;
        offset += block;
        remaining -= block;
        i++;
      }
    }
    else if (remaining > 0)
    {
      lli[i].src = src + offset;
      lli[i].dst = dst + offset;
      lli[i].ctrl = dc(0, remaining);
      lli[i].status = 1;
      lli[i].next = 0;
      i++;
      remaining = 0;
    }
  }

  sceKernelDcacheWritebackInvalidateRange((void *)lli, byteCount);
  return lli;
}

static int gather()
{
  cleanChannels();
  dmacplusLLIFromMe(lli_to_sc);
  // waitChannels();
  return 0;
}

static int scatter()
{
  cleanChannels();
  dmacplusLLIFromSc(lli_to_me_main_gpu);
  waitChannels();
  /*cleanChannels();
  dmacplusLLIFromSc(lli_to_me_sub_gpu);
  waitChannelSc();*/
  return 0;
}

__attribute__((noinline, aligned(4))) int cpuInterruptHandler()
{
  extern bool drawn;
  drawn = true;

  kcall(&gather);
  // sceKernelDcacheWritebackInvalidateAll();

  /*const int sz_SCR = 256 * 192 * 4;
  sceKernelDcacheWritebackInvalidateAll();

  u32 *DISP_POINTER = (u32 *)0x04100000;
  //extern void* memcpy_vfpu( void* dst, void* src, unsigned int size );
memcpy(&DISP_POINTER[0], (u32*)&GPU_Screen[0], sz_SCR);*/

  syncDrawSc = 1;

  return -1;
}

void cpuSendInterrupt()
{
  sceKernelDcacheWritebackInvalidateRange((void *)MainScreen.gpu, sizeof(GPU));
  sceKernelDcacheWritebackInvalidateRange((void *)SubScreen.gpu, sizeof(GPU));
  sceKernelDcacheWritebackInvalidateRange((void *)&MMU.LCDCenable, sizeof(MMU.ARM9_VMEM) + ((u32)MMU.ARM9_VMEM - (u32)MMU.LCDCenable));
  asm volatile("sync");

  framesSubmitted = framesSubmitted + 1;
  // kcall(&waitChannels);
  start_draw = 1;
}

// How many frames the CPU has queued that the ME has not finished yet.
// 0 = ME idle/caught up, grows when the ME falls behind.
int meFrameLag()
{
  if (!mem)
    return 0;
  return (int)(framesSubmitted - framesCompleted);
}

inline u32 *meGetUncached32(const u32 size)
{
  static void *_base = nullptr;
  if (!_base)
  {
    _base = memalign(16, size * 4);
    memset(_base, 0, size * 4);
    sceKernelDcacheWritebackInvalidateAll();
    return (u32 *)(UNCACHED_USER_MASK | (u32)_base);
  }
  else if (!size)
  {
    free(_base);
  }
  return nullptr;
}

void initMeEngine()
{
  if (pspSdkLoadStartModule("kcalli.prx", PSP_MEMORY_PARTITION_KERNEL) < 0)
  {
    printf("Failed to load kcalli.prx\n");
    isEmu = true;
    return;
  }
  if (kernelRegisterIntrHandler(0x1f, 1, (void *)(0x80000000 | (u32)cpuInterruptHandler)) < 0)
  {
    printf("Failed to kernelRegisterIntrHandler\n");
    isEmu = true;
    return;
  }

  mem = meGetUncached32(8);

  lli_to_sc = dmacplusInitLLIs(dmaControlMe2Sc, ME_EDRAM_DISPLAY, GE_EDRAM_BASE, (256 * 192 * 4 + 16) & ~16); // Align to 16 bytes so we don't have to use extra transfer
  lli_to_me_main_gpu = dmacplusInitLLIs(dmaControlSc2Me, (u32)MainScreen.gpu, ME_EDRAM_MAIN_GPU, sizeof(GPU));
  // lli_to_me_sub_gpu  = dmacplusInitLLIs(dmaControlSc2Me, (u32)SubScreen.gpu,  ME_EDRAM_SUB_GPU, sizeof(GPU));

  isEmu = false;

  syncDrawSc = 1;
  start_draw = 0;

  kcall(initMe);
  printf("Me Engine started\n");
}