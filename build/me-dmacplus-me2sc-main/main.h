// m-c/d 2025
#pragma once
#include <psppower.h>
#include <pspdisplay.h>
#include <pspsdk.h>
#include <pspkernel.h>
#include <pspctrl.h>
#include <cstring>
#include <malloc.h>
#include <cmath>
#include "kcall.h"

#define u8  unsigned char
#define u16 unsigned short int
#define u32 unsigned int

#define hwp          volatile u32*
#define hw(addr)     (*((hwp)(addr)))

#define ME_EDRAM_BASE         0
#define GE_EDRAM_BASE         0x04100000
#define UNCACHED_USER_MASK    0x40000000
#define CACHED_KERNEL_MASK    0x80000000
#define UNCACHED_KERNEL_MASK  0xA0000000

#define ME_HANDLER_BASE       0xbfc00000

inline void meDCacheWritebackInvalidAll() {
  asm volatile ("sync");
  for (int i = 0; i < 8192; i += 64) {
    asm("cache 0x14, 0(%0)" :: "r"(i));
    asm("cache 0x14, 0(%0)" :: "r"(i));
  }
  asm volatile ("sync");
}

inline void meHalt() {
  asm volatile(".word 0x70000000");
}

inline u32* meGetUncached32(const u32 size) {
  static void* _base = nullptr;
  if (!_base) {
    _base = memalign(16, size*4);
    memset(_base, 0, size);
    sceKernelDcacheWritebackInvalidateAll();
    return (u32*)(UNCACHED_USER_MASK | (u32)_base);
  } else if (!size) {
    free(_base);
  }
  return nullptr;
}

template<typename T>
inline T xorshift() {
  static T state = 1;
  T x = state;
  x ^= x << 13;
  x ^= x >> 17;
  x ^= x << 5;
  return state = x;
}

inline unsigned short int randInRange(const unsigned short int range) {
  unsigned short int x = xorshift<unsigned int>();
  unsigned int m = (unsigned int)x * (unsigned int)range;
  return (m >> 16);
}

// dmacplus me2sc
#define DMA_CONTROL_SC2SC(width, size)  DMA_CONTROL_CONFIG(1, 1, 0, 0, (width), (size))
#define DMA_CONTROL_SC2ME(width, size)  DMA_CONTROL_CONFIG(0, 0, 1, 0, (width), (size))
#define DMA_CONTROL_ME2SC(width, size)  DMA_CONTROL_CONFIG(0, 0, 0, 1, (width), (size))

#define DMA_CONTROL_CONFIG(idst, isrc, mdst, msrc, width, size) ( \
  (1U << 31) |                  /* terminal count interrupt enable bit */ \
  ((unsigned)(idst) << 27) |    /* dst addr increment ?                */ \
  ((unsigned)(isrc) << 26) |    /* src addr increment ?                */ \
  ((unsigned)(mdst) << 25) |    /* dst AHB master select               */ \
  ((unsigned)(msrc) << 24) |    /* src AHB master select               */ \
  ((unsigned)(width) << 21) |   /* dst transfer width                  */ \
  ((unsigned)(width) << 18) |   /* src transfer width                  */ \
  (1U << 15) |                  /* dst burst size or dst step ?        */ \
  (1U << 12) |                  /* src burst size or src step ?        */ \
  ((unsigned)(size)))           /* transfer size                       */

struct DMADescriptor {
  u32 src;
  u32 dst;
  u32 next;
  u32 ctrl;
  u32 status;
} __attribute__((aligned(16)));


inline void cleanChannel() {
  hw(0xbc8001b0) = 0;
  asm volatile("sync");
}

inline void waitChannel() {
  while (hw(0xbc8001b0)) {
    asm volatile("sync");
    sceKernelDelayThread(10);
  };
}

inline void dmacplusLLIFromMe(volatile const DMADescriptor* const lli) {
  hw(0xbc8001a0) = lli->src;
  hw(0xbc8001a4) = lli->dst;
  hw(0xbc8001a8) = lli->next;
  hw(0xbc8001ac) = lli->ctrl;
  hw(0xbc8001b0) = lli->status;
  asm volatile("sync");
}

inline DMADescriptor* dmacplusInitMe2ScLLI(const u32 src, const u32 dst, const u32 size) {
  constexpr u32 MAX_TRANSFER_SIZE = 4095;
  constexpr struct { u32 width; u32 widthBit; } modes[] = {
    {16, 4}, {8, 3}, {4, 2}, {2, 1}, {1, 0}
  };

  u32 lliCount = 0;
  u32 remaining = size;
  
  for (int w = 0; w < 4 && remaining > 0; w++) {
    u32 block = modes[w].width * MAX_TRANSFER_SIZE;
    lliCount += remaining / block;
    remaining %= block;
  }
  lliCount += (remaining > 0);

  const u32 byteCount = sizeof(DMADescriptor) * lliCount;
  DMADescriptor* const lli = (DMADescriptor*) memalign(64, (byteCount + 63) & ~63);

  u32 i = 0;
  u32 offset = 0;
  remaining = size;

  for (int w = 0; w < 5 && remaining > 0; w++) {
    u32 width = modes[w].width;
    u32 widthBit = modes[w].widthBit;
    u32 block = width * MAX_TRANSFER_SIZE;

    if (w < 4 && remaining >= block) {
      u32 blockCount = remaining / block;
      for (u32 j = 0; j < blockCount; j++) {
        lli[i].src = src + offset;
        lli[i].dst = dst + offset;
        lli[i].ctrl = DMA_CONTROL_ME2SC(widthBit, MAX_TRANSFER_SIZE);
        lli[i].status = 1;
        lli[i].next = (i < lliCount - 1) ? (UNCACHED_USER_MASK | (u32)&lli[i + 1]) : 0;
        offset += block;
        remaining -= block;
        i++;
      }
    }
    else if (remaining > 0) {
      lli[i].src = src + offset;
      lli[i].dst = dst + offset;
      lli[i].ctrl = DMA_CONTROL_ME2SC(0, remaining);
      lli[i].status = 1;
      lli[i].next = 0;
      i++;
      remaining = 0;
    }
  }
  
  sceKernelDcacheWritebackInvalidateRange((void*)lli, byteCount);
  return lli;
}
