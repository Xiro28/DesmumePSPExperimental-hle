// Code taken from https://github.com/mcidclan/me-sample-interrupt/

#pragma once
#include <psppower.h>
#include <pspdisplay.h>
#include <pspsdk.h>
#include <pspkernel.h>
#include <pspctrl.h>
#include <malloc.h>
#include <cstring>
#include "kcall.h"

#include "NDSSystem.h"
#include "GPU.h"
#include"PSP/PSPDisplay.h"

#define u8  unsigned char
#define u16 unsigned short int
#define u32 unsigned int

#define vrp                  volatile u32*
#define vrg(addr)            (*((volatile u32*)(addr)))
#define uncached_vrg(addr)   (*((volatile u32*)(0x40000000 | addr)))
#define uncached(var)        (*((volatile u32*)(0x40000000 | ((u32)&var))))

#define me_section_size ((&__stop__me_section - &__start__me_section + 3) & ~3)
#define _meLoop      vrg((0xbfc00040 + me_section_size))


#define setStatus(status) \
   asm volatile( \
       "mtc0   %0, $12\n" \
       "sync" \
       : \
       : "r" (status) \
   )


static inline void meDCacheWritebackInvalidAll() {
  asm volatile ("sync");
  for (int i = 0; i < 8192; i += 64) {
    asm("cache 0x14, 0(%0)" :: "r"(i));
    asm("cache 0x14, 0(%0)" :: "r"(i));
  }
  asm volatile ("sync");
}

static inline void meDCacheWritebackAll() {
  asm volatile ("sync");
  for (int i = 0; i < 8192; i += 64) {
    asm("cache 0x1a, 0(%0)" :: "r"(i));
    asm("cache 0x1a, 0(%0)" :: "r"(i));
  }
  asm volatile ("sync");
}

static inline void meDCacheInvalidAll() {
  asm volatile ("sync");
  for (int i = 0; i < 8192; i += 64) {
    asm("cache 0x19, 0(%0)" :: "r"(i));
    asm("cache 0x19, 0(%0)" :: "r"(i));
  }
  asm volatile ("sync");
}

static inline void meDCacheWritebackInvalidRange(const u32 addr, const u32 size) {
  asm volatile("sync");
  for (volatile u32 i = addr; i < addr + size; i += 64) {
    asm volatile(
      "cache 0x1b, 0(%0)\n"
      "cache 0x1b, 0(%0)\n"
      :: "r"(i)
    );
  }
  asm volatile("sync");
}

static inline void meDCacheInvalidRange(const u32 addr, const u32 size) {
  asm volatile("sync");
  for (volatile u32 i = addr; i < addr + size; i += 64) {
    asm volatile(
      "cache 0x19, 0(%0)\n"
      "cache 0x19, 0(%0)\n"
      :: "r"(i)
    );
  }
  asm volatile("sync");
}

static inline void meDCacheWritebackRange(const u32 addr, const u32 size) {
  asm volatile("sync");
  for (volatile u32 i = addr; i < addr + size; i += 64) {
    asm volatile(
      "cache 0x1a, 0(%0)\n"
      "cache 0x1a, 0(%0)\n"
      :: "r"(i)
    );
  }
  asm volatile("sync");
}

static inline u32 getlocalUID() {
  u32 unique;
  asm volatile(
    "sync\n"
    "mfc0 %0, $22\n"
    "sync"
    : "=r" (unique)
  );
  return (unique + 1) & 3;
  // reads processor id from cp0 register $22
  // 0 = main cpu
  // 1 = me
}
extern bool ShouldSkip2dFrame();
extern bool ShouldSkip3dFrame();

static volatile u32 _meExit = 0;
static inline void meExit() {
  uncached(_meExit) = 1;
}