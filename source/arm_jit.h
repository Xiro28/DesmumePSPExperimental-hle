/*	Copyright (C) 2006 yopyop
	Copyright (C) 2011 Loren Merritt
	Copyright (C) 2012-2013 DeSmuME team

	This file is free software: you can redistribute it and/or modify
	it under the terms of the GNU General Public License as published by
	the Free Software Foundation, either version 3 of the License, or
	(at your option) any later version.

	This file is distributed in the hope that it will be useful,
	but WITHOUT ANY WARRANTY; without even the implied warranty of
	MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
	GNU General Public License for more details.

	You should have received a copy of the GNU General Public License
	along with the this software.  If not, see <http://www.gnu.org/licenses/>.
*/

#ifndef ARM_JIT
#define ARM_JIT

#include "types.h"
#ifndef _MSC_VER 
#include <stdint.h>
#endif

typedef u32 (FASTCALL* ArmOpCompiled)();

void arm_jit_reset(bool enable, bool suppress_msg = false);
void arm_jit_close();
void arm_jit_sync();
template<int PROCNUM> u32 arm_jit_compile();

struct JIT_struct 
{
	// only include the memory types that code can execute from
	uintptr_t MAIN_MEM[4*1024*1024/2];
	uintptr_t SWIRAM[0x8000/2];
	uintptr_t ARM9_ITCM[0x8000/2];
	// ARM9_LCDC (code executed straight out of VRAM in LCDC mode, 1.34MB of
	// table) is dropped: BSS is the binding constraint on PSP and executing from
	// VRAM is vanishingly rare. Its rows map to NULL, so such code falls back to
	// the interpreter instead of being compiled -- correct, just not accelerated.
	uintptr_t ARM9_BIOS[0x8000/2];
	// ARM7-only executable regions. Restored (with the per-CPU JIT_MEM below)
	// from upstream so the ARM7 can run under the JIT too. The ARM7 sees
	// 0x03xxxxxx differently from the ARM9 -- shared SIWRAM in the low half,
	// its own non-shared ERAM in the high half -- so it genuinely needs its own
	// table rather than a filter over the ARM9's.
	uintptr_t ARM7_BIOS[0x4000/2];
	uintptr_t ARM7_ERAM[0x10000/2];
	uintptr_t ARM7_WIRAM[0x10000/2];
	// ARM7_WRAM (VRAM allocated as ARM7 work RAM) dropped for the same reason as
	// ARM9_LCDC above -- 512KB of table for a region games rarely execute from.

	static uintptr_t *JIT_MEM[2][0x4000];
};

extern CACHE_ALIGN JIT_struct JIT;
#define JIT_COMPILED_FUNC(adr, PROCNUM) JIT.JIT_MEM[PROCNUM][((adr)&0x0FFFC000)>>14][((adr)&0x00003FFE)>>1]
#define JIT_COMPILED_FUNC_PREMASKED(adr, PROCNUM, ofs) JIT.JIT_MEM[PROCNUM][(adr)>>14][(((adr)&0x00003FFE)>>1)+ofs]
#define JIT_COMPILED_FUNC_KNOWNBANK(adr, bank, mask, ofs) JIT.bank[(((adr)&(mask))>>1)+ofs]
#define JIT_MAPPED(adr, PROCNUM) JIT.JIT_MEM[PROCNUM][(adr)>>14]

extern u32 saveBlockSizeJIT;

// One flag per 16KB main-RAM page (4MB / 16KB = 256): set when a JIT block is
// compiled into that page. DMA into main RAM only needs to invalidate JIT entries
// for pages that actually contain code; data DMAs to never-compiled pages skip it.
// Cleared on arm_jit_reset, set wherever a main-RAM block is registered.
extern u8 jit_mainmem_code[256];

// Experimental JIT optimization toggles (Experimental settings tab). Read during
// compilation; changing any requires a JIT reset (recompile). Default 1 = on.
extern u8 jit_opt_constprop;   // constant propagation pass (ARM)
extern u8 jit_opt_condmerge;   // condition-merge pass (ARM)
extern u8 jit_opt_thumbflags;  // Thumb dead-flag elimination pass
extern u8 jit_opt_fastmem;     // inline main-RAM fast path for word LDR (ARM9)
extern u8 jit_opt_litfold;     // compile PC-relative literal loads as constants

static inline void JIT_MarkCodePage(u32 adr)
{
	if ((adr >> 24) == 0x02)
		jit_mainmem_code[(adr >> 14) & 0xFF] = 1;
}

// True if any 16KB page touched by [adr, adr+bytes) holds compiled code.
static inline bool JIT_MainMemRangeHasCode(u32 adr, u32 bytes)
{
	u32 page  = (adr >> 14) & 0xFF;
	u32 last  = ((adr + bytes - 1) >> 14) & 0xFF;
	u32 count = ((last - page) & 0xFF) + 1;   // wrap-safe page span
	for (u32 i = 0; i < count; i++)
		if (jit_mainmem_code[(page + i) & 0xFF])
			return true;
	return false;
}

#endif
