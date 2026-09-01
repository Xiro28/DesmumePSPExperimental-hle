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
	uintptr_t ARM9_LCDC[0xA4000/2];
	uintptr_t ARM9_BIOS[0x8000/2];

	static uintptr_t *JIT_MEM[0x4000];
};

extern CACHE_ALIGN JIT_struct JIT;
#define JIT_COMPILED_FUNC(adr, PROCNUM) JIT.JIT_MEM[((adr)&0x0FFFC000)>>14][((adr)&0x00003FFE)>>1]
#define JIT_COMPILED_FUNC_PREMASKED(adr, PROCNUM, ofs) JIT.JIT_MEM[(adr)>>14][(((adr)&0x00003FFE)>>1)+ofs]
#define JIT_COMPILED_FUNC_KNOWNBANK(adr, bank, mask, ofs) JIT.bank[(((adr)&(mask))>>1)+ofs]
// NOTE: JIT_MEM is an ARM9-only table (upstream had JIT_MEM[2][0x4000], one per
// CPU, plus ARM7_BIOS/ERAM/WIRAM/WRAM blocks). Its 0x3X row maps 0x03xxxxxx via
// the ARM9's view of SIWRAM, which is *not* how the ARM7 sees that region -- the
// ARM7's non-shared IWRAM is a different block entirely. So an ARM7 write must
// never be allowed to invalidate through this table: it would clobber unrelated
// ARM9 entries. The ARM7 runs the interpreter, so it needs no invalidation at
// all; PROCNUM 1 is simply excluded.
//
// The ARM7 still has to invalidate the *shared* regions it writes to, because
// the ARM9 compiles code out of them: main RAM (0x02) and SIWRAM (0x03) when
// WRAMCNT maps it to the ARM9. Its own non-shared IWRAM (0x0380xxxx and up) is
// invisible to the ARM9 and must be left alone -- that is the range this table
// would mis-map.
#define JIT_ARM7_SHARED(adr) (((adr) >> 24) == 0x02 || (((adr) >> 24) == 0x03 && !((adr) & 0x00800000)))

#define JIT_MAPPED(adr, PROCNUM) 	(((PROCNUM) == 0 || JIT_ARM7_SHARED(adr)) ? JIT.JIT_MEM[(adr)>>14] : NULL)

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
