// Temporary loop instrumentation. Reports once a second what the scheduler and
// the two CPUs are actually doing, so we stop guessing at the black screen.
#include "types.h"
#include "MMU.h"
#include "armcpu.h"
#include "NDSSystem.h"
#include "FIFO.h"
#include <pspkernel.h>
#include <pspthreadman.h>
#include <stdio.h>

extern s32 arm9, arm7, s32next;

u32 g_dbg_hw = 0, g_dbg_cpu = 0, g_dbg_irq = 0, g_dbg_a9 = 0, g_dbg_a7 = 0;
u32 g_dbg_draw = 0, g_dbg_vbl = 0;
u32 g_dbg_hw_disp = 0, g_dbg_hw_none = 0;
u64 g_dbg_ts = 0; u32 g_dbg_en = 0;
s32 g_dbg_head = -2;

void loop_dbg_report(void)
{
    static u32 last = 0;
    const u32 now = sceKernelGetSystemTimeLow();
    if (now - last < 1000000) return;
    last = now;

    printf("[loop] hw=%u cpu=%u irq=%u a9=%u a7=%u draw=%u vbl=%u | s32next=%d arm9=%d arm7=%d | vc=%u a9pc=%08X a7pc=%08X\n",
        (unsigned)g_dbg_hw, (unsigned)g_dbg_cpu, (unsigned)g_dbg_irq,
        (unsigned)g_dbg_a9, (unsigned)g_dbg_a7,
        (unsigned)g_dbg_draw, (unsigned)g_dbg_vbl,
        (int)s32next, (int)arm9, (int)arm7,
        (unsigned)nds.hw_status.VCount,
        (unsigned)NDS_ARM9.instruct_adr, (unsigned)NDS_ARM7.instruct_adr);
    printf("[loop2] IME9=%u IE9=%08X IF9=%08X | IME7=%u IE7=%08X IF7=%08X | a7pc=%08X a7r0=%08X a7r1=%08X a7r2=%08X a7r3=%08X a7lr=%08X\n",
        (unsigned)MMU.reg_IME[0], (unsigned)MMU.reg_IE[0], (unsigned)MMU.gen_IF<0>(),
        (unsigned)MMU.reg_IME[1], (unsigned)MMU.reg_IE[1], (unsigned)MMU.gen_IF<1>(),
        (unsigned)NDS_ARM7.instruct_adr,
        (unsigned)NDS_ARM7.R[0], (unsigned)NDS_ARM7.R[1], (unsigned)NDS_ARM7.R[2],
        (unsigned)NDS_ARM7.R[3], (unsigned)NDS_ARM7.R[14]);
    g_dbg_hw_disp = g_dbg_hw_none = 0;

    g_dbg_hw = g_dbg_cpu = g_dbg_irq = g_dbg_a9 = g_dbg_a7 = g_dbg_draw = g_dbg_vbl = 0;
}
