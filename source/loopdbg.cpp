// Temporary loop instrumentation. Reports once a second what the scheduler and
// the two CPUs are actually doing, so we stop guessing at the black screen.
#include "types.h"
#include "MMU.h"
#include "armcpu.h"
#include "NDSSystem.h"
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
    printf("[loop2] fired=%u ts=%u timer=%u diff=%d en=%u none=%u\n",
        (unsigned)g_dbg_hw_disp, (unsigned)g_dbg_ts, (unsigned)nds_timer,
        (int)((s64)g_dbg_ts - (s64)nds_timer), (unsigned)g_dbg_en,
        (unsigned)g_dbg_hw_none);
    g_dbg_hw_disp = g_dbg_hw_none = 0;

    g_dbg_hw = g_dbg_cpu = g_dbg_irq = g_dbg_a9 = g_dbg_a7 = g_dbg_draw = g_dbg_vbl = 0;
}
