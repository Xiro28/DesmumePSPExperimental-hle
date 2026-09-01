#pragma once
#include "types.h"
#include "FIFO.h"
#include "armcpu.h"
#include "registers.h"
#include "rtc.h"

// --- IPC register access ----------------------------------------------------
// The -hle tree rewrote FIFO.cpp to keep IPCSYNC/IPCFIFOCNT in four globals,
// replacing every MMU access. This tree keeps upstream's FIFO.cpp, where the
// registers live in MMU memory and are read/written by the CPUs directly.
//
// So these are NOT storage: they are fast typed accessors onto the real
// registers. The HLE code reads and writes them by the same names it used
// before, but every access now goes to the one place both CPUs also see --
// otherwise the HLE would be looking at values nothing ever updates.
//
// 0x40 is the I/O page; 0x180 = IPCSYNC, 0x184 = IPCFIFOCNT.
struct IPCRegRef
{
    u8 proc;        // ARMCPU_ARM9 / ARMCPU_ARM7
    u16 offset;     // 0x180 or 0x184

    operator u16() const
    {
        return T1ReadWord(MMU.MMU_MEM[proc][0x40], offset);
    }

    IPCRegRef& operator=(u16 v)
    {
        T1WriteWord(MMU.MMU_MEM[proc][0x40], offset, v);
        return *this;
    }

    IPCRegRef& operator|=(u16 v) { return *this = (u16)(*this | v); }
    IPCRegRef& operator&=(u16 v) { return *this = (u16)(*this & v); }
};

static IPCRegRef IPCSync9   = { ARMCPU_ARM9, 0x180 };
static IPCRegRef IPCSync7   = { ARMCPU_ARM7, 0x180 };
static IPCRegRef IPCFIFOCnt9 = { ARMCPU_ARM9, 0x184 };
static IPCRegRef IPCFIFOCnt7 = { ARMCPU_ARM7, 0x184 };

void SendIPCReply(u32 service, u32 data, u32 flag = 0);

void OnIPCRequest();

void HLE_Reset();

// Services every pending IPC request the HLE owns. Must be called from outside
// the FIFO code (execHardware does), never from inside IPC_FIFOsend: servicing
// a request sends a reply, which re-enters that function.
void HLE_DrainIPC();
void HLE_IPCSYNC();

void StartScanline(u16 line);

void executeARM7Stuff();

// --- ARM7 hybrid gate (see arm7_hle.cpp) ------------------------------------
// In this tree the real ARM7 core always runs (it is scheduled from exec_cpu at
// boot), so this is a compile-time constant rather than a variable. The dynamic
// wake-on-unhandled-service design was abandoned in the -hle tree for a reason
// that still applies here: NDS_setup() never returns, so a mode switch would
// have to reset the machine from inside a half-executed frame.
//
// It is kept as a named constant because the HLE bails out on it in three
// places -- touch auto-sampling, ARM7 housekeeping and the IPCSYNC handshake --
// which are precisely the jobs the real core does for itself.
static const bool arm7_core_active = true;

// Bitmask of IPC services still answered by the HLE; everything else belongs to
// the real ARM7 core. Touch (0x6), powerman (0x8) and cart save (0xB) are
// excluded deliberately.
extern u32 HLE_SERVICE_MASK;
bool HLE_handles(u32 service);

