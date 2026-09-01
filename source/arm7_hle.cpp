
//LOGIC TAKEN FROM MELONDS : https://github.com/melonDS-emu/melonDS/

#include "MMU.h"
#include "FIFO.h"
#include "arm7_hle.h"
#include "NDSSystem.h"
#include "wifi_hle.h"


namespace Sound_Nitro{
    extern void OnIPCRequest(u32 data);
    extern void Reset();
    extern void Process(u32 param);
}

u32 SM_Command;
u32 SM_DataPos;
u32 SM_Buffer;

u16 PM_Data[16];

int TS_Status;
u16 TS_Data[16];
u16 TS_NumSamples;
u16 TS_SamplePos[4];

u16 FW_Data[16];

u16 Mic_Data[16];

int Sound_Engine;

// Post an IPCSYNC value to the ARM9 on the ARM7's behalf.
//
// The -hle tree poked IPCSync9 and raised the IRQ by hand, which worked because
// its rewritten FIFO.cpp owned those registers outright. Here the registers are
// real, so go through the same path a guest ARM7 write takes -- MMU_IPCSync()
// mirrors the low nibble into the other CPU's register, honours the enable bits
// and raises the IRQ itself. Open-coding it would get the masking subtly wrong
// and would skip the ensata handshake path.
void SendIPCSync(u8 val)
{
    // Write from the ARM7 side: MMU_IPCSync takes the value in the same layout
    // the guest would write, i.e. the data to send to the other CPU in bits 8-11.
    _MMU_write16<ARMCPU_ARM7>(REG_IPCSYNC, (u16)((IPCSync7 & 0x6F00) | ((val & 0xF) << 8)));
}

void SendIPCReply(u32 service, u32 data, u32 flag)
{
    u32 val = (service & 0x1F) | (data << 6) | ((flag & 0x1) << 5);

    // The full-FIFO case is left to IPC_FIFOsend(): upstream's version already
    // sets FIFOERROR and updates both IPCFIFOCNT words. The -hle tree open-coded
    // it here because its rewritten FIFO.cpp kept those registers in globals;
    // duplicating it now would write the real registers twice.
    IPC_FIFOsend(ARMCPU_ARM7, val);
}

// ---------------------------------------------------------------------------
// ARM7 hybrid gate
//
// By default the HLE layer owns the ARM7 and the real core stays frozen. When
// the ARM7 touches something the HLE does not model (see HLE_SERVICE_MASK and
// arm7_core_wake_on_unhandled()), the real core is booted and takes over.
//
// Invariant: for any given IPC service exactly one of the two answers. The HLE
// only replies for services still in HLE_SERVICE_MASK, and once the core is
// awake it is authoritative for everything not in that mask.
// ---------------------------------------------------------------------------

// Services still handled by the HLE. Touchscreen (0x6), powerman (0x8) and
// cart save (0xB) are deliberately left to the real ARM7 core.
// IMPORTANT: only claim a service whose handler ALWAYS replies.
//
// Claiming consumes the message, so the real ARM7 core never sees it. Several
// of the handlers below silently do nothing for inputs they do not model --
// OnIPCRequest_Cart() has an empty else branch, others start with
// `if (flag) break;`. In the -hle tree that merely meant "unimplemented",
// because the HLE was the only responder. Here it means the ARM9 waits forever
// for a reply nobody will send: the message is gone and the core cannot answer
// it either. That is a deadlock, not a missing feature.
//
// So the mask starts at sound only -- the service you actually want offloaded,
// and the one whose handler is complete. Add others only after checking that
// their handler replies on every path.
u32 HLE_SERVICE_MASK =
      (1u << 0x7);  // sound (Nitro HLE)

// Not claimed, and why:
//   0x4 firmware   } handlers bail out on some inputs without replying
//   0x5 RTC        } (see the note above) -- would deadlock the ARM9
//   0x9 mic        }
//   0xD cart       } OnIPCRequest_Cart: empty else branch
//   0x6 touch, 0x8 powerman, 0xB cart save: deliberately the core's job
//   0xA wifi: needs a WIFI_HLE sequencer event this tree does not have

bool HLE_handles(u32 service)
{
    return (HLE_SERVICE_MASK >> (service & 0x1F)) & 1u;
}

// The real ARM7 core is always enabled: it runs from the cart entry point at
// boot and the HLE only answers the services still listed in HLE_SERVICE_MASK.
// (A deferred NDS_Reset() to switch modes mid-run is not possible here --
// NDS_setup() never returns, so a reset from the frame loop would come back into
// a half-executed frame.)

void HLE_Reset(){
    // IPCSYNC/IPCFIFOCNT are the real MMU registers now (see arm7_hle.h), and
    // MMU_clearMem() -> IPC_FIFOinit() already resets them. Writing them here
    // would clobber that, and the FIFOENABLE bit is the guest's to set: the -hle
    // tree forced it on because its ARM7 was HLE and nobody else would, but here
    // the real ARM7 firmware code does it itself.
    SM_Command = 0;
    SM_DataPos = 0;
    SM_Buffer = 0;

    Sound_Engine = -1;

    memset(PM_Data, 0, sizeof(PM_Data));

    memset(Mic_Data, 0, sizeof(Mic_Data));

    TS_Status = 0;
    memset(TS_Data, 0, sizeof(TS_Data));
    TS_NumSamples = 0;
    memset(TS_SamplePos, 0, sizeof(TS_SamplePos));

    memset(FW_Data, 0, sizeof(FW_Data));

    Sound_Nitro::Reset();
    WiFIReset();

    // arm7_core_active is a compile-time constant here (see arm7_hle.h): the
    // real ARM7 core always runs, so there is nothing to switch on.
}

uint8_t buffer[8192];

void OnIPCRequest_CartSave(u32 data)
{
    if (SM_DataPos == 0)
        SM_Command = data;

    //printf("OnIPCRequest_CartSave: %08X\n", SM_Command);

    switch (SM_Command)
    {
    case 0:
        if (SM_DataPos == 1)
        {
            SM_Buffer = data;
            SendIPCReply(0xB, 0x1, 1);
            SM_DataPos = 0;
            return;
        }
        break;

    case 2: // identify savemem
        if (MMU_new.backupDevice.info.mem_size == 0)
        {
            const u32 save_param = _MMU_read32<ARMCPU_ARM7>(SM_Buffer+0x04);
            const u32 save_size_shift = (save_param >> 8) & 0xFF;
            MMU_new.backupDevice.info.mem_size = 1 << save_size_shift;
        }
        SendIPCReply(0xB, 0x1, 1);
        SM_DataPos = 0;
        return;

    case 6: // read
        {
            const u32 offset = _MMU_read32<ARMCPU_ARM7>(SM_Buffer+0x0C);
            const u32 dst = _MMU_read32<ARMCPU_ARM7>(SM_Buffer+0x10);
            const u32 len = _MMU_read32<ARMCPU_ARM7>(SM_Buffer+0x14);

            u32 memlen = MMU_new.backupDevice.info.mem_size - 1;
            
            s32 remaining = len;
            u32 currOffset = offset;
            u32 currDst = dst;

            while (remaining > 0)
            {
                u32 toRead = (remaining > sizeof(buffer)) ? sizeof(buffer) : remaining;
                MMU_new.backupDevice.loadToBuffer(buffer, currOffset & memlen, toRead);

                for (u32 i = 0; i < toRead; i++)
                {
                    _MMU_write08<ARMCPU_ARM7>(currDst++, buffer[i]);
                }

                currOffset += toRead;
                remaining -= toRead;
            }


            SendIPCReply(0xB, 0x1, 1);
            
            SM_DataPos = 0;  
            return;
        }
        break;

    case 7:
    case 8: // write
        {
            const u32 src = _MMU_read32<ARMCPU_ARM7>(SM_Buffer+0x0C);
            const u32 offset = _MMU_read32<ARMCPU_ARM7>(SM_Buffer+0x10);
            const u32 len = _MMU_read32<ARMCPU_ARM7>(SM_Buffer+0x14);


            const u32 memlen = MMU_new.backupDevice.info.mem_size - 1;
            for (u32 i = 0; i < len; i++)
            {
                const char val = _MMU_read08<ARMCPU_ARM7>(src + i);
                MMU_new.backupDevice.writeByte((offset + i) & memlen, val);
            }

            _MMU_write16<ARMCPU_ARM7>(SM_Buffer + 0x02, 0);

            
            SendIPCReply(0xB, 0x1, 1);
            SM_DataPos = 0;
            return;
        }
        break;

    case 9: // verify
        {
            _MMU_write16<ARMCPU_ARM7>(SM_Buffer + 0x02, 0);
            SendIPCReply(0xB, 0x1, 1);
            SM_DataPos = 0;
            return;
        }
        break;

    default:
        printf("SAVEMEM: unknown cmd %08X\n", SM_Command);
        break;
    }

    SM_DataPos++;
}

void OnIPCRequest_Cart(u32 data)
{
    if ((data & 0x3F) == 1)
    {
        // TODO other shito?

        SendIPCReply(0xD, 0x1);
    }
    else
    {
        // do something else
    }
    /*if (data & 0x1)
    {
        // init

        SendIPCReply(0xD, 0x1);
    }*/
}

void Touchscreen_Sample()
{
    u32 ts = (((u32)_MMU_read16<ARMCPU_ARM7>(0x027FFFAA)) | ((u32)_MMU_read16<ARMCPU_ARM7>(0x027FFFAC) << 16));
    
    if (!nds.hw_status.Touching)
    {
        ts &= 0xFE000000;
        ts |= 0x06000000;
    }
    else
    {
        ts &= 0xF9000000;
        ts |= (nds.scr_touchX & 0xFFF);
        ts |= ((nds.scr_touchY & 0xFFF) << 12);
        ts |= 0x01000000;
    }

    _MMU_write16<ARMCPU_ARM7>(0x027FFFAA, ts & 0xFFFF);
    _MMU_write16<ARMCPU_ARM7>(0x027FFFAC, ts >> 16);
}

void OnIPCRequest_Touchscreen(u32 data)
{
    if (data & (1<<25))
    {
        memset(TS_Data, 0, sizeof(TS_Data));
    }

    TS_Data[(data >> 16) & 0xF] = data & 0xFFFF;

    if (!(data & (1<<24))) return;

    switch (TS_Data[0] >> 8)
    {
    case 0: // manual sampling
        {
            Touchscreen_Sample();
            SendIPCReply(0x6, 0x03008000);
        }
        break;

    case 1: // setup auto sampling
        {
            if (TS_Status != 0)
            {
                SendIPCReply(0x6, 0x03008103);
                break;
            }

            // samples per frame
            u8 num = TS_Data[0] & 0xFF;
            if (num == 0 || num > 4)
            {
                SendIPCReply(0x6, 0x03008102);
                break;
            }

            // offset in scanlines for first sample
            u16 offset = TS_Data[1];
            if (offset >= 263)
            {
                SendIPCReply(0x6, 0x03008102);
                break;
            }

            TS_Status = 1;

            TS_NumSamples = num;
            for (int i = 0; i < num; i++)
            {
                u32 ypos = (offset + ((i * 263) / num)) % 263;
                TS_SamplePos[i] = ypos;
            }

            TS_Status = 2;
            SendIPCReply(0x6, 0x03008100);
        }
        break;

    case 2: // stop autosampling
        {
            if (TS_Status != 2)
            {
                SendIPCReply(0x6, 0x03008203);
                break;
            }

            TS_Status = 3;

            // TODO CHECKME
            TS_NumSamples = 0;

            TS_Status = 0;
            SendIPCReply(0x6, 0x03008200);
        }
        break;

    case 3: // manual sampling but with condition (TODO)
        {
            Touchscreen_Sample();
            SendIPCReply(0x6, 0x03008300);
        }
        break;

    default:
        printf("unknown TS request %08X\n", data);
        break;
    }
}

void OnIPCRequest_Powerman(u32 data)
{
    extern void FASTCALL MMU_writeToSPIData(u16 val);

    if (data & (1<<25))
    {
        memset(PM_Data, 0, sizeof(PM_Data));
    }

    PM_Data[(data >> 16) & 0xF] = data & 0xFFFF;

    if (!(data & (1<<24))) return;

    u32 cmd = (PM_Data[0] >> 8) - 0x60;
    //printf("PM CMD %d %04X %04X\n", cmd, PM_Data[0], PM_Data[1]);
    switch (cmd)
    {
    case 1:
        MMU_writeToSPIData(PM_Data[0]); //?
        SendIPCReply(0x8, 0x03008000);
    break;
    case 3: // utility
        {
            switch (PM_Data[1] & 0xFF)
            {
            case 1: // power LED: steady
                MMU.powerMan_Reg[0] &= ~0x10;
                break;
            case 2: // power LED: fast blink
                MMU.powerMan_Reg[0] |= 0x30;
                break;
            case 3: // power LED: slow blink
                MMU.powerMan_Reg[0] &= ~0x20;
                MMU.powerMan_Reg[0] |= 0x10;
                break;
            case 4: // lower backlights on
                MMU.powerMan_Reg[0] |= 0x04;
                break;
            case 5: // lower backlights off
                MMU.powerMan_Reg[0] &= ~0x04;
                break;
            case 6: // upper backlights on
                MMU.powerMan_Reg[0] |= 0x08;
                break;
            case 7: // upper backlights off
                MMU.powerMan_Reg[0] &= ~0x08;
                break;
            case 8: // backlights on
                MMU.powerMan_Reg[0] |= 0x0C;
                break;
            case 9: // backlights off
                MMU.powerMan_Reg[0] &= ~0x0C;
                break;
            case 10: // sound amp on
                MMU.powerMan_Reg[0] |= 0x01;
                break;
            case 11: // sound amp off
                MMU.powerMan_Reg[0] &= ~0x01;
                break;
            case 12: // sound mute on
                MMU.powerMan_Reg[0] |= 0x02;
                break;
            case 13: // sound mute off
                MMU.powerMan_Reg[0] &= ~0x02;
                break;
            case 14: // shutdown
                MMU.powerMan_Reg[0] &= ~0x01;
                MMU.powerMan_Reg[0] |= 0x40;
                //NDS::Stop();
                break;
            case 15: // ????
                MMU.powerMan_Reg[0] &= ~0x40;
                break;
            }

            SendIPCReply(0x8, 0x0300E300);
        }
        break;

    case 4: // write register
        {
            u8 addr = PM_Data[0] & 0xFF;
            u8 val = PM_Data[1] & 0xFF;

            extern void MMU_writePowerMan(u8 val, bool hold);

            MMU_writePowerMan(addr & 0x7F, true);
            MMU_writePowerMan(val, false);

            SendIPCReply(0x8, 0x03008000 | (((PM_Data[1] + 0x70) & 0xFF) << 8));
        }
        break;

    case 5: // read register
        {
            u8 addr = PM_Data[0] & 0xFF;
            u8 reg = (addr & 0x7F) & 0x7;

            extern u8 MMU_readPowerMan();   // was declared u32 in -hle: mismatched the u8 definition

            extern void MMU_writePowerMan(u8 val, bool hold);

            MMU_writePowerMan((addr & 0x7F) | 0x80, true);
            MMU_writePowerMan(0, false);

            u8 ret = MMU_readPowerMan();

            //printf("PM read reg=%02X val=%02X\n", addr & 0x7F, ret);

            SendIPCReply(0x8, 0x03008000 | (ret & 0xFF) | (((PM_Data[1] + 0x70) & 0xFF) << 8));
        }
        break;

    case 6:
        {
            // TODO

            SendIPCReply(0x8, 0x03008000 | (((PM_Data[1] + 0x70) & 0xFF) << 8));
        }
        break;

    default:
        printf("unknown PM request %08X, %08X, %08X\n", cmd, PM_Data[0], PM_Data[1]);
        break;
    }
}

void RTC_Read(u8 reg, u32 addr, u32 len)
{
    u8 old_cmd = rtc.cmd;

    rtc.cmd = reg;

    rtcRecv();

    static u8 prev_sec = 0xFF;
    u8 cur_sec = (len >= 7) ? rtc.data[6] : ((len == 3) ? rtc.data[2] : 0);
    if (cur_sec != prev_sec) {
        //printf("RTC read: sec=%02X (BCD)\n", cur_sec);
        prev_sec = cur_sec;
    }

    for (u32 i = 0; i < len; i++)
    {
        //printf("%d: %d\n", i, rtc.data[i]);
        _MMU_write08<ARMCPU_ARM7>(addr+i, rtc.data[i]);
    }
    //rtc.cmd = old_cmd;
}

void OnIPCRequest_RTC(u32 data)
{
    u32 cmd = (data >> 8) & 0x7F;

    if ((cmd >= 2 && cmd <= 15) ||
        (cmd >= 26 && cmd <= 34) ||
        (cmd >= 42))
    {
        SendIPCReply(0x5, 0x8001 | (cmd << 8));
        return;
    }

    switch (cmd)
    {
    case 0x10: // read date and time
        RTC_Read(0x4, 0x027FFDE8, 7);
        SendIPCReply(0x5, 0x9000);
        break;

    case 0x11: // read date
        RTC_Read(0x4, 0x027FFDE8, 4);
        SendIPCReply(0x5, 0x9100);
        break;

    case 0x12: // read time
        RTC_Read(0x6, 0x027FFDE8, 3);
        SendIPCReply(0x5, 0x9200);
        break;

    default:
        printf("HLE: unknown RTC command %02X (%08X)\n", cmd, data);
        break;
    }
}

void OnIPCRequest_Firmware(u32 data)
{
    if (data & (1<<25))
    {
        memset(FW_Data, 0, sizeof(FW_Data));
    }

    FW_Data[(data >> 16) & 0xF] = data & 0xFFFF;

    if (!(data & (1<<24))) return;

    u32 cmd = (FW_Data[0] >> 8) - 0x20;
    switch (cmd)
    {
    case 0: // write enable
        MMU.fw.write_enable = true;
        SendIPCReply(0x4, 0x0300A000);
        break;

    case 1: // write disable
        MMU.fw.write_enable = false;
        SendIPCReply(0x4, 0x0300A100);
        break;

    case 2: // read status register
        {
            u32 addr = ((FW_Data[0] & 0xFF) << 24) | (FW_Data[1] << 8) | ((FW_Data[2] >> 8) & 0xFF);
            if (addr < 0x02000000 || addr >= 0x02800000)
            {
                SendIPCReply(0x4, 0x0300A202);
                break;
            }

            _MMU_write08<ARMCPU_ARM7>(addr, MMU.fw.write_enable ? 0x02 : 0x00);

            SendIPCReply(0x4, 0x0300A200);
        }
        break;

    case 3: // firmware read
        {
            u32 addr = (FW_Data[4] << 16) | FW_Data[5];
            if (addr < 0x02000000 || addr >= 0x02800000)
            {
                SendIPCReply(0x4, 0x0300A302);
                break;
            }

            u32 src = ((FW_Data[0] & 0xFF) << 16) | FW_Data[1];
            u32 len = (FW_Data[2] << 16) | FW_Data[3];

            for (u32 i = 0; i < len; i++)
            {
                u8 val = MMU.fw.data[src & (0x20000 -1)];
                _MMU_write08<ARMCPU_ARM7>(addr, val);
                src++;
                addr++;
            }


            SendIPCReply(0x4, 0x0300A300);
        }
        break;

    case 5: // firmware write
        {
            u32 addr = (FW_Data[3] << 16) | FW_Data[4];
            if (addr < 0x02000000 || addr >= 0x02800000)
            {
                SendIPCReply(0x4, 0x0300A502);
                break;
            }

            u32 dst = ((FW_Data[0] & 0xFF) << 16) | FW_Data[1];
            u32 len = FW_Data[2];

            for (u32 i = 0; i < len; i++)
            {
                u8 val =_MMU_read08<ARMCPU_ARM7>(addr);
                MMU.fw.data[dst & (0x20000 -1)] = val;
                dst++;
                addr++;
            }


            SendIPCReply(0x4, 0x0300A500);
        }
        break;

    default:
        printf("unknown FW request %08X (%04X)\n", data, FW_Data[0]);
        break;
    }
}

void OnIPCRequest_Sound(u32 data)
{
    if (Sound_Engine == -1)
    {
        if (data >= 0x02000000)
        {
            Sound_Engine = 0;
            Sound_Nitro::Reset();
        }
        else
        {
            Sound_Engine = 1;
            printf("HLE: Peach Engine\n");
            //Sound_Peach::Reset();
        }
    }

    if (Sound_Engine == 0) return Sound_Nitro::OnIPCRequest(data);

    // Sound_Engine == 1 is the Peach engine, whose HLE was never written (the
    // call below is commented out upstream too). Claiming the message and then
    // not replying would hang the ARM9, so hand this game's sound back to the
    // real ARM7 core by dropping the service from the mask. The message we are
    // holding is lost, but the ARM9 retries -- one dropped request beats a
    // deadlock, and every later one goes to the core.
    HLE_SERVICE_MASK &= ~(1u << 0x7);
    printf("HLE: Peach sound engine not implemented -- handing sound to the ARM7 core\n");
   // if (Sound_Engine == 1) return Sound_Peach::OnIPCRequest(data);
}

void OnIPCRequest_Mic(u32 data)
{
    if (data & (1<<25))
    {
        memset(Mic_Data, 0, sizeof(Mic_Data));
    }

    Mic_Data[(data >> 16) & 0xF] = data & 0xFFFF;

    if ((data & (1<<24)) == 0) return;

    u32 cmd = (Mic_Data[0] >> 8) - 0x40;
    switch (cmd)
    {
    case 0: // sampling?
        {
            // TODO

            SendIPCReply(0x9, 0x0300C000);
        }
        break;

    default:
        //printf("unknown mic request %08X\n", data);
        SendIPCReply(0x9, 0x0300C000);
        break;
    }
}

// Hybrid HLE ownership split. Called from IPC_FIFOsend() right after the ARM9
// posts a message, BEFORE the real ARM7 core gets a chance to read it.
//
// Peeks at the head of the ARM9's FIFO without consuming it, and only calls
// OnIPCRequest() -- which does consume -- when the service belongs to the HLE.
// For every other service the message is left exactly as it was, so the ARM7
// reads it normally and the HLE never interferes.
//
// Set by IPC_FIFOsend() when the ARM9 posts a message; drained by
// HLE_DrainIPC() from execHardware, outside the FIFO code itself.
volatile bool HLE_IPCPending = false;

// Returns true if the HLE took the message.
static bool HLE_TryClaimIPC()
{
    void OnIPCRequest();

    if (ipc_fifo[ARMCPU_ARM9].size == 0)
        return false;

    // Both FIFOENABLE bits must be set. Claiming reads through
    // IPC_FIFOrecv(ARMCPU_ARM7), which checks the ARM7's own enable bit and
    // returns 0 WITHOUT consuming when it is clear -- OnIPCRequest() then bails
    // on val==0 while we report success, and HLE_DrainIPC()'s loop spins on the
    // same message forever. That was the black screen with a live fps counter.
    if (!(IPCFIFOCnt9 & IPCFIFOCNT_FIFOENABLE))
        return false;
    if (!(IPCFIFOCnt7 & IPCFIFOCNT_FIFOENABLE))
        return false;

    // Peek: head is the oldest queued word, which is the one the ARM7 would read.
    const u32 val = ipc_fifo[ARMCPU_ARM9].buf[ipc_fifo[ARMCPU_ARM9].head];
    const u32 service = val & 0x1F;

    if (!HLE_handles(service))
        return false;   // belongs to the real ARM7 core -- leave it alone

    const u8 before = ipc_fifo[ARMCPU_ARM9].size;
    OnIPCRequest();

    // Only report success if the message was actually consumed, so a request
    // that could not be serviced can never turn into an infinite drain loop.
    return ipc_fifo[ARMCPU_ARM9].size < before;
}

// Drain every message the HLE owns, called from execHardware -- i.e. between
// instructions, never from inside the FIFO code. Stops at the first message
// that belongs to the real ARM7 core: the FIFO is ordered, so claiming past it
// would reorder that CPU's requests.
void HLE_DrainIPC()
{
    if (!HLE_IPCPending)
        return;

    HLE_IPCPending = false;

    // Bounded by the FIFO depth (16): HLE_TryClaimIPC() only returns true when
    // it actually consumed a message, so this cannot spin, and the cap means a
    // future bug there degrades into a dropped tick rather than a hang.
    for (int i = 0; i < 16 && HLE_TryClaimIPC(); i++)
        ;   // keep going while the head is ours
}

void OnIPCRequest()
{
    u32 val = IPC_FIFOrecv(ARMCPU_ARM7);

    if (val == 0)
    {
        return;
    }

    
    //printf("%x %x\n", val, IPCFIFO7.Peek());

    u32 service = val & 0x1F;
    u32 data = val >> 6;
    bool flag = ((val >> 5) & 0x1) != 0;

    // Second line of defence. HLE_TryClaimIPC() already checked the mask before
    // consuming, so this should never fire -- but OnIPCRequest() consumes from
    // the FIFO, so if it were ever reached for a service the real ARM7 owns it
    // would silently swallow that CPU's message. Keep the check.
    if (!HLE_handles(service))
        return;

    switch (service)
    {
    case 0x4: // firmware
        if (flag) break;
        OnIPCRequest_Firmware(data);
        break;

    case 0x5: // RTC
        if (flag) break;
        OnIPCRequest_RTC(data);
        break;

    case 0x6: // touchscreen
        if (flag) break;
        OnIPCRequest_Touchscreen(data);
        break;

    case 0x7: // sound
        OnIPCRequest_Sound(data);
        break;

    case 0x8: // powerman
        if (flag) break;
        OnIPCRequest_Powerman(data);
        break;

    case 0x9: // mic
        if (flag) break;
        OnIPCRequest_Mic(data);
        break;

    case 0xA: // wifi
        if (flag) break;
        printf("HLE: wifi request %08X\n", data);
        WifiOnIPCRequest(data);
        //Wifi::OnIPCRequest(data);
        break;

    case 0xB: // cart savemem
        if (!flag) break;
        OnIPCRequest_CartSave(data);
        break;

     case 0xC:
        if (data == 0x1000)
        {
            // TODO: stop/reset hardware

            SendIPCReply(0xC, 0x1000);
        }
        break;

    case 0xD: // cart
        OnIPCRequest_Cart(data);
        break;

    case 0xF:
        //if (data == 0x10000)
        {
            SendIPCReply(0xF, data);
        }
        break;

    default:
        printf("HLE: unknown IPC request %08X service=%02X data=%08X flag=%d\n", val, service, data, flag);
        break;
    }
}

void StartScanline(u16 line)
{
    // Touchscreen auto-sampling is HLE service 0x6, which the real ARM7 core now
    // owns (see HLE_SERVICE_MASK). Replying here as well would inject a reply the
    // ARM7 never asked for.
    if (arm7_core_active || !HLE_handles(0x6))
        return;

    for (int i = 0; i < TS_NumSamples; i++)
    {
        if (line == TS_SamplePos[i])
        {
            Touchscreen_Sample();
            SendIPCReply(0x6, 0x03009000 | i);
            break;
        }
    }
}

#define HW_WRAM_END                 0x03800000
#define HW_PRV_WRAM                 0x03800000
#define HW_PRV_WRAM_END             0x03810000

#define HW_PRV_WRAM_SIZE            (HW_PRV_WRAM_END-HW_PRV_WRAM)

#define HW_PRV_WRAM_SYSRV_SIZE      0x40

#define HW_PRV_WRAM_SYSRV           (HW_PRV_WRAM + HW_PRV_WRAM_SIZE - HW_PRV_WRAM_SYSRV_SIZE)

#define HW_INTR_CHECK_BUF           (HW_PRV_WRAM_SYSRV + 0x38)

void executeARM7Stuff(){
    // Once the real ARM7 core runs it does this housekeeping itself; doing both
    // would fight over the shared WRAM locations below.
    if (arm7_core_active)
        return;

    extern u16 get_keypad();
	// 0x027FFFA8 holds ext key state: bits 10=X, 11=Y, 13=lid
	// get_keypad already returns regular keypad in bits 0-9 + X/Y in bits 10-11
	_MMU_write16<ARMCPU_ARM7>(0x027FFFA8, get_keypad() & 0x2C00);
   
   //u32 val = _MMU_ARM7_read32(HW_INTR_CHECK_BUF);

   //printf("ARM7: %08X\n", val);

   u32 fram_counter = _MMU_read32<ARMCPU_ARM7>(0x27FFC3C);
    _MMU_write32<ARM7>(0x27FFC3C, fram_counter+1);

    if (nds.is_twl_sdk()){
        bool sync = _MMU_read08<ARMCPU_ARM7>(0x2FFFFF0) & 0x1;
        _MMU_write08<ARM7>(0x2FFFFF1, !sync);
        _MMU_write08<ARM7>(0x2FFFFF2, 1);
        _MMU_write08<ARM7>(0x2FFFFF3, 1);
    }

   /* _MMU_write16<ARM7>(0x02FFFC20, 0);
    _MMU_write16<ARM7>(0x02FFFC22, 0);
    _MMU_write16<ARM7>(0x02FFFC24, 2);
    _MMU_write16<ARM7>(0x02FFFC26, 8);
    _MMU_write16<ARM7>(0x02FFFC28, 4);
    _MMU_write16<ARM7>(0x04000006, 0);*/

    //_MMU_write32<ARM7>(0x02067340, fram_counter+1);
    //_MMU_write32<ARM9>(0x02067340, fram_counter+1);

   //Sound_Nitro::Process(1);
}

void HLE_IPCSYNC(){
    // The boot handshake is itself an HLE service: when the real ARM7 core is
    // running it performs its own handshake, and answering as well makes the
    // two fight over IPCSync (the ARM7 posts a value the HLE state machine does
    // not know, gets no reply, and retries forever).
    if (arm7_core_active)
        return;

    u8 val = IPCSync7 & 0xF;

    printf("[HLE] HLE_IPCSYNC: IPCSync7=0x%04X IPCSync9=0x%04X val=%d\n",
        IPCSync7, IPCSync9, val);

    if (val < 5)
    {
        SendIPCSync(val+1);
    }
    else if (val == 5)
    {
        SendIPCSync(0);

        printf("[HLE] IPCSYNC handshake complete! Writing ARM7 ready flag\n");
        // presumably ARM7-side ready flags for each IPC service
        _MMU_write32<ARMCPU_ARM7>(0x027FFF8C, 0x3FFF0);
    }
    else
    {
        printf("[HLE] HLE_IPCSYNC: unexpected val=%d, ignoring\n", val);
    }
}
