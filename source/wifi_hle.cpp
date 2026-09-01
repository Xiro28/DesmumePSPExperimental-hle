#include "wifi_hle.h"

#include "MD_FIFO.h"
#include "MMU.h"
#include "wifi.h"
#include "NDSSystem.h"

u32 IPCCmdAddr;
u32 SharedMem[2];

u16 Channel;
u32 FifoPtr;

extern void SendIPCReply(u32 service, u32 data, u32 flag = 0);

// The wifi HLE is not wired into this tree's sequencer (no WIFI_HLE event), and
// service 0xA is not in HLE_SERVICE_MASK, so nothing here should run. Stubbed so
// the file still links if it ever is reached.
static void NDS_RescheduleWiFi(u64 cycles) { (void)cycles; }

void WiFIReset()
{
    IPCCmdAddr = 0;
    SharedMem[0] = 0;  // arm7_buf_ptr
    SharedMem[1] = 0;  // status_ptr
    Channel = 0;
    FifoPtr = 0;
    
    _MMU_ARM7_write16(0x027FFCFA, 0x2082);
}

void WifiIPCReply(u16 cmd, u16 status, int numextra=0, u16* extra=nullptr)
{
    _MMU_ARM7_write16(FifoPtr + 0x00, cmd);
    _MMU_ARM7_write16(FifoPtr + 0x02, status);

    SendIPCReply(0xA, FifoPtr, 0);

}

void WifiOnIPCRequest(u32 addr)
{
    IPCCmdAddr = addr;

    u16 cmd = _MMU_ARM7_read16(addr);
    cmd &= ~0x8000;

    //printf("WIFI cmd=%04X addr=%08X SharedMem=%08X %08X FifoPtr=%08X\n", cmd, addr, SharedMem[0], SharedMem[1], FifoPtr);

    if (cmd < 0x2E)
    {
      switch (cmd)
        {
        case 0x0: // init
            {
                SharedMem[0] = _MMU_ARM7_read32(addr+0x4);
                SharedMem[1] = _MMU_ARM7_read32(addr+0x8);
                FifoPtr = _MMU_ARM7_read32(addr+0xC);

                _MMU_ARM7_write32(SharedMem[0], SharedMem[1]);
                _MMU_ARM7_write32(SharedMem[0]+0x4, 0);         // _reserved
                _MMU_ARM7_write32(SharedMem[0]+0x8, FifoPtr);


                _MMU_ARM7_write16(SharedMem[1] + 0x0, 2);  // WmStatus.state = Idle
                _MMU_ARM7_write16(SharedMem[1] + 0x2, 0);  // WmStatus.busy_app_iid
                WifiIPCReply(0x0, 0);
            }
            break;

        case 0x2: // deinit
            {
                _MMU_ARM7_write16(SharedMem[1] + 0x0, 0);  // WmStatus.state = Ready
                _MMU_ARM7_write16(SharedMem[1] + 0x2, 0);  // WmStatus.busy_app_iid
                WifiIPCReply(0x2, 0);
            }
            break;
        

        case 0xA: // start host scan
            {
                Channel = _MMU_ARM7_read16(addr + 0x2);
                u16 max_channel_time = _MMU_ARM7_read16(addr + 0x8);

                _MMU_ARM7_write16(SharedMem[1] + 0x0, 5);  // state = Scan
                _MMU_ARM7_write16(SharedMem[1] + 0x2, 0);

                u32 delay = (u32)max_channel_time * 34418 * 1024;
                NDS_RescheduleWiFi(delay);
            }
            return;

        case 0xB: // stop host scan
        {
            _MMU_ARM7_write16(SharedMem[1] + 0x0, 2);  // state = Idle
            _MMU_ARM7_write16(SharedMem[1] + 0x2, 0);

            _MMU_ARM7_write16(FifoPtr + 0x00, 0xB);  // apiID = StopScan
            _MMU_ARM7_write16(FifoPtr + 0x02, 0);     // errCode = 0
            _MMU_ARM7_write16(FifoPtr + 0x04, 0);
            _MMU_ARM7_write16(FifoPtr + 0x06, 0);
            _MMU_ARM7_write16(FifoPtr + 0x08, 0);     // state = 0 (clear ParentNotFound leftover)
            _MMU_ARM7_write16(FifoPtr + 0x0A, 0);
            _MMU_ARM7_write16(FifoPtr + 0x0C, 0);
            _MMU_ARM7_write16(FifoPtr + 0x0E, 0);
            _MMU_ARM7_write16(FifoPtr + 0x10, 0);
            _MMU_ARM7_write16(FifoPtr + 0x12, 0);
            SendIPCReply(0xA, FifoPtr, 0);
        }
        break;

        case 0xC: // connect to host
        WifiIPCReply(0xC, 0);
        return;

        case 0xE: // start local MP
        WifiIPCReply(0xE, 0);
            break;

        default:
    printf("WIFI HLE: unknown command %04X\n", cmd);
    break;
        }

        _MMU_ARM7_write32(SharedMem[1]+0x4, 0);
    }

    _MMU_ARM7_write16(addr, cmd|0x8000);
}

void wifi_scan_callback() {
    u16 state = _MMU_ARM7_read16(SharedMem[1]);
    //printf("wifi_scan_callback: state=%d\n", state);

    if (state == 5)
    {
        // WMStartScanCallback
        _MMU_ARM7_write16(FifoPtr + 0x00, 0xA);   // api_id = StartScan
        _MMU_ARM7_write16(FifoPtr + 0x02, 0);      // err_code = 0
        _MMU_ARM7_write16(FifoPtr + 0x04, 0);      // wl_cmd_id
        _MMU_ARM7_write16(FifoPtr + 0x06, 0);      // wl_result
        _MMU_ARM7_write16(FifoPtr + 0x08, 4);      // state = ParentNotFound
        // mac_addr[6] già zero
        _MMU_ARM7_write16(FifoPtr + 0x10, Channel);
        _MMU_ARM7_write16(FifoPtr + 0x12, 0);      // link_level

        extern void SendIPCReply(u32 service, u32 data, u32 flag = 0);
        SendIPCReply(0xA, FifoPtr, 0);

        _MMU_ARM7_write32(SharedMem[1] + 0x4, 0);
        _MMU_ARM7_write16(IPCCmdAddr, 0x8000 | 0xA);
    }
}