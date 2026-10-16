/* main.c - this file is part of DeSmuME
 *
 * Copyright (C) 2006-2015 DeSmuME Team
 * Copyright (C) 2007 Pascal Giard (evilynux)
 * Used under fair use by the DSonPSP team, 2019
 *
 * This file is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2, or (at your option)
 * any later version.
 *
 * This file is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; see the file COPYING.  If not, write to
 * the Free Software Foundation, Inc., 59 Temple Place - Suite 330,
 * Boston, MA 02111-1307, USA.
 */

#include <stdlib.h>
#include <string.h>
#include <stdio.h>

#include <psppower.h>
#include <pspdebug.h>
#include <pspctrl.h>
#include <pspdisplay.h>
#include <pspsuspend.h>
#include <pspkernel.h>
#include <psprtc.h>

#include"pspdmac.h"

#include "PSP/FrontEnd.h"
#include "PSP/video.h"

#include "NDSSystem.h"
#include "GPU.h"
#include "SPU.h"
#include "sndsdl.h"
#include "sndpsp.h"
#include "ctrlssdl.h"
#include "slot2.h"

#include "render3D.h"
#include "rasterize.h"

#include <unistd.h>
#include "dirent.h"
#include "PSP/vram.h"
#include "PSP/PSPDisplay.h"

#ifdef PROFILE
#include "pspprof.h"
#endif


#include "PSP/pspvfpu.h"

PSP_MODULE_INFO("DesmuME PSP", 0, 3, 0);

PSP_MAIN_THREAD_ATTR(PSP_THREAD_ATTR_USER | PSP_THREAD_ATTR_VFPU);



//From Daedalus
extern "C" {

	void _DisableFPUExceptions();

}

int RAMAMOUNT()
{
	int iStep = 1024;
	int iAmount = 0;
	short shSalir = 0;

	char* pchAux = NULL;

	while (!shSalir)
	{
		iAmount += iStep;
		pchAux = (char*)malloc(iAmount);
		if (pchAux == NULL)
		{
			//No hay memoria libre!!! = There is no free memory!
			iAmount -= iStep;
			shSalir = 1;

		}
		else
		{
			free(pchAux);
			pchAux = NULL;
		}
	}

	return iAmount;
}

char rom_filename[256];
int FPS_Counter = 0;

volatile bool execute = false;

SoundInterface_struct *SNDCoreList[] = {
  &SNDDummy,
  &SNDPSP,
  NULL
};

GPU3DInterface *core3DList[] = {
  &gpu3DNull,
  &gpu3DRasterize,
//  &gpu3DGU,
  NULL
};

const char * save_type_names[] = {
  "Autodetect",
  "EEPROM 4kbit",
  "EEPROM 64kbit",
  "EEPROM 512kbit",
  "FRAM 256kbit",
  "FLASH 2mbit",
  "FLASH 4mbit",
  NULL
};

configured_features my_config;

// Per-frame statistics shown by ShowFPS. Filled by the 3D core / JIT so the
// display code does not have to reach into their internals every frame.
int g_stat_vtx = 0;
int g_stat_poly = 0;
int g_stat_itp_pct = 0;

// Host-side per-frame timings (microseconds), for the stats overlay.


// One NDS frame is 355*263 ARM9 cycles. nds.runCycleCollector holds the
// non-idle cycles of the last 16 frames for each CPU (filled at vblank in
// NDSSystem.cpp), so averaging it gives a load percentage that does not jump
// around frame to frame.
static const s32 kFrameCycles = 1120380;

static int CpuLoadPercent(int cpu)
{
  s64 sum = 0;
  for (int i = 0; i < 16; i++)
    sum += nds.runCycleCollector[cpu][i];
  const s64 pct = (sum * 100) / ((s64)kFrameCycles * 16);
  return (int)(pct < 0 ? 0 : (pct > 999 ? 999 : pct));
}

void ShowFPS(int x,int y){
  // intraFont is invisible under PPSSPP, so this stays on pspDebugScreen. What
  // it does have is per-call colour, so use that instead of one flat line:
  // labels dim, values bright, and the numbers that matter tinted by how good
  // they are. Colours are 0xAABBGGRR.
  const u32 C_LABEL = 0xFF909090;   // grey
  const u32 C_VALUE = 0xFFF0F0F0;   // near-white
  const u32 C_GOOD  = 0xFF60E060;   // green
  const u32 C_WARN  = 0xFF40D0F0;   // amber
  const u32 C_BAD   = 0xFF5050F0;   // red

  const int speed = (FPS_Counter * 100) / 60;
  const u32 c_fps = (FPS_Counter >= 55) ? C_GOOD
                  : (FPS_Counter >= 30) ? C_WARN : C_BAD;

  const int a9 = CpuLoadPercent(0);
  const int a7 = CpuLoadPercent(1);
  const u32 c_a9 = (a9 >= 90) ? C_BAD : (a9 >= 70) ? C_WARN : C_GOOD;
  const u32 c_a7 = (a7 >= 90) ? C_BAD : (a7 >= 70) ? C_WARN : C_GOOD;

  extern u32 g_total_ops, g_itp_ops;
  g_stat_itp_pct = g_total_ops ? (int)((u64)g_itp_ops * 100 / g_total_ops) : 0;
  const u32 c_itp = (g_stat_itp_pct >= 30) ? C_BAD
                  : (g_stat_itp_pct >= 10) ? C_WARN : C_GOOD;

  // Row 1: speed.
  pspDebugScreenSetXY(x, y);
  pspDebugScreenSetTextColor(C_LABEL); pspDebugScreenPrintf("fps ");
  pspDebugScreenSetTextColor(c_fps);   pspDebugScreenPrintf("%2d", FPS_Counter);
  pspDebugScreenSetTextColor(C_LABEL); pspDebugScreenPrintf(" (");
  pspDebugScreenSetTextColor(c_fps);   pspDebugScreenPrintf("%3d%%", speed);
  pspDebugScreenSetTextColor(C_LABEL); pspDebugScreenPrintf(")   arm9 ");
  pspDebugScreenSetTextColor(c_a9);    pspDebugScreenPrintf("%3d%%", a9);
  pspDebugScreenSetTextColor(C_LABEL); pspDebugScreenPrintf("  arm7 ");
  pspDebugScreenSetTextColor(c_a7);    pspDebugScreenPrintf("%3d%%", a7);
  pspDebugScreenSetTextColor(C_LABEL); pspDebugScreenPrintf("    ");

  // Row 2: what the frame cost.
  pspDebugScreenSetXY(x, y + 1);
  pspDebugScreenSetTextColor(C_LABEL); pspDebugScreenPrintf("vtx ");
  pspDebugScreenSetTextColor(C_VALUE); pspDebugScreenPrintf("%5d", g_stat_vtx);
  pspDebugScreenSetTextColor(C_LABEL); pspDebugScreenPrintf("  poly ");
  pspDebugScreenSetTextColor(C_VALUE); pspDebugScreenPrintf("%4d", g_stat_poly);
  pspDebugScreenSetTextColor(C_LABEL); pspDebugScreenPrintf("   itp ");
  pspDebugScreenSetTextColor(c_itp);   pspDebugScreenPrintf("%3d%%", g_stat_itp_pct);
  pspDebugScreenSetTextColor(C_LABEL); pspDebugScreenPrintf("    ");

  pspDebugScreenSetTextColor(0xFFFFFFFF);
}


void PrintfXY(const char* text, int x, int y) {
	pspDebugScreenSetXY(x, y);
	pspDebugScreenPrintf(text);
}

// Called by the ported ctrlssdl when the user picks Exit from the menu.
void deinit()
{
	NDS_DeInit();
	sceKernelExitGame();
}

constexpr int max_framerate[] = {
    1000000 / 1000, 1000000 / 60, 1000000 / 50,
    1000000 / 40,   1000000 / 30, 1000000 / 25,
};
// Called once per frame by the sequencer at the end of vblank. Everything that
// used to live in main()'s while-loop happens here now: input, frame pacing and
// the FPS counter.
// Provided by NDSSystem.cpp / GPU.cpp for the per-frame draw below.
extern void renderScreenFull();
extern volatile u8 __attribute__((aligned(16))) *GPU_Screen;
extern volatile u8 __attribute__((aligned(16))) *GPU_Screen_extra;

void desmume_cycle()
{
	u16 pad = 0;
	static u32 last_fps_timing = 0;
	static u32 previous_time = 0;
	static u32 fps_frame_counter = 0;
	static u8  _frameskip = 0;

	process_ctrls_event(pad);

	/* Update mouse position and click */
	if (!mouse.click)
		NDS_releaseTouchDirect();
	else
		NDS_setTouchPosDirect(mouse.x, mouse.y);

	if (my_config.frameskip != 0)
	{
		if (_frameskip--)
			NDS_SkipNextFrame();
		else
			_frameskip = my_config.frameskip;
	}

	const u32 curr_timing = sceKernelGetSystemTimeLow();

	++fps_frame_counter;
	if (curr_timing - last_fps_timing >= 1000000)
	{
		FPS_Counter = fps_frame_counter;
		fps_frame_counter = 0;
		last_fps_timing = curr_timing;

		static u32 hot_secs = 0;
		if (++hot_secs % 10 == 0)
		{
			extern void jit_hotblock_report();
			jit_hotblock_report();
		}
	}

	// Frame-rate limiting. Advancing previous_time by exactly one frame period
	// (rather than snapping it to now) keeps the pacing steady and lets a late
	// frame be made up for, instead of drifting.
	if (my_config.fps_cap_num > 0)
	{
		const int cap_idx = (my_config.fps_cap_num < (int)(sizeof(max_framerate) / sizeof(max_framerate[0])))
		                    ? my_config.fps_cap_num : 1;
		const int FRAME_TIME_MICROSEC = max_framerate[cap_idx];
		const u32 elapsed = curr_timing - previous_time;

		if (elapsed < (u32)FRAME_TIME_MICROSEC)
		{
			sceKernelDelayThread(FRAME_TIME_MICROSEC - elapsed);
			previous_time += FRAME_TIME_MICROSEC;
		}
		else
			previous_time = curr_timing;
	}


	// Draw the frame. This used to sit in NDS_exec; with the ported sequencer
	// all host-side per-frame work happens here.
	//
	// GPU_RenderLine writes into GPU_Screen/GPU_Screen_extra (memalign'd), but
	// EMU_SCREEN samples VRAM at DISP_POINTER, so copy them across first.
	{
		extern bool isEmu;
		extern void cpuSendInterrupt();
		extern int meFrameLag();

		// SPU mixing used to live inside renderScreenFull(), so it silently
		// stopped running as soon as the ME took over the 2D. It has nothing to
		// do with which CPU renders, so it runs here either way.
		SPU_Emulate_core();

		if (!isEmu)
		{
			// The Media Engine renders the DS 2D and DMAs the result to the GE
			// itself, so the main CPU only kicks it and draws. Closed loop: ask
			// how far behind it is rather than guessing, and skip the kick when
			// it has not caught up -- queueing regardless just makes the lag
			// grow.
			#define ME_MAX_LAG 1
			extern bool drawn;
			static bool drawingInterruptPending = false;

			// Only touch the display when the ME has actually produced a frame.
			// Blitting unconditionally cost a full EMU_SCREEN every iteration
			// even with nothing new to show -- ~17ms per the profile.
			if (drawn || drawingInterruptPending)
			{
				const bool frame_drawn = drawn || drawingInterruptPending;

				// "ITP At Boot" mode: interpret everything until the first frame
				// reaches the screen, then let the JIT compile. Games that wedge
				// during early init but run fine afterwards start with this.
				// Blocks built while interpreting stay interpreted until the code
				// cache is reset, so the switch is gradual rather than instant.
				extern int g_jit_booted;
				if (frame_drawn && !g_jit_booted)
					g_jit_booted = 1;

				// Queue the next draw only when the ME has the budget. If it is
				// behind, defer: the interrupt goes out on a later frame once it
				// catches up. Sending regardless just grows the lag.
				if (meFrameLag() > ME_MAX_LAG)
				{
					drawingInterruptPending = true;
				}
				else
				{
					cpuSendInterrupt();
					drawn = false;
					drawingInterruptPending = false;
				}

				if (frame_drawn)
					EMU_SCREEN(false, false);
			}
		}
		else
		{
			// No ME: render on this CPU. GPU_RenderLine writes into
			// GPU_Screen/GPU_Screen_extra (memalign'd), but EMU_SCREEN samples
			// VRAM at DISP_POINTER, so copy them across first.
			const int sz_SCR = 512 * 192 * 2;
			sceDmacMemcpy((void*)(0x04100000), (void*)&GPU_Screen[0], sz_SCR);
			sceDmacMemcpy((void*)(0x04100000 + sz_SCR), (void*)&GPU_Screen_extra[0], sz_SCR);

			renderScreenFull();
			EMU_SCREEN(false, false);   // draw both; frameskip is applied above
		}
	}

	if (my_config.showfps)
		ShowFPS(0, 3);

	if (my_config.enable_sound)
		SPU_Emulate_user();

	// The ported PSPDisplay presents here: its double-buffer scheme moved
	// sceGuSwapBuffers() out of EMU_SCREEN and into Finish.
	EMU_SCREEN_Finish();
}



void vdDejaLog(char *msg)
{
    FILE *fd;
    fd = fopen("ZZLOG.TXT", "a");
    fprintf(fd, "%s\n", msg);
    fclose(fd);
}

void WriteLog(char* msg)
{
	FILE* fd;
	fd = fopen("debug_log.txt", "a");
	fprintf(fd, "%s\n", msg);
	fclose(fd);
}

void WriteHash(char* msg)
{
	FILE* fd;
	fd = fopen("hashes.txt", "a");
	fprintf(fd, "%s\n", msg);
	fclose(fd);
}

bool audio_inited = false;

#define PSP_AUDIO_SAMPLE_MAX  8192

void EMU_Conf(){

  //ReAddress display to 32bit vram
  pspDebugScreenInitEx((void*)(0x44000000), PSP_DISPLAY_PIXEL_FORMAT_5551, 1);

  // Settings are chosen in the new split-screen ROM menu (DSEmuGui) and are
  // already finalized into my_config there, so the old DoConfig() screen
  // must not run again -- that is the config page that kept appearing.
  FinalizeMainSettings(&my_config);

  NDS_3D_ChangeCore(my_config.Render3D);
  backup_setManualBackupType(my_config.savetype);

  pspDebugScreenClear();

  if (my_config.enable_sound && !audio_inited) {
	  SPU_ChangeSoundCore(SNDCORE_PSP, PSP_AUDIO_SAMPLE_MAX);
	  SPU_SetSynchMode(0, 0 /*CommonSettings.SPU_sync_method*/);
	  audio_inited = true;
  }
  else if (audio_inited && !my_config.enable_sound){
	  SPU_ChangeSoundCore(SNDCORE_DUMMY, 0);
	  audio_inited = false;
  }

  PrintfXY("ROM: ", 0, 1);
  PrintfXY(gameInfo.ROMname, 5, 1);

  char number;
  sprintf(&number, "%1d", my_config.frameskip);
  PrintfXY("Frameskip: ", 55, 1);
  PrintfXY(&number, 65, 1);
}

void ChangeRom(bool reset){

  if (reset){
    NDS_Reset();

    #ifdef PROFILE
      gprof_cleanup();
    #endif

    //Init psp display again
    pspDebugScreenInitEx((void*)(0x44000000), PSP_DISPLAY_PIXEL_FORMAT_5551, 1);
    Init_PSP_DISPLAY_FRAMEBUFF();
  }

  DSEmuGui("",rom_filename);
  EMU_Conf();   // apply what the menu just chose (the fork does this here)
  pspDebugScreenClear();

  int error = NDS_LoadROM(rom_filename);

  if (error < 0) {
     vdDejaLog("ERROR ROM:");
     vdDejaLog(rom_filename);
	   exit(-1);
  }

  execute = true;
}

void ResetRom() {

	NDS_Reset();
	pspDebugScreenClear();

	if (NDS_LoadROM(rom_filename) < 0) {
		exit(-1);
	}

	execute = true;
}

// Frame-rate cap, selectable from the menu (my_config.fps_cap_num):
//   0 = uncapped (frameskip only), then 60, 50, 40, 30, 25 fps.


int main(int argc, char **argv) {

  u32 last_fps_timing = 0;
  u32 previous_time = 0;
  u32 fps_frame_counter = 0;

  /* the firmware settings */
  struct NDS_fw_config_data fw_config;

  scePowerSetClockFrequency(333, 333, 166);

  _DisableFPUExceptions();

  pspDebugScreenInitEx((void*)(0x44000000), PSP_DISPLAY_PIXEL_FORMAT_5551, 1);


  Init_PSP_DISPLAY_FRAMEBUFF();

  // Must precede NDS_Init(): Screen_Init() picks the framebuffer location from
  // isEmu, which only becomes false once the ME is actually up. If kcalli.prx
  // is missing this leaves isEmu true and everything falls back to the CPU.
  extern void initMeEngine();
  initMeEngine();

  NDS_Init();

  /* default the firmware settings, they may get changed later */
  NDS_FillDefaultFirmwareConfigData(&fw_config);

  slot2_Init();

  slot2_Change(NDS_SLOT2_NONE);

  /* Create the dummy firmware */
  NDS_CreateDummyFirmware( &fw_config);

  #ifdef PROFILE
    execute = true;
    NDS_LoadROM("test.nds");
    my_config.Render3D = true;
    my_config.showfps = true;
  #else
    ChangeRom(false);   // this already runs DSEmuGui + EMU_Conf
  #endif

  EMU_SCREEN();
  EMU_SCREEN_Finish();   // present it: the swap lives in Finish now

  u8 _frameskip = my_config.frameskip;

  printf("Ram: %d\n", RAMAMOUNT());

  // The exec loop is generated MIPS (NDS_setup -> exec_hw / exec_cpu /
  // execHardware_interrupts) and never returns; per-frame work that used to sit
  // in this while-loop now runs from desmume_cycle(), which the sequencer calls
  // at the end of vblank.
  extern void NDS_setup();
  NDS_setup();

  NDS_DeInit();
  return 0;
}
