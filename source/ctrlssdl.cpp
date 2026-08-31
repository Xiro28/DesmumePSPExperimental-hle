/*
	Copyright (C) 2007 Pascal Giard
	Copyright (C) 2007-2011 DeSmuME team

	This file is free software: you can redistribute it and/or modify
	it under the terms of the GNU General Public License as published by
	the Free Software Foundation, either version 2 of the License, or
	(at your option) any later version.

	This file is distributed in the hope that it will be useful,
	but WITHOUT ANY WARRANTY; without even the implied warranty of
	MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
	GNU General Public License for more details.

	You should have received a copy of the GNU General Public License
	along with the this software.  If not, see <http://www.gnu.org/licenses/>.
*/

#include "ctrlssdl.h"
#include "saves.h"
#include "SPU.h"
#include "NDSSystem.h"
#include "melib.h"
#include "PSP/FrontEnd.h"
#include "PSP/video.h"
#ifdef FAKE_MIC
#include "mic.h"
#endif
#include <pspctrl.h>
#include <pspdisplay.h>
#include <pspkernel.h>
#include <pspdebug.h>
#include <psppower.h>

#include "armcpu.h"
#include "GPU.h"
#include "PSP/PSPDisplay.h"

#define NB_KEYS 12

const u32 default_psp_cfg_h[NB_KEYS] =
  { PSP_CTRL_CIRCLE,    //A
	PSP_CTRL_CROSS,     //B
	PSP_CTRL_SELECT|PSP_CTRL_HOME,	//Select
	PSP_CTRL_START,		//Start
	PSP_CTRL_RIGHT,		//Right
	PSP_CTRL_LEFT,		//Left
	PSP_CTRL_UP,		//Up
	PSP_CTRL_DOWN,		//Down
	PSP_CTRL_RTRIGGER,	//R
	PSP_CTRL_LTRIGGER,	//L
	PSP_CTRL_TRIANGLE,  //X
	PSP_CTRL_SQUARE     //Y
  };

mouse_status mouse;

#define MAX_CURSOR_COLORS 4
int ashCursorColors[MAX_CURSOR_COLORS] = { 0x1C34, 0xD2E3, 0x1A54, 0xFFFF };

char achCursor[] =
{
	
	0, 1, 1, 1, 0, 
	1, 1, 1, 1, 1, 
	0, 1, 0, 1, 0, 
	0, 1, 1, 1, 1, 
	1, 0, 0, 0, 1 
};

const int bottom_index = 256*192;

#define VRAM_START 0x4000000
const int top_padding = 48 * 1024;

u8* GetFrameBuffer() {
	return (u8*)(VRAM_START + top_padding);
}


/* Load default joystick and keyboard configurations */
void load_default_config(const u16 kbCfg[]){}

/* Set all buttons at once */
static void set_joy_keys(const u16 joyCfg[]){}

/* Initialize joysticks */
BOOL init_joy( void) {
  return TRUE;
}

/* Unload joysticks */
void uninit_joy( void)
{
 
}

/* Return keypad vector with given key set to 1 */
u16 lookup_joy_key (u16 keyval) { return 0; }

/* Return keypad vector with given key set to 1 */
u16 lookup_key (u16 keyval) { return 0; }

/* Get pressed joystick key */
u16 get_joy_key(int index) { return 0; }

/* Get and set a new joystick key */
u16 get_set_joy_key(int index) { return 0; }

static signed long
screen_to_touch_range( signed long scr, float size_ratio) {
  return (signed long)((float)scr * size_ratio);
}

/* Set mouse coordinates */
static void set_mouse_coord(signed long x,signed long y)
{
  if(x<0) x = 0; else if(x>255) x = 255;
  if(y<0) y = 0; else if(y>192) y = 192;
  mouse.x = x;
  mouse.y = y;
  //mouse.psp_x = mouse.x;
 // mouse.psp_y = mouse.y;
}


/* Retrieve current NDS keypad */
u16 get_keypad( void)
{
  u16 keypad;
  keypad = MMU.ARM7_REG[0x136];
  keypad = (keypad & 0x3) << 10;
#ifdef WORDS_BIGENDIAN
  keypad |= ~(MMU.ARM9_REG[0x130] | (MMU.ARM9_REG[0x131] << 8)) & 0x3FF;
#else
  keypad |= ((u16 *)MMU.ARM9_REG)[0x130>>1] & 0x3FF;
#endif
  return keypad;
}

typedef struct{
	char name[32];
	int var;
}option;

option Options[] = {{"Resume",-1},{"Change Rom",-1},{"Reset Rom",-1},{"Save State",-1},{"Load State",-1},{"Rom Settings",-1},{"Exit",-1}};

u8 curr_index = 0;
u8 N_options = 7;
bool menu_quit = false;

void MenuAction(){
	switch(curr_index){
		case 1:
			ChangeRom(true);
		break;

		case 2:
			ResetRom();
		break;

		case 3:
			savestate_slot(0);
		break;

		case 4:
			loadstate_slot(0);
		break;

		case 5:
		{
			extern void RomSettingsMenu();
			RomSettingsMenu();
		}
		break;

		case 6:
		extern void deinit ();
		deinit();
		return;
	}

	menu_quit = true;
}

void Menu(){
	SceCtrlData pad, oldPad = {0};
	menu_quit = false;
	curr_index = 0;

	while (!menu_quit) {
		DrawPauseMenu(curr_index);

		if (sceCtrlPeekBufferPositive(&pad, 1)) {
			if (pad.Buttons != oldPad.Buttons) {
				if (pad.Buttons & PSP_CTRL_UP)
					curr_index = (curr_index == 0) ? (N_options - 1) : (curr_index - 1);
				if (pad.Buttons & PSP_CTRL_DOWN) {
					++curr_index;
					if (curr_index > N_options - 1) curr_index = 0;
				}
				if (pad.Buttons & PSP_CTRL_CROSS)
					MenuAction();
				if (pad.Buttons & PSP_CTRL_CIRCLE)
					return;
			}
			oldPad = pad;
		}
	}
}

extern bool ARM7_SKIP_HACK;
extern armcpu_t NDS_ARM9;
void
//process_ctrls_event( SDL_Event& event,
process_ctrls_event(u16 &keypad)
{
	  SceCtrlData pad;
	  sceCtrlSetSamplingCycle(0);
	  sceCtrlSetSamplingMode(PSP_CTRL_MODE_ANALOG);
	  sceCtrlPeekBufferPositive(&pad, 1); 

	  //printf("ARM9 PC: %08x, CPSR: %08x\n", NDS_ARM9.R[15], NDS_ARM9.CPSR);

	  mouse.click = FALSE;

	  // Wrap around the DS touch screen (256x192): going past an edge makes the
	  // cursor reappear on the opposite side. Compute in int and use a positive
	  // modulo so negative values wrap correctly (C's % keeps the sign).
	  int mx = mouse.x;
	  int my = mouse.y;

	  if (pad.Lx < 10)  mx -= 4;
	  if (pad.Lx > 245) mx += 4;
	  if (pad.Ly < 10)  my -= 4;
	  if (pad.Ly > 245) my += 4;

	  mx = ((mx % 256) + 256) % 256;   // 0..255
	  my = ((my % 192) + 192) % 192;   // 0..191

	  mouse.x = (u8)mx;
	  mouse.y = (u8)my;

	  set_mouse_coord(mouse.x, mouse.y);

	  const bool select_combo = (pad.Buttons & PSP_CTRL_HOME) && (pad.Buttons & PSP_CTRL_SELECT);

	  if (pad.Buttons & PSP_CTRL_HOME && !select_combo) {
		Menu();
	  }	else if (pad.Buttons & PSP_CTRL_SELECT) {
	  	mouse.click = TRUE;
	  }	else{
			u16	nds_pad	= (0 |
				(((pad.Buttons & default_psp_cfg_h[0]) ? 0 : 0x80) >> 7) |
				(((pad.Buttons & default_psp_cfg_h[1]) ? 0 : 0x80) >> 6) |
				(((pad.Buttons & default_psp_cfg_h[2]) ? 0 : 0x80) >> 5) |
				(((pad.Buttons & default_psp_cfg_h[3]) ? 0 : 0x80) >> 4) |
				(((pad.Buttons & default_psp_cfg_h[4]) ? 0 : 0x80) >> 3) |
				(((pad.Buttons & default_psp_cfg_h[5]) ? 0 : 0x80) >> 2) |
				(((pad.Buttons & default_psp_cfg_h[6]) ? 0 : 0x80) >> 1) |
				(((pad.Buttons & default_psp_cfg_h[7]) ? 0 : 0x80)     ) |
				(((pad.Buttons & default_psp_cfg_h[8]) ? 0 : 0x80) << 1) |
				(((pad.Buttons & default_psp_cfg_h[9]) ? 0 : 0x80) << 2)) ;

			u16 padExt = (((pad.Buttons & default_psp_cfg_h[10])? 0 : 0x80) >> 7) |
				(((pad.Buttons & default_psp_cfg_h[11]) ? 0 : 0x80) >> 6) |
				(((pad.Buttons & default_psp_cfg_h[12]) ? 0 : 0x80) >> 4) |
				((0) << 7) |
				0x0034;
						
			NDS_applyFinalInputDirect(nds_pad, padExt);
	  }
	  
}
