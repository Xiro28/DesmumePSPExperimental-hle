#ifndef _PSP_Display_H
#define _PSP_Display_H

#include "../types.h"
#include "FrontEnd.h"
#include "video.h"

#define TEXTURE_FLAGS (GU_TEXTURE_16BIT | GU_VERTEX_16BIT | GU_TRANSFORM_2D)
#define TEXTURE_FLAGS_COLOR (GU_TEXTURE_16BIT | GU_COLOR_5551 | GU_VERTEX_16BIT | GU_TRANSFORM_2D)
#define TEXTURE_FLAGS_COLOR_NT (GU_COLOR_5551 | GU_VERTEX_16BIT | GU_TRANSFORM_2D)
#define TEXTURE_R3D_FLAGS (GU_TEXTURE_16BIT | GU_VERTEX_16BIT | GU_TRANSFORM_3D)

extern unsigned int __attribute__((aligned(64))) gulist[256 * 1024 / 4];

void Init_PSP_DISPLAY_FRAMEBUFF();


void Set_POSX(int pos);
void Set_PAGE(int pos);

extern void (*UpdateScreen)();

void SetupEmuDisplay(bool direct);
void UpdateSingleScreen();
void DrawRom(char* file, f_list* list, int pos,bool reload);

// Split-screen ROM + per-ROM settings menu. Renders one frame using the shared
// selection state below; the input loop in DSEmuGui updates that state.
void DrawRomMenu(char* file, f_list* list, int total);
extern int  menu_romSel;     // selected ROM (absolute list index)
extern int  menu_setSel;     // selected setting (index into settings[])
extern bool menu_focusRight; // false = ROM list focused, true = settings focused
void DrawSettingsMenu(configP* params,int size,int currPos);
void DrawStartUpMenu();
void OSLFINISH();

void SetupDisp_EMU();
void EMU_SCREEN(bool, bool);
// The baseline calls EMU_SCREEN() with no arguments; keep that entry point.
void EMU_SCREEN();
void EMU_SCREEN_Finish(); 
void StartGU_RENDER();
void ENDGU_RENDER();
void SEND_DISP();
void SYNC_GUDISP();

/*OSL_IMAGE* GetUpperFrameBuffer();
OSL_IMAGE* GetLowerFrameBuffer();*/

void* GetPointerFrameBuffer(bool upper);
void* GetPointerUpperFrameBuffer();
void* GetPointerLowerFrameBuffer();

// GU-rendered in-game pause menu (one frame).
// selected: index of the highlighted item.
void DrawPauseMenu(int selected);

// GU-rendered in-game ROM settings editor (one frame).
// sel: index of the highlighted setting (into settings[]).
void DrawRomSettings(int sel, int tab = 0);

// GU-rendered AOT progress screen.
// pct: 0..100, blocks: compiled block count so far.
void DrawAOTProgress(const char* phase, int pct, int blocks);

#endif  // _PSP_VIDEO_H