#include "../arm7_hle.h"
#include <pspkernel.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <stdarg.h>
#include <errno.h>
#include <string.h>
#include <time.h>
#include <pspdebug.h>
#include <math.h>
#include <pspctrl.h>
#include <pspgu.h>
#include "FrontEnd.h"
#include <psprtc.h>
#include <pspdisplay.h>
#include <strings.h> // Provides strcasecmp
#include <pspiofilemgr.h> // sceIoMkdir for per-ROM config storage
#include <unistd.h>       // getcwd
#include <dirent.h>       // opendir/readdir for the ROM file list
#include "PSPDisplay.h"

// JIT-vs-interpreter differential test (source/jit_difftest.cpp), run from the
// ROM-selection menu via TRIANGLE.
// g_jit_cycles_in_reg and the jit_opt_* toggles are defined in arm_jit.cpp;
// jit_difftest_run_force() lives in jit_difftest.cpp. Both are now ported, so
// the placeholder definitions that used to sit here are gone.
extern "C" void jit_difftest_run_force();

//---------------------------------------------------------------------
// SETTINGS structure (SETTINGS / MAX_SETTINGS now declared in FrontEnd.h so
// the ROM-menu renderer can read settings[] for the split-screen panel).
//---------------------------------------------------------------------
SETTINGS settings[MAX_SETTINGS];
int totalSettings = 0;		// Count of the main settings
int totalSettingsDebug = 0; // Count of main settings plus debug settings

SETTINGS expSettings[MAX_EXP_SETTINGS];  // Experimental optimizations tab
int totalExp = 0;
int g_arm9_mode_idx = -1;   // rows in expSettings[] that hold a 0..3 mode
int g_arm7_mode_idx = -1;
int g_arm7_hle_idx  = -1;   // row holding the 0..2 ARM7 HLE mode

extern "C" const int g_jit_cycles_in_reg;   // reflects compile-time JIT_CYCLES_IN_REG
extern unsigned char jit_opt_constprop;     // runtime JIT optimization toggles (arm_jit.h)
extern unsigned char jit_opt_condmerge;
extern unsigned char jit_opt_thumbflags;
extern unsigned char jit_opt_fastmem;
extern unsigned char jit_opt_litfold;

//---------------------------------------------------------------------
// Global variables used by the menus
//---------------------------------------------------------------------
int selposconfig = 0;
int selposDebug = 0;
bool changed = false;

//---------------------------------------------------------------------
// Helper function: Adds a new setting into the global settings array.
//---------------------------------------------------------------------
int addSetting(SETTINGS *setArr, int index, const char *name, const char *desc, int defaultValue)
{
	if (index >= MAX_SETTINGS)
		return index;
	strncpy(setArr[index].name, name, sizeof(setArr[index].name) - 1);
	setArr[index].name[sizeof(setArr[index].name) - 1] = '\0';
	strncpy(setArr[index].description, desc, sizeof(setArr[index].description) - 1);
	setArr[index].description[sizeof(setArr[index].description) - 1] = '\0';
	setArr[index].value = defaultValue;
	return index + 1;
}

//---------------------------------------------------------------------
// Helper function: Map a language number to its text representation.
// Mapping: 0 = JAP, 1 = ENG, 2 = FRE, 3 = GER, 4 = ITA, 5 = SPA, 6 = CHI, 7 = RES
//---------------------------------------------------------------------
const char *GetLanguageText(int lang)
{
	static const char *langMap[8] = {"JAP", "ENG", "FRE", "GER", "ITA", "SPA", "CHI", "RES"};
	if (lang < 0 || lang >= 8)
		return "UNK";
	return langMap[lang];
}

const char *GetFPSCapText(int val){
	static const char *capMap[] = {"No cap", "60", "50", "40", "30", "25"};
	if (val < 0 || val >= 6)
		return "UNK";
	return capMap[val];
}

//---------------------------------------------------------------------
// Initialize the main configuration settings and map them to the
// corresponding fields in the configured_features structure.
//---------------------------------------------------------------------
void InitMainSettings(configured_features *params)
{
	int c = 0;
	c = addSetting(settings, c, "Screen SWAP", "Swap the screen buffers", 0);
	params->swap = settings[0].value;

	c = addSetting(settings, c, "Show Touch Cursor", "Display the touch cursor", 0);
	params->cur = settings[1].value;

	c = addSetting(settings, c, "Show FPS", "Display frame rate on screen", 1);
	params->showfps = settings[2].value;

	c = addSetting(settings, c, "Enable Audio", "Enable sound output", 0);
	params->enable_sound = settings[3].value;

	c = addSetting(settings, c, "Frameskip", "Skip frames to improve performance", 0);
	params->frameskip = settings[4].value;

	// The language setting will be displayed as text.
	c = addSetting(settings, c, "Language", "Select firmware language", 1);
	params->firmware_language = settings[5].value;

	c = addSetting(settings, c, "Render 3D", "Enable 3D rendering", 1);
	params->Render3D = settings[6].value;

	//c = addSetting(settings, c, "Fast ME Rendering", "Enable fast ME rendering", 0);
	//params->FastMERendering = settings[7].value;

	c = addSetting(settings, c, "FPS Cap", "Limit the frames per second", 1);
	params->fps_cap_num = settings[7].value;

	//c = addSetting(settings, c, "Use cached Interpreter", "For a better stability but sometimes slower", 0);
	//params->cached_interpreter = settings[8].value;

	c = addSetting(settings, c, "3D UI depth fix", "OFF=normal depth order. ON=fix coplanar 2D-UI games (Tetris DS menus) by ordering full-screen backgrounds first. Wrong for some games, so opt-in.", 0);
	params->sort_3d = settings[8].value;

	c = addSetting(settings, c, "3D always on top", "Draw 3D always on top of 2D scene", 0);
	params->_3d_always_on_top = settings[9].value;

	c = addSetting(settings, c, "AOT Pre-compile", "Compile all code before game starts (saved for next run)", 0);
	params->aot_precompile = settings[10].value;

	totalSettings = c;

	// Experimental JIT flags default ON, so boot paths that never open the
	// Experimental tab still get the optimizations. (InitExperimentalSettings
	// overrides from expSettings[] when the menu/per-ROM config is loaded.)
	params->exp_idle_loop   = true;
	params->exp_block_link  = true;
	params->exp_cycles_reg  = g_jit_cycles_in_reg;
	params->exp_icache_align = true;
	params->exp_early_term  = true;
}

void FinalizeMainSettings(configured_features *params)
{
	params->swap = settings[0].value;
	params->cur = settings[1].value;
	params->showfps = settings[2].value;
	params->enable_sound = settings[3].value;
	params->frameskip = settings[4].value;
	params->firmware_language = settings[5].value;
	params->Render3D = settings[6].value;
	//params->FastMERendering = settings[7].value;
	params->fps_cap_num = settings[7].value;
	params->sort_3d = settings[8].value;
	params->_3d_always_on_top = settings[9].value;
	params->aot_precompile = settings[10].value;
}

//---------------------------------------------------------------------
// Experimental JIT optimizations tab.
//---------------------------------------------------------------------
void InitExperimentalSettings(configured_features *params)
{
	int c = 0;
	c = addSetting(expSettings, c, "Idle Loop Skip", "Detect idle loops and skip their cycles. Big speedup; rarely causes timing glitches.", 1);
	params->exp_idle_loop = expSettings[0].value;

	c = addSetting(expSettings, c, "Block Linking", "Chain directly between compiled blocks (inline dispatch). Faster; disable to debug.", 1);
	params->exp_block_link = expSettings[1].value;

	c = addSetting(expSettings, c, "Cycles In Register", "Hold the cycle counter in a CPU register (compile-time JIT_CYCLES_IN_REG; rebuild to change).", g_jit_cycles_in_reg);
	params->exp_cycles_reg = expSettings[2].value;

	c = addSetting(expSettings, c, "Const Propagation", "Fold compile-time-constant ALU results into immediates. Faster code; disable to debug.", 1);
	c = addSetting(expSettings, c, "Cond Merging", "Skip re-evaluating flags for consecutive same-condition ops. Disable to debug.", 1);
	c = addSetting(expSettings, c, "Thumb Flag Elim", "Drop dead flag computations in Thumb blocks. Disable to debug.", 1);
	c = addSetting(expSettings, c, "Fast Memory", "Inline main-RAM word loads (skip the MMU C call). VERIFY with difftest; default off.", 0);

	c = addSetting(expSettings, c, "ICache Align", "Align each compiled block to a 64-byte ICache line. Reduces cache-line splits on the PSP Allegrex (16KB 2-way ICache).", 1);
	params->exp_icache_align = expSettings[7].value;

	c = addSetting(expSettings, c, "Suffix Reuse", "Stop block early when the remaining instructions are already compiled; jump into that block. Reduces code-cache usage and recompilation.", 1);
	params->exp_early_term = expSettings[8].value;

	c = addSetting(expSettings, c, "Literal Folding", "Compile PC-relative literal-pool loads as constants (no memory read). Disable to debug.", 1);

	// Per-CPU JIT mode. Not a toggle: 0..4, cycled with LEFT/RIGHT.
	//   0 Full JIT      everything compiled (default)
	//   1 Half JIT      ARM compiled, Thumb interpreted
	//   2 ITP At Boot   interpret until the first frame, then compile
	//   3 JIT ITP       blocks/lookup/chaining active, all instructions interpreted
	//   4 ITP til DTCM  interpret until the ARM9 relocates DTCM, then compile
	c = addSetting(expSettings, c, "ARM9 JIT Mode", "LEFT/RIGHT. Full JIT: compile everything (fastest). Half JIT: compile ARM, interpret Thumb. ITP At Boot: interpret until the first frame, then compile. JIT ITP: keep block caching but interpret every instruction (slowest, most compatible). ITP til DTCM: interpret until the ARM9 relocates DTCM, then compile.", 0);
	g_arm9_mode_idx = c - 1;
	c = addSetting(expSettings, c, "ARM7 JIT Mode", "Same four modes as ARM9. The ARM7 runs on the interpreter in this build, so this only takes effect once its JIT is enabled.", 0);
	g_arm7_mode_idx = c - 1;

	// ARM7 HLE service ownership. Applies immediately, including mid-game.
	c = addSetting(expSettings, c, "ARM7 HLE Mode", "LEFT/RIGHT. Sound Only: the HLE answers sound IPC, the real ARM7 core does the rest (default, the only audited setting). Core+Snd+Tch: the HLE also answers touch (breaks the stylus in Layton). Core Only: the HLE claims nothing. HLE Only: full HLE and the real ARM7 core stops being executed (the actual speed win; modes 0-2 keep the core running).", 0);
	g_arm7_hle_idx = c - 1;
	HLE_ApplyMode(expSettings[g_arm7_hle_idx].value);

	totalExp = c;

	extern int g_jit_mode[2];
	g_jit_mode[0] = expSettings[g_arm9_mode_idx].value;
	g_jit_mode[1] = expSettings[g_arm7_mode_idx].value;

	jit_opt_constprop  = expSettings[3].value;
	jit_opt_condmerge  = expSettings[4].value;
	jit_opt_thumbflags = expSettings[5].value;
	jit_opt_fastmem    = expSettings[6].value;
	jit_opt_litfold    = expSettings[9].value;
}

void FinalizeExperimental(configured_features *params)
{
	params->exp_idle_loop   = expSettings[0].value;
	params->exp_block_link  = expSettings[1].value;
	params->exp_cycles_reg  = expSettings[2].value;
	params->exp_icache_align = expSettings[7].value;
	params->exp_early_term  = expSettings[8].value;

	jit_opt_constprop  = expSettings[3].value;
	jit_opt_condmerge  = expSettings[4].value;
	jit_opt_thumbflags = expSettings[5].value;
	jit_opt_fastmem    = expSettings[6].value;
	jit_opt_litfold    = expSettings[9].value;

	// JIT mode takes effect on the next block compiled; already-compiled blocks
	// keep the code they were built with until the cache is reset.
	extern int g_jit_mode[2];
	if (g_arm9_mode_idx >= 0) g_jit_mode[0] = expSettings[g_arm9_mode_idx].value % 5;
	if (g_arm7_mode_idx >= 0) g_jit_mode[1] = expSettings[g_arm7_mode_idx].value % 5;

	// HLE ownership is a plain mask rewrite; no reset, effective next message.
	if (g_arm7_hle_idx >= 0) HLE_ApplyMode(expSettings[g_arm7_hle_idx].value % 4);
}



//---------------------------------------------------------------------
// Initialize the debug/display (GPU layer) settings.
// These settings are appended to the main settings.
//---------------------------------------------------------------------
void InitDisplayParams(configured_features *params)
{
	if (changed)
		return;

	int c = totalSettings;
	const char *namesTop[5] = {"BG0 TOP", "BG1 TOP", "BG2 TOP", "BG3 TOP", "OBJ TOP"};
	const char *namesBottom[5] = {"BG0 Bottom", "BG1 Bottom", "BG2 Bottom", "BG3 Bottom", "OBJ Bottom"};

	for (int j = 0; j < 5; j++)
	{
		char desc[128];
		sprintf(desc, "Display flag for %s", namesTop[j]);
		c = addSetting(settings, c, namesTop[j], desc, 1);
		params->gpuLayerEnabled[0][j] = settings[c - 1].value;
	}
	for (int j = 0; j < 5; j++)
	{
		char desc[128];
		sprintf(desc, "Display flag for %s", namesBottom[j]);
		c = addSetting(settings, c, namesBottom[j], desc, 1);
		params->gpuLayerEnabled[1][j] = settings[c - 1].value;
	}
	totalSettingsDebug = c;
}

//---------------------------------------------------------------------
// Update the GPU/display settings from the settings array into
// the configured_features structure.
//---------------------------------------------------------------------
void ChangeValueDebug(configured_features *params)
{
	int c = totalSettings;
	for (int j = 0; j < 5; j++, c++)
	{
		params->gpuLayerEnabled[0][j] = settings[c].value;
	}
	for (int j = 0; j < 5; j++, c++)
	{
		params->gpuLayerEnabled[1][j] = settings[c].value;
	}
	totalSettingsDebug = c;
}

//---------------------------------------------------------------------
// Display the main configuration settings. For the "Language" setting,
// show its mapped text (JAP, ENG, etc.) instead of a number.
//---------------------------------------------------------------------
void DisplayConfigParms()
{
	int yPos = 8;
	for (int i = 0; i < totalSettings; i++, yPos++)
	{
		// Insert section headers if desired.
		if (i == 3)
		{
			pspDebugScreenSetXY(5, yPos++);
			pspDebugScreenSetTextColor(0xff00ff00); // Green for sections
			pspDebugScreenPrintf("[ MISC ]\n");
		}
		else if (i == 0)
		{
			pspDebugScreenSetXY(5, yPos++);
			pspDebugScreenSetTextColor(0xff00ff00);
			pspDebugScreenPrintf("[ GUI ]\n");
		}
		pspDebugScreenSetXY(5, yPos);
		if (strcmp(settings[i].name, "Language") == 0)
		{
			// Display language value as text.
			if (i == selposconfig)
			{
				pspDebugScreenSetTextColor(0xffffff00);
				pspDebugScreenPrintf("> %-12s : %s\n", settings[i].name, GetLanguageText(settings[i].value));
			}
			else
			{
				pspDebugScreenSetTextColor(0xffffffff);
				pspDebugScreenPrintf("  %-12s : %s\n", settings[i].name, GetLanguageText(settings[i].value));
			}
		}
		else if (strcmp(settings[i].name, "FPS Cap") == 0)
		{
			// Display language value as text.
			if (i == selposconfig)
			{
				pspDebugScreenSetTextColor(0xffffff00);
				pspDebugScreenPrintf("> %-12s : %s\n", settings[i].name, GetFPSCapText(settings[i].value));
			}
			else
			{
				pspDebugScreenSetTextColor(0xffffffff);
				pspDebugScreenPrintf("  %-12s : %s\n", settings[i].name, GetFPSCapText(settings[i].value));
			}
		}
		else
		{
			// All other settings display as an integer.
			if (i == selposconfig)
			{
				pspDebugScreenSetTextColor(0xffffff00);
				pspDebugScreenPrintf("> %-12s : %d\n", settings[i].name, settings[i].value);
			}
			else
			{
				pspDebugScreenSetTextColor(0xffffffff);
				pspDebugScreenPrintf("  %-12s : %d\n", settings[i].name, settings[i].value);
			}
		}
	}

	// Leave a blank line and display the description of the currently selected setting.
	yPos += 5;
	pspDebugScreenSetXY(5, yPos);
	pspDebugScreenSetTextColor(0xffffffff); // Green for the description text
	pspDebugScreenPrintf("%s\n", settings[selposconfig].description);
}

//---------------------------------------------------------------------
// Display the Experimental optimizations tab.
//---------------------------------------------------------------------
void DisplayExperimentalParms(int sel)
{
	int yPos = 8;
	pspDebugScreenSetXY(5, yPos++);
	pspDebugScreenSetTextColor(0xff00ff00);
	pspDebugScreenPrintf("[ JIT EXPERIMENTAL ]\n");

	// The two JIT-mode rows show a mode name; everything else is ON/OFF.
	static const char *kJitModeNames[5] = {
	    "0 Full JIT",      // everything compiled                    (fastest)
	    "1 JIT ARM only",  // ARM compiled, Thumb interpreted
	    "2 ITP->1st frame",// interpret, compile after the first frame
	    "3 All Interp",    // block cache kept, every op interpreted (slowest)
	    "4 ITP->DTCM",     // interpret, compile once DTCM is moved
	};

	// Named after what the real ARM7 CORE does, because that is what decides
	// both speed and compatibility. In 0-2 the core runs and the only question
	// is who answers IPC; only 3 stops executing it, which is the mode that
	// actually gets the old HLE-only speed back.
	static const char *kHleModeNames[4] = {
	    "0 Core+Snd HLE",  // core runs; HLE answers sound only  (default, audited)
	    "1 Core+Snd+Tch",  // core runs; HLE answers sound + touch
	    "2 Core Only",     // core runs; HLE answers nothing
	    "3 No Core (HLE)", // core STOPPED; HLE answers everything (fastest)
	};

	for (int i = 0; i < totalExp; i++, yPos++)
	{
		const bool is_mode = (i == g_arm9_mode_idx || i == g_arm7_mode_idx);
		const bool is_hle  = (i == g_arm7_hle_idx);
		const char *val = is_mode
		    ? kJitModeNames[expSettings[i].value % 5]
		    : is_hle
		    ? kHleModeNames[expSettings[i].value % 4]
		    : (expSettings[i].value ? "ON" : "OFF");

		pspDebugScreenSetXY(5, yPos);
		pspDebugScreenSetTextColor((i == sel) ? 0xffffff00 : 0xffffffff);
		pspDebugScreenPrintf("%s %-18s : %s\n", (i == sel) ? ">" : " ",
		                     expSettings[i].name, val);
	}

	yPos += 4;
	pspDebugScreenSetXY(5, yPos);
	pspDebugScreenSetTextColor(0xffffffff);
	pspDebugScreenPrintf("%s\n", expSettings[sel].description);
}

//---------------------------------------------------------------------
// Display the debug (GPU/display) settings.
//---------------------------------------------------------------------
void DisplayDebugParms()
{
	for (int c = totalSettings; c < totalSettingsDebug; c++)
	{
		if (selposDebug == c)
			pspDebugScreenSetTextColor(0x0f00ffff); // Yellow highlight
		else
			pspDebugScreenSetTextColor(0xff00f55f); // Red
		pspDebugScreenPrintf("  %s : %d\n", settings[c].name, settings[c].value);
	}
}

//---------------------------------------------------------------------
// Debug menu loop for adjusting GPU/display settings.
//---------------------------------------------------------------------
void Debug(configured_features *params)
{
	int done = 0;
	SceCtrlData pad, oldPad = {0};
	pspDebugScreenSetXY(0, 0);
	selposDebug = totalSettings; // Start at first debug setting

	while (!done)
	{
		sceDisplayWaitVblankStart();
		pspDebugScreenSetTextColor(0xffffffff);
		pspDebugScreenSetXY(0, 3);
		pspDebugScreenPrintf("\n\n  Debug:\n\n  Disable Rendering: BG0 - BG1 - BG2 - BG3 - OBJ\n\n\n");

		ChangeValueDebug(params);
		DisplayDebugParms();

		if (sceCtrlPeekBufferPositive(&pad, 1))
		{
			if (pad.Buttons != oldPad.Buttons)
			{
				if (pad.Buttons & PSP_CTRL_LEFT)
					settings[selposDebug].value = 0;
				else if (pad.Buttons & PSP_CTRL_RIGHT)
				{
					settings[selposDebug].value = 1;
					changed = true;
				}
				if (pad.Buttons & PSP_CTRL_START)
				{
					pspDebugScreenSetTextColor(0xffffffff);
					done = 1;
					return;
				}
				if (pad.Buttons & PSP_CTRL_UP)
				{
					selposDebug--;
					if (selposDebug < totalSettings)
						selposDebug = totalSettings;
				}
				if (pad.Buttons & PSP_CTRL_DOWN)
				{
					selposDebug++;
					if (selposDebug >= totalSettingsDebug - 1)
						selposDebug = totalSettingsDebug - 1;
				}
			}
			oldPad = pad;
		}
	}
	pspDebugScreenSetTextColor(0xffffffff);
}

//---------------------------------------------------------------------
// Per-ROM settings persistence.
//
// Each ROM gets its own config file under CONFIG/<rombase>.cfg, storing the
// menu settings as "name=value" lines. Saving/loading by NAME (not by index or
// by dumping the struct) keeps old config files compatible when settings are
// added, removed, or reordered. Files live next to the executable.
//---------------------------------------------------------------------

// Extract the bare ROM file name (no directory, no extension) from a full path.
static void RomBaseName(const char *romPath, char *out, size_t outSize)
{
	const char *slash = strrchr(romPath, '/');
	const char *name  = slash ? slash + 1 : romPath;
	strncpy(out, name, outSize - 1);
	out[outSize - 1] = '\0';
	char *dot = strrchr(out, '.');
	if (dot) *dot = '\0';
}

// Build "<cwd>/CONFIG/<rombase>.cfg" and ensure the CONFIG directory exists.
static void RomConfigPath(const char *romPath, char *out, size_t outSize)
{
	char app_path[160];
	getcwd(app_path, sizeof(app_path));

	char dir[256];
	snprintf(dir, sizeof(dir), "%s/CONFIG", app_path);
	sceIoMkdir(dir, 0777);   // no-op if it already exists

	char base[128];
	RomBaseName(romPath, base, sizeof(base));
	snprintf(out, outSize, "%s/%s.cfg", dir, base);
}

// Save the current menu settings[] to the ROM's config file.
void SaveRomConfig(const char *romPath)
{
	if (!romPath || !romPath[0]) return;

	char cfgPath[300];
	RomConfigPath(romPath, cfgPath, sizeof(cfgPath));

	FILE *f = fopen(cfgPath, "w");
	if (!f) return;
	for (int i = 0; i < totalSettings; i++)
		fprintf(f, "%s=%d\n", settings[i].name, settings[i].value);
	for (int i = 0; i < totalExp; i++)
		fprintf(f, "%s=%d\n", expSettings[i].name, expSettings[i].value);
	fclose(f);
}

// Load saved values into settings[] (matched by name). Settings absent from the
// file keep the defaults already placed by InitMainSettings. Returns true if a
// config file was found.
bool LoadRomConfig(const char *romPath)
{
	if (!romPath || !romPath[0]) return false;

	char cfgPath[300];
	RomConfigPath(romPath, cfgPath, sizeof(cfgPath));

	FILE *f = fopen(cfgPath, "r");
	if (!f) return false;

	char line[160];
	while (fgets(line, sizeof(line), f))
	{
		char *eq = strchr(line, '=');
		if (!eq) continue;
		*eq = '\0';
		const char *key = line;
		int val = atoi(eq + 1);
		bool matched = false;
		for (int i = 0; i < totalSettings; i++)
			if (strcmp(settings[i].name, key) == 0) { settings[i].value = val; matched = true; break; }
		if (!matched)
			for (int i = 0; i < totalExp; i++)
				if (strcmp(expSettings[i].name, key) == 0) { expSettings[i].value = val; break; }
	}
	fclose(f);

	// Full JIT is forced on: a stale .cfg from the mode-bisection sessions must
	// not silently put a CPU back on the interpreter.
	if (g_arm9_mode_idx >= 0) expSettings[g_arm9_mode_idx].value = 0;
	if (g_arm7_mode_idx >= 0) expSettings[g_arm7_mode_idx].value = 0;
	return true;
}

//---------------------------------------------------------------------
// Main configuration menu loop. Special handling is provided for the
// "Language" (displayed as text) and "Frameskip" settings.
//---------------------------------------------------------------------
void DoConfig(configured_features *params)
{
	int done = 0;
	SceCtrlData pad, oldPad = {0};

	pspDebugScreenSetXY(0, 0);
/*
#ifdef LOWRAM
	settings[5].value = 1;
	settings[6].value = 1;
#else
	settings[1].value = 1;
	settings[2].value = 1;
	settings[4].value = 1;
	settings[6].value = 1;
	settings[8].value = 1;
#endif
*/
	int langposconfig = 0;
	int frameposconfig = 0;

	InitMainSettings(params);

	// Load this ROM's saved settings over the defaults (matched by name).
	extern char rom_filename[256];
	LoadRomConfig(rom_filename);

	while (!done)
	{
		sceDisplayWaitVblankStart();
		pspDebugScreenSetXY(5, 2);
		pspDebugScreenSetTextColor(0xff00ff00);
		pspDebugScreenPrintf("[ Emulator Settings ]\n");
		pspDebugScreenSetTextColor(0xffffffff);

		DisplayConfigParms();

		if (sceCtrlPeekBufferPositive(&pad, 1))
		{
			if (pad.Buttons != oldPad.Buttons)
			{
				switch (pad.Buttons)
				{
					case PSP_CTRL_LEFT:
						if (strcmp(settings[selposconfig].name, "Language") == 0)
						{
							settings[selposconfig].value = std::max(0, (settings[selposconfig].value - 1) % 8);
						}
						else if (strcmp(settings[selposconfig].name, "Frameskip") == 0)
						{
							settings[selposconfig].value = std::max(0, (settings[selposconfig].value - 1) % 10);
						}
						else if (strcmp(settings[selposconfig].name, "FPS Cap") == 0)
						{
							settings[selposconfig].value = std::max(0, (settings[selposconfig].value - 1) % 6);
						}
						else
						{
							settings[selposconfig].value = 0;
						}
					break;

					case PSP_CTRL_RIGHT:
						if (strcmp(settings[selposconfig].name, "Language") == 0)
						{
							settings[selposconfig].value = (settings[selposconfig].value + 1) % 8;
						}
						else if (strcmp(settings[selposconfig].name, "Frameskip") == 0)
						{
							settings[selposconfig].value = (settings[selposconfig].value + 1) % 10;
						}
						else if (strcmp(settings[selposconfig].name, "FPS Cap") == 0)
						{
							settings[selposconfig].value = (settings[selposconfig].value + 1) % 6;
						}
						else
						{
							settings[selposconfig].value = 1;
						}
					break;

					case PSP_CTRL_START:
						done = 1;
					break;

					case PSP_CTRL_UP:
						selposconfig = ((unsigned int)(selposconfig - 1)) % totalSettings;
					break;

					case PSP_CTRL_DOWN:
						selposconfig = ((unsigned int)(selposconfig + 1)) % totalSettings;
					break;
				
				default:
					break;
				}
			}
			oldPad = pad;
		}
	}
	pspDebugScreenSetTextColor(0xffffffff);
	FinalizeMainSettings(params);

	// Persist the chosen settings for this ROM.
	SaveRomConfig(rom_filename);
}

//---------------------------------------------------------------------
// File Browser functions (unchanged)
//---------------------------------------------------------------------

f_list filelist;

void ClearFileList()
{
	filelist.cnt = 0;
	filelist.dir_cnt = 0;
}

#include <iostream>
#include <unordered_map>
#include <string_view>
#include <algorithm>
#include <cctype>
#include <array>

// File system includes for PSP (if required)
#include <pspiofilemgr.h>  

enum ExtId {
    EXT_NDS,
    EXT_GZ,
    EXT_ZIP,
    EXT_UNKNOWN
};

constexpr std::array extensions{
    std::pair<std::string_view, ExtId>{"nds", EXT_NDS},
    // {"gz", EXT_GZ},
    // {"zip", EXT_ZIP},
};

bool iequals(std::string_view a, std::string_view b) {
    return std::ranges::equal(a, b, [](char c1, char c2) {
        return std::tolower(c1) == std::tolower(c2);
    });
}

ExtId getExtId(std::string_view filePath) {
    auto pos = filePath.rfind('.');
    if (pos == std::string_view::npos || pos + 1 == filePath.size()) {
        return EXT_UNKNOWN; // No extension or empty extension
    }

    std::string_view ext = filePath.substr(pos + 1);
    for (const auto& [key, value] : extensions) {
        if (iequals(key, ext)) {
            return value;
        }
    }
    return EXT_UNKNOWN;
}


// Max entries the fixed-size filelist arrays (f_name fname[256]) can hold.
#define FILELIST_MAX 256

void GetFileList(const char *root) {
    // Standard C dirent instead of sceIo* — more portable and well-behaved on
    // hosts like PPSSPP. Filters by extension; ignores anything we can't read.
    DIR *d = opendir(root);
    if (!d) return;

    struct dirent *ent;
    while ((ent = readdir(d)) != NULL) {
        // Bound check: writing past fname[256] smashes adjacent globals → crash.
        if (filelist.cnt >= FILELIST_MAX) break;
        if (ent->d_name[0] == '.') continue;            // skip "." / ".."
        if (getExtId(ent->d_name) == EXT_UNKNOWN) continue;
        strncpy(filelist.fname[filelist.cnt].name, ent->d_name,
                sizeof(filelist.fname[filelist.cnt].name) - 1);
        filelist.fname[filelist.cnt].name[sizeof(filelist.fname[filelist.cnt].name) - 1] = '\0';
        filelist.cnt++;
    }
    closedir(d);
}




void GetPrevDir(char *path)
{
	int index = strlen(path) - 1;
	for (; path[--index] != '/' && index > 4;)
		;
	path[index] = 0;
}

int selpos = 0, oldpos = -1, oldpage = 0, ndir = 0;
void DisplayFileList(char *root)
{
	DrawRom(root, &filelist, selpos, true);
	return;
}

// Load the settings[] for the ROM at the given list index, so the split-screen
// settings panel always reflects the highlighted game (defaults + saved file).
static void LoadSettingsForRom(const char *romsDir, int idx)
{
	if (idx < 0 || idx >= filelist.cnt) return;
	InitMainSettings(&my_config);   // defaults into settings[]
	InitExperimentalSettings(&my_config); // defaults into expSettings[]
	char full[300];
	snprintf(full, sizeof(full), "%s/%s", romsDir, filelist.fname[idx].name);
	LoadRomConfig(full);            // overlay saved values (matched by name)
	FinalizeExperimental(&my_config); // apply loaded experimental flags to globals
}

// Adjust the focused setting left/right, mirroring the DoConfig step logic.
static void StepSetting(int dir)  // dir = -1 or +1
{
	const char *name = settings[menu_setSel].name;
	int *v = &settings[menu_setSel].value;
	if (strcmp(name, "Language") == 0)
		*v = (*v + dir + 8) % 8;
	else if (strcmp(name, "Frameskip") == 0)
		*v = (*v + dir + 10) % 10;
	else if (strcmp(name, "FPS Cap") == 0)
		*v = (*v + dir + 6) % 6;
	else
		*v = (dir > 0) ? 1 : 0;   // boolean toggles
}

void DSEmuGui(char *path, char *out)
{
	char tmp[256];
	char app_path[128];
	SceCtrlData pad, oldPad = {0};

	ClearFileList();
	getcwd(app_path, 128);
	sprintf(tmp, "%s/ROMS/", app_path);
	GetFileList(tmp);

	// ROMS dir without the trailing slash, for per-ROM config paths.
	char romsDir[256];
	snprintf(romsDir, sizeof(romsDir), "%s/ROMS", app_path);

	menu_romSel = 0;
	menu_setSel = 0;
	menu_focusRight = false;
	int lastLoadedRom = -1;

	while (1)
	{
		// Keep the settings panel in sync with the highlighted ROM.
		if (menu_romSel != lastLoadedRom) {
			LoadSettingsForRom(romsDir, menu_romSel);
			lastLoadedRom = menu_romSel;
		}

		sceDisplayWaitVblankStart();
		DrawRomMenu(tmp, &filelist, filelist.cnt);

		if (sceCtrlPeekBufferPositive(&pad, 1) && pad.Buttons != oldPad.Buttons)
		{
			oldPad = pad;

			if (pad.Buttons & PSP_CTRL_HOME)
				sceKernelExitGame();

			// L / R toggle focus between the two panels.
			if (pad.Buttons & (PSP_CTRL_LTRIGGER | PSP_CTRL_RTRIGGER))
				menu_focusRight = !menu_focusRight;

			if (pad.Buttons & PSP_CTRL_TRIANGLE)
			{
				pspDebugScreenClear();
				pspDebugScreenSetXY(0, 0);
				jit_difftest_run_force();
				pspDebugScreenPrintf("\n  Press X to return.\n");
				SceCtrlData wpad;
				do { sceCtrlPeekBufferPositive(&wpad, 1); sceKernelDelayThread(50000); }
				while (!(wpad.Buttons & PSP_CTRL_CROSS));
				pspDebugScreenClear();
				oldPad = wpad;
				continue;
			}

			if (pad.Buttons & PSP_CTRL_CROSS)
			{
				// Persist the edited settings for this ROM, then launch it.
				snprintf(out, 256, "%s/%s", tmp, filelist.fname[menu_romSel].name);
				FinalizeMainSettings(&my_config);
				FinalizeExperimental(&my_config);
				char full[300];
				snprintf(full, sizeof(full), "%s/%s", romsDir,
				         filelist.fname[menu_romSel].name);
				SaveRomConfig(full);
				break;
			}

			if (menu_focusRight)
			{
				// Settings panel: Up/Down move, Left/Right change value.
				if (pad.Buttons & PSP_CTRL_UP)
					menu_setSel = (menu_setSel - 1 + totalSettings) % totalSettings;
				else if (pad.Buttons & PSP_CTRL_DOWN)
					menu_setSel = (menu_setSel + 1) % totalSettings;
				else if (pad.Buttons & PSP_CTRL_LEFT)
					StepSetting(-1);
				else if (pad.Buttons & PSP_CTRL_RIGHT)
					StepSetting(+1);
			}
			else
			{
				// ROM list panel: Up/Down by 1, Left/Right by a page of 8.
				if (pad.Buttons & PSP_CTRL_UP)              menu_romSel -= 1;
				else if (pad.Buttons & PSP_CTRL_DOWN)       menu_romSel += 1;
				else if (pad.Buttons & PSP_CTRL_LEFT)       menu_romSel -= 8;
				else if (pad.Buttons & PSP_CTRL_RIGHT)      menu_romSel += 8;
				if (menu_romSel < 0)               menu_romSel = 0;
				if (menu_romSel >= filelist.cnt)   menu_romSel = filelist.cnt - 1;
			}
		}
	}
}

// In-game ROM settings editor. Reached from the pause menu; edits settings[]
// for the running ROM, applies them live and persists them on exit.
void RomSettingsMenu()
{
	extern char rom_filename[256];
	extern bool NDS_3D_ChangeCore(int newCore);
	extern void backup_setManualBackupType(int type);
	extern void arm_jit_reset(bool enable, bool suppress_msg);

	SceCtrlData pad, oldPad = {0};
	if (menu_setSel < 0 || menu_setSel >= totalSettings)
		menu_setSel = 0;
	if (totalExp == 0)
		InitExperimentalSettings(&my_config);

	int tab    = 0;        // 0 = ROM settings, 1 = Experimental
	int expSel = 0;

	// Snapshot the runtime-relevant experimental flags to detect changes that
	// require a JIT recompile on exit. (Index 2 = Cycles In Register is
	// compile-time only, so it is excluded.)
	int expBefore[MAX_EXP_SETTINGS];
	for (int i = 0; i < totalExp; i++) expBefore[i] = expSettings[i].value;

	// This menu is opened by pressing X in the pause menu; that same X press is
	// still held when we get here and would instantly trigger the X (save & exit)
	// case below. Wait for the button to be released first (with a small delay)
	// so the screen is actually accessible.
	sceKernelDelayThread(150000); // 150 ms
	do {
		sceCtrlPeekBufferPositive(&pad, 1);
		sceKernelDelayThread(20000);
	} while (pad.Buttons & PSP_CTRL_CROSS);
	oldPad = pad;

	while (1)
	{
		sceDisplayWaitVblankStart();
		DrawRomSettings(tab ? expSel : menu_setSel, tab);

		if (sceCtrlPeekBufferPositive(&pad, 1) && pad.Buttons != oldPad.Buttons)
		{
			oldPad = pad;

			if (pad.Buttons & (PSP_CTRL_LTRIGGER | PSP_CTRL_RTRIGGER))
			{
				tab ^= 1;
			}
			else if (pad.Buttons & (PSP_CTRL_CROSS | PSP_CTRL_CIRCLE))
				break;
			else if (tab == 0)
			{
				if (pad.Buttons & PSP_CTRL_UP)
					menu_setSel = (menu_setSel - 1 + totalSettings) % totalSettings;
				else if (pad.Buttons & PSP_CTRL_DOWN)
					menu_setSel = (menu_setSel + 1) % totalSettings;
				else if (pad.Buttons & PSP_CTRL_LEFT)
					StepSetting(-1);
				else if (pad.Buttons & PSP_CTRL_RIGHT)
					StepSetting(+1);
			}
			else // Experimental tab: booleans, the 0..4 JIT rows, the 0..2 HLE row
			{
				const bool is_mode =
				    (expSel == g_arm9_mode_idx || expSel == g_arm7_mode_idx);
				const bool is_hle = (expSel == g_arm7_hle_idx);
				const int  span   = is_mode ? 5 : is_hle ? 4 : 0;

				if (pad.Buttons & PSP_CTRL_UP)
					expSel = (expSel - 1 + totalExp) % totalExp;
				else if (pad.Buttons & PSP_CTRL_DOWN)
					expSel = (expSel + 1) % totalExp;
				else if (pad.Buttons & PSP_CTRL_LEFT)
					expSettings[expSel].value = span
					    ? (expSettings[expSel].value + span - 1) % span
					    : 0;
				else if (pad.Buttons & PSP_CTRL_RIGHT)
					expSettings[expSel].value = span
					    ? (expSettings[expSel].value + 1) % span
					    : 1;

				// The HLE mask is read per IPC message, so this is live.
				if (is_hle) HLE_ApplyMode(expSettings[expSel].value);
			}
		}
	}

	// Apply the edited values live and persist them for this ROM. Most fields
	// (frameskip, fps cap, showfps, screen swap) are read every frame from
	// my_config, so they take effect immediately. The 3D core and backup type
	// need an explicit re-apply, matching EMU_Conf().
	FinalizeMainSettings(&my_config);
	FinalizeExperimental(&my_config);
	NDS_3D_ChangeCore(my_config.Render3D);
	backup_setManualBackupType(my_config.savetype);
	SaveRomConfig(rom_filename);

	// Any runtime opt change alters emitted code: recompile blocks. Index 2
	// (Cycles In Register) is compile-time and never differs at runtime.
	bool needReset = false;
	for (int i = 0; i < totalExp; i++)
		if (i != 2 && expSettings[i].value != expBefore[i]) needReset = true;
	if (needReset)
		arm_jit_reset(true, true);
}
