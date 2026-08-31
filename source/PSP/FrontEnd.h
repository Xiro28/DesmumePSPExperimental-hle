#ifndef FRONTEND
#define FRONTEND
//const char *nds_file;

class configured_features 
{
public:

	bool enable_sound;
	bool showfps;
	bool swap;
	bool cur;
	bool Render3D;
	bool FastMERendering;
	bool sort_3d;
	bool _3d_always_on_top;

	bool gpuLayerEnabled[2][5];

	int frameskip;
	int fps_cap_num;
	int DynarecBlockSize;
	bool aot_precompile;    // run AOT pre-compilation before game starts
	int VcountStart;
	int firmware_language;
	int savetype;

	// Experimental JIT optimizations (Experimental settings tab).
	bool exp_idle_loop;     // skip detected idle loops (runtime)
	bool exp_block_link;    // inline direct block-to-block dispatch (runtime)
	bool exp_cycles_reg;    // cycle accumulator in a register (compile-time: JIT_CYCLES_IN_REG)
	bool exp_icache_align;  // align each compiled block to a 64-byte ICache line start
	bool exp_early_term;    // stop block early when suffix is already compiled; jump into it
};

typedef struct configparm {
	char name[32];
	int var;
}configP;

// A single editable setting row, shared with the ROM-menu renderer.
#define MAX_SETTINGS 30
typedef struct
{
	char name[64];
	char description[256];
	int value;
} SETTINGS;

extern SETTINGS settings[MAX_SETTINGS];
extern int totalSettings;       // number of main (per-ROM) settings

// Experimental optimizations tab (separate array from settings[]).
#define MAX_EXP_SETTINGS 12
extern SETTINGS expSettings[MAX_EXP_SETTINGS];
extern int totalExp;
void InitExperimentalSettings(configured_features *params);
void FinalizeExperimental(configured_features *params);

// Text helpers for value-as-label settings (Language / FPS Cap).
const char *GetLanguageText(int lang);
const char *GetFPSCapText(int val);

// Load a ROM's saved settings into settings[] without entering the menu.
// Call after the ROM is chosen so the split-screen menu shows saved values.
void InitMainSettings(configured_features *params);
void FinalizeMainSettings(configured_features *params);

typedef struct fname {
	char name[256];
}f_name;

typedef struct flist {
	f_name fname[256];
	f_name dir[256];
	int cnt;
	int dir_cnt;
}f_list;

extern configured_features my_config;

extern void ChangeRom(bool reset);
extern void ResetRom();
extern void EMU_Conf();
void InitDisplayParams(configured_features* params);
void EXEC_NDS();

void DrawTouchPointer();

void DoConfig(configured_features * params);

// Per-ROM settings persistence (CONFIG/<rombase>.cfg, "name=value" lines).
void SaveRomConfig(const char *romPath);
bool LoadRomConfig(const char *romPath);

void DSEmuGui(char *path,char *out);

// In-game ROM settings editor: edit settings[] live, apply + persist on exit.
void RomSettingsMenu();

void WriteLog(char* msg);

#endif
