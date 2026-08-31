#include <pspkernel.h>
#include <pspdebug.h>
#include <pspdisplay.h>
#include <pspctrl.h>
#include <pspvfpu.h>
#include <stdio.h>
#include <pspgu.h>
#include <pspgum.h>
#include <psprtc.h>
#include <psppower.h>

#include <string.h>
#include <malloc.h>
#include <math.h>   // sinf for menu animation

#include "../common.h"
#include "../utils/decrypt/header.h"
#include "../ctrlssdl.h"
#include "vram.h"
#include "pspvfpu.h"
#include "PSPDisplay.h"
#include "pspDmac.h"
#include "../GPU.h"
#include "intraFont.h"
#include "pspdisplay.h"
#include "../rasterize.h"

#include <png.h>

#define SLICE_SIZE 16
#define MAX_COL 32
#define TOTAL_ICONS (MAX_COL * 2)
#define ICON_H 150
#define ICON_W 120
#define RGB(r, v, b) ((r) | ((v) << 8) | ((b) << 16) | (0xff << 24))

#define GU_VRAM_WIDTH 512
#define VRAM_START 0x4000000

// FIX: display list must be 64-byte aligned and sized correctly.
// A display list of 256KB is more than enough for any frame.
unsigned int __attribute__((aligned(64))) gulist[256 * 1024 / 4];

// VRAM layout (PSP VRAM = 2MB total):
// Each buffer: 512 * 272 * 2 bytes (GU_PSM_5551) = 0x44000 bytes
//   frameBuffer  @ 0x00000  (display buffer)
//   doubleBuffer @ 0x44000  (draw buffer)
//   depthBuffer  @ 0x88000  (depth buffer, same size)
// Total used: 0xCC000 = 835584 bytes < 2MB  ✓
//
// Vertex buffers must NOT be placed inside framebuffer/depth VRAM.
// FIX: use uncached system RAM for vertex data (memalign).
#define VRAM_BASE_UNCACHED  0x44000000u
#define VRAM_FB0_OFFSET     0x000000u   // frameBuffer
#define VRAM_FB1_OFFSET     0x044000u   // doubleBuffer
#define VRAM_DEPTH_OFFSET   0x088000u   // depthBuffer
#define VRAM_NDS2D_OFFSET   0x0CC000u   // NDS 2D framebuffer staging area

void* frameBuffer  = (void*)VRAM_FB0_OFFSET;
void* doubleBuffer = (void*)VRAM_FB1_OFFSET;
void* depthBuffer  = (void*)VRAM_DEPTH_OFFSET;


u8* DISP_POINTER = (u8*)(0x44000000u + 0x100000u);
u8* DISP_POINTER_FRONT = (u8*)(0x44000000u + 0x100000u + 512 * 192 * 2);

// ─── FRONT layer double-buffer ───────────────────────────────────────────────
//
// Problem this solves:
//   The NDS 2D pipeline writes the FRONT (above-3D) overlay sparsely — it only
//   touches pixels that actually belong to a FRONT-priority sprite/BG. The
//   rest of the buffer keeps whatever value was there before. With a single
//   persistent buffer, sprites that move leave "ghost" copies at their old
//   positions (the alpha bit stays set, so the alpha test in Draw2DTexture
//   keeps drawing them on top of the 3D every frame).
//
// Why not just memset GPU_Screen_extra each frame:
//   `renderScreenFull()` in NDSSystem.cpp alternates between MAIN and SUB
//   every frame as a frame-skip optimization. Only MAIN writes the FRONT
//   layer (see GPU.cpp ~line 2095: `gpu->core == GPU_MAIN && dispCnt->BG0_3D`).
//   So clearing every frame wipes MAIN's FRONT on SUB frames → HUD flickers.
//   AND the existing DMA in NDSSystem.cpp runs at the START of each frame
//   and copies GPU_Screen_extra → VRAM 0x04130000; if we cleared at end of
//   one frame, next frame's DMA copies the clear and the display goes blank.
//
// Solution (true double-buffer, see EMU_SCREEN_Finish for the swap logic):
//   - Allocate TWO FRONT buffers in system RAM.
//   - GPU_Screen_extra always points at the "write" buffer; the GPU's sparse
//     writes accumulate there for the current MAIN render.
//   - EMU_SCREEN reads the OTHER buffer ("display") via sceGuTexImage,
//     bypassing the existing DMA-to-VRAM path entirely (the DMA still runs
//     but its destination 0x04130000 is no longer consulted).
//   - After a MAIN render finishes, swap the two indices and CLEAR the new
//     write buffer. That way the next MAIN render starts from zero (no ghost)
//     while the display keeps showing the freshly captured frame.
//
// Double is sufficient (not triple) because the EMU_SCREEN_Finish sceGuSync
// already guarantees the GU has consumed the texture before we touch it —
// no reader/writer race, so we don't need a spare "in-flight" buffer.
#define FRONT_BUF_SIZE (512 * 192 * 2)   // 192 KB, stride 1024 bytes/line
static u8* front_bufs[2] = { NULL, NULL };
static int front_display_idx = 0;
static int front_write_idx   = 1;

intraFont* Font;
intraFont* RomFont;

struct DispVertex {
    unsigned short u, v;
    signed short x, y, z;
};

struct DispVertexColored {
    unsigned short u, v;
    unsigned short color;
    signed short x, y, z;
};

struct DispVertexColoredNT {
    unsigned short color;
    signed short x, y, z;
};

#define TEXTURE_FLAGS          (GU_TEXTURE_16BIT | GU_VERTEX_16BIT | GU_TRANSFORM_2D)
#define TEXTURE_FLAGS_COLOR_NT (GU_COLOR_5551    | GU_VERTEX_16BIT | GU_TRANSFORM_2D)

// ─── vertex buffers in system RAM ────────────────────────────────────────────
#define MAX_SLICES 64   // ceil(512/SLICE_SIZE) = 32, ×2 for safety

static struct DispVertex          sprt_gpuvtx_buf[MAX_SLICES * 2] __attribute__((aligned(16)));
static struct DispVertexColoredNT bck_gpuvtx_buf[4]               __attribute__((aligned(16)));
static struct DispVertex          icons_gpuvtx_buf[2]              __attribute__((aligned(16)));
static struct DispVertex          mouse_vtx[2]                     __attribute__((aligned(16)));

struct DispVertex*           sprt_gpuvtx  = sprt_gpuvtx_buf;
struct DispVertexColoredNT*  bck_gpuvtx   = bck_gpuvtx_buf;
struct DispVertex*           icons_gpuvtx = icons_gpuvtx_buf;


// ─── Icon / menu ─────────────────────────────────────────────────────────────

#define ICON_SZ 32

class Icon {
public:
    u16*  GetIconData()  { return data; }
    char* GetIconName()  { return RomName; }
    char* GetDevName()   { return Developer; }
    char* GetFileName()  { return Filename; }
    char* GetTitle()     { return Title; }
    u16   GetAccent()    { return accent; }

    void SetIconPixel(u8 X, u8 Y, u16 pixel) { data[X + Y * ICON_W] = pixel; }

    void SetIconName(const char* Name) {
        strncpy(RomName, (*Name == '.') ? "Homebrew" : Name, 11);
        RomName[11] = 0;
    }
    void SetTitle(const char* Name)    { strncpy(Title, Name, 63); Title[63] = 0; }
    void SetDevName(const char* Name)  { strncpy(Developer, Name, 63); Developer[63] = 0; }
    void SetFileName(const char* Name) { strncpy(Filename,  Name, 127); Filename[127] = 0; }

    void ClearIcon(u16 color)   { memset(data, color, ICON_W * ICON_H * 2); }
    void MEMSetIcon(u16* buff)  { memcpy(data, buff, ICON_W * ICON_H * 2); }

    // Average the opaque icon pixels (32x32, 5551) into a single accent color,
    // boosted toward a vivid tint so it reads as a UI accent. Computed once at
    // load time and reused by the renderer for per-game theming.
    void ComputeAccent() {
        u32 r = 0, g = 0, b = 0, n = 0;
        for (int yy = 0; yy < ICON_SZ; yy++)
            for (int xx = 0; xx < ICON_SZ; xx++) {
                u16 p = data[xx + yy * 32];      // icon stride is 32
                if (!(p & 0x8000)) continue;     // skip transparent
                r += (p) & 0x1F; g += (p >> 5) & 0x1F; b += (p >> 10) & 0x1F;
                n++;
            }
        if (!n) { accent = 0xFFFF; return; }     // default white-ish
        r /= n; g /= n; b /= n;
        // Boost saturation a bit: push the brightest channel up.
        u32 mx = r > g ? (r > b ? r : b) : (g > b ? g : b);
        if (mx < 24 && mx > 0) { r = r * 24 / mx; g = g * 24 / mx; b = b * 24 / mx; }
        if (r > 31) r = 31; if (g > 31) g = 31; if (b > 31) b = 31;
        accent = (u16)(0x8000 | (b << 10) | (g << 5) | r);
    }

private:
    char RomName[12];
    char Title[64];
    char Developer[64];
    char Filename[128];
    u16  accent = 0xFFFF;
    __attribute__((aligned(16))) u16 data[ICON_H * ICON_W];
};

Icon menu[TOTAL_ICONS];

void DrawIcon(u16 x, u16 y, u8 sprX, bool curON)
{
    sceGuColor(0xffffffff);

    struct DispVertex* vertices = (struct DispVertex*)sceGuGetMemory(2 * sizeof(struct DispVertex));

    sceGuSendCommandi(0x12, 1 << 23);   // TMAP enable: honor per-row texture switch
    sceGuTexMode(GU_PSM_5551, 0, 0, 0);
    sceGuTexImage(0, ICON_SZ, ICON_SZ, ICON_SZ, menu[sprX].GetIconData());
    sceGuTexFlush();
    sceGuTexSync();
    sceGuTexFunc(GU_TFX_REPLACE, GU_TCC_RGB);
    sceGuTexFilter(GU_LINEAR, GU_LINEAR);
    sceGuTexWrap(GU_CLAMP, GU_CLAMP);

    vertices[0].u = 0;      vertices[0].v = 0;
    vertices[0].x = x;      vertices[0].y = y - 5 * (1 + curON);  vertices[0].z = 0;

    vertices[1].u = ICON_SZ; vertices[1].v = ICON_SZ;
    vertices[1].x = x + 23;  vertices[1].y = y + 23 - 5 * (1 + curON);  vertices[1].z = 0;

    sceGuDrawArray(GU_SPRITES, TEXTURE_FLAGS, 2, NULL, vertices);
}

int curr_posX = -1;
int curr_page = 0;
int old_page  = -1;
int N_Roms    = 0;

void Set_POSX(int pos) { curr_posX = pos; }
void Set_PAGE(int pos) { curr_page = pos; }

// ─── Modern split-screen ROM menu ─────────────────────────────────────────────
//
// Left panel: scrollable ROM list (name + NDS icon, selected row highlighted).
// Right panel: per-ROM settings for the highlighted ROM, editable in place.
// L/R switch focus between the two panels. The look is built from flat colored
// rectangles (FillRect) plus intraFont text — no extra assets.

// Defined further down (loads NDS banner icons for the current page).
bool CreateRomIcon(char* file, f_list* list);

// Convert 0xAARRGGBB (intuitive) to PSP 5551 (BGR + 1 alpha bit), the same
// vertex color format the rest of this file already uses (DispVertexColoredNT).
static inline u16 to5551(u8 r, u8 g, u8 b, u8 a)
{
    return (u16)(((r >> 3) & 0x1F)
              | (((g >> 3) & 0x1F) << 5)
              | (((b >> 3) & 0x1F) << 10)
              | ((a ? 1 : 0) << 15));
}

// Filled, untextured rectangle in screen space using the proven 5551 colored
// vertex format (GU_COLOR_8888 + 16-bit verts is not a valid GE combo and
// faults the hardware → black-screen crash).
static void FillRect(int x, int y, int w, int h, u16 color5551)
{
    struct DispVertexColoredNT* v =
        (struct DispVertexColoredNT*)sceGuGetMemory(2 * sizeof(struct DispVertexColoredNT));

    sceGuDisable(GU_TEXTURE_2D);
    v[0].color = color5551; v[0].x = x;     v[0].y = y;     v[0].z = 0;
    v[1].color = color5551; v[1].x = x + w; v[1].y = y + h; v[1].z = 0;
    sceGuDrawArray(GU_SPRITES, TEXTURE_FLAGS_COLOR_NT, 2, NULL, v);
    sceGuEnable(GU_TEXTURE_2D);
}

// Menu palette (5551).
#define COL_BG        to5551(0x14, 0x1C, 0x2A, 0xFF)  // background
#define COL_ACCENT    to5551(0x1B, 0x5F, 0xE6, 0xFF)  // selection / bars (blue)
#define COL_PANEL     to5551(0x1E, 0x2A, 0x3A, 0xFF)  // active panel
#define COL_PANEL_DIM to5551(0x10, 0x16, 0x20, 0xFF)  // inactive panel
#define COL_ROW_DIM   to5551(0x40, 0x40, 0x40, 0xFF)  // dim row highlight
#define COL_TRACK     to5551(0x30, 0x30, 0x30, 0xFF)  // scrollbar track
#define COL_BAR       to5551(0x08, 0x08, 0x08, 0xFF)  // bottom bar

// Solid dark background.
static void DrawBackgroundGradient()
{
    FillRect(0, 0, 480, 272, COL_BG);
}

// Draw the NDS icon for an on-page slot at (x,y), scaled to `size` px.
static void DrawIconAt(int slot, int x, int y, int size)
{
    // Make this fully self-contained: FillRect and intraFont leave the GU
    // texture state in whatever they last set (FillRect disables texturing,
    // intraFont sets its own tex func / disables it on exit). On real hardware
    // that stale state makes the icon vanish, so re-establish everything here.
    sceGuEnable(GU_TEXTURE_2D);
    sceGuSendCommandi(0x12, 1 << 23);   // TMAP enable: honor per-row texture switch
    sceGuColor(0xffffffff);
    sceGuShadeModel(GU_FLAT);
    // intraFont enables GU_ALPHA_TEST (to skip transparent glyph pixels) and
    // leaves it on. We rely on the texture's 5551 alpha bit + blend, not the
    // alpha test, so that stale test rejects the icon pixels and the icon
    // vanishes after any intraFont print this frame. Reset alpha test + blend.
    sceGuDisable(GU_ALPHA_TEST);
    sceGuEnable(GU_BLEND);
    sceGuBlendFunc(GU_ADD, GU_SRC_ALPHA, GU_ONE_MINUS_SRC_ALPHA, 0, 0);

    struct DispVertex* vtx = (struct DispVertex*)sceGuGetMemory(2 * sizeof(struct DispVertex));
    sceGuTexMode(GU_PSM_5551, 0, 0, 0);
    sceKernelDcacheWritebackRange(menu[slot].GetIconData(), ICON_SZ * ICON_SZ * 2);
    sceGuTexImage(0, ICON_SZ, ICON_SZ, ICON_SZ, menu[slot].GetIconData());
    sceGuTexFlush();   // flush GU texture cache after changing texture pointer
    sceGuTexSync();    // also sync: without it the GE keeps sampling the first
                       // icon of the frame for every row (3D path does both)
    sceGuTexFunc(GU_TFX_REPLACE, GU_TCC_RGBA);  // use alpha bit so transparent pixels show bg
    sceGuTexFilter(GU_LINEAR, GU_LINEAR);
    sceGuTexWrap(GU_CLAMP, GU_CLAMP);
    // intraFont draws with normalized UVs and leaves a non-1 tex scale/offset.
    // We use integer (16-bit) UVs, so reset these or the icon UVs collapse and
    // the texture vanishes after any intraFont print this frame.
    sceGuTexScale(1.0f, 1.0f);
    sceGuTexOffset(0.0f, 0.0f);
    vtx[0].u = 0;       vtx[0].v = 0;       vtx[0].x = x;        vtx[0].y = y;        vtx[0].z = 0;
    vtx[1].u = ICON_SZ; vtx[1].v = ICON_SZ; vtx[1].x = x + size; vtx[1].y = y + size; vtx[1].z = 0;
    sceGuDrawArray(GU_SPRITES, TEXTURE_FLAGS, 2, NULL, vtx);
}

// State shared with the menu input loop (DSEmuGui).
int  menu_romSel     = 0;   // absolute index into the ROM list
int  menu_setSel     = 0;   // index into settings[]
bool menu_focusRight = false;

// Convert a 5551 color to intraFont's 0xAABBGGRR (8888), full alpha.
static inline u32 col5551to8888(u16 c)
{
    u32 r = ((c)        & 0x1F) << 3;
    u32 g = ((c >> 5)   & 0x1F) << 3;
    u32 b = ((c >> 10)  & 0x1F) << 3;
    return 0xFF000000u | (b << 16) | (g << 8) | r;   // ABGR for intraFont
}

// Scale a 5551 color toward black by num/den (for dim variants of the accent).
static inline u16 scale5551(u16 c, int num, int den)
{
    u32 r = (((c)       & 0x1F) * num / den) & 0x1F;
    u32 g = (((c >> 5)  & 0x1F) * num / den) & 0x1F;
    u32 b = (((c >> 10) & 0x1F) * num / den) & 0x1F;
    return (u16)(0x8000 | (b << 10) | (g << 5) | r);
}

// Render one full frame of the split-screen menu.
void DrawRomMenu(char* file, f_list* list, int total)
{
    // ── Original State Logic ──
    int page = (TOTAL_ICONS > 0) ? menu_romSel / TOTAL_ICONS : 0;
    if (old_page != page) {
        char rompath[256];
        strncpy(rompath, file, sizeof(rompath) - 1);
        rompath[sizeof(rompath) - 1] = 0;
        curr_page = page;
        CreateRomIcon(rompath, list);
        old_page = page;
    }
    curr_page = page;

    const int selSlot = menu_romSel - page * TOTAL_ICONS;
    const bool haveSel = (total > 0 && selSlot >= 0 && selSlot < N_Roms);

    // ── Original Color Logic ──
    const u16 accent5551 = haveSel ? menu[selSlot].GetAccent() : COL_ACCENT;
    const u32 accent8888 = col5551to8888(accent5551);
    const u16 accentDim  = scale5551(accent5551, 1, 3);
    const u16 accentDark = scale5551(accent5551, 1, 5); // Deep accent for card backgrounds

    // ── Original Animation Logic ──
    static float selAnim = 0.0f;     
    static float pulse    = 0.0f;    
    selAnim += ((float)menu_romSel - selAnim) * 0.06f;   
    pulse += 0.015f; if (pulse > 6.2831853f) pulse -= 6.2831853f;  
    const float pulseScale = 1.0f + 0.035f * sinf(pulse);  

    // ── Original Hardware Setup ──
    sceGuSync(0, 0);
    sceGuStart(GU_DIRECT, gulist);

    // ── Deferred text ────────────────────────────────────────────────────────
    // On real hardware, any intraFont print poisons the GU state so that the
    // NEXT textured sprite (our icons via DrawIconAt) vanishes, and no per-draw
    // state reset recovers it. So we draw ALL icons/rects first and queue every
    // bit of text here, flushing it only at the very end. Icons and text never
    // overlap (text sits beside icons), so the z-order is unaffected.
    struct QText { float x, y, size; u32 color; char str[48]; };
    static QText qtext[64];
    int qN = 0;
    auto QT = [&](float x, float y, float size, u32 color, const char* s) {
        if (qN >= 64) return;
        qtext[qN].x = x; qtext[qN].y = y; qtext[qN].size = size; qtext[qN].color = color;
        strncpy(qtext[qN].str, s, sizeof(qtext[qN].str) - 1);
        qtext[qN].str[sizeof(qtext[qN].str) - 1] = 0;
        qN++;
    };

    sceGuDisable(GU_DEPTH_TEST);
    sceGuDepthMask(GU_TRUE);
    sceGuDisable(GU_STENCIL_TEST);
    sceGuDisable(GU_CULL_FACE);
    sceGuDisable(GU_SCISSOR_TEST);
    sceGuDisable(GU_ALPHA_TEST);
    sceGuEnable(GU_BLEND);
    sceGuBlendFunc(GU_ADD, GU_SRC_ALPHA, GU_ONE_MINUS_SRC_ALPHA, 0, 0);
    sceGuEnable(GU_TEXTURE_2D);

    sceGuClearColor(0xFF000000);
    sceGuClear(GU_COLOR_BUFFER_BIT);

    // ── Soft palette ─────────────────────────────────────────────────────────
    const u16 COL_PANEL_A = to5551(0x18, 0x1C, 0x26, 0xFF); // focused panel bg
    const u16 COL_PANEL_B = to5551(0x12, 0x15, 0x1D, 0xFF); // unfocused panel bg
    const u16 COL_BACK    = to5551(0x0C, 0x0E, 0x14, 0xFF); // window background
    const u16 COL_TRACK_C = to5551(0x2A, 0x2E, 0x38, 0xFF); // scrollbar track
    const u16 COL_DIVIDER = to5551(0x26, 0x2B, 0x36, 0xFF);
    const u32 C_TITLE = 0xFFF2F2F2;   // bright
    const u32 C_NAME  = 0xFFDCDCDC;
    const u32 C_SUB   = 0xFF8C8C8C;   // dim
    const u32 C_FAINT = 0xFF6A6A6A;
    const u32 C_ONACC = 0xFF101010;   // text on an accent highlight

    FillRect(0, 0, 480, 272, COL_BACK); // solid soft background

    // ════ Top header bar ════
    FillRect(0, 0, 480, 24, accentDark);
    FillRect(0, 24, 480, 2, accent5551);
    QT(10, 16, 0.50f, accent8888, "DeSmuME PSP");
    {
        char hbuf[48];
        snprintf(hbuf, sizeof(hbuf), "%d / %d", (total > 0 ? menu_romSel + 1 : 0), total);
        QT(210, 16, 0.45f, C_NAME, hbuf);
        snprintf(hbuf, sizeof(hbuf), "BAT %d%%", scePowerGetBatteryLifePercent());
        QT(412, 16, 0.45f, C_NAME, hbuf);
    }

    const int CT_Y = 30, CT_H = 212;            // content band: 30..242
    const int LEFT_X = 6,   LEFT_W = 192;
    const int RIGHT_X = 204, RIGHT_W = 270;

    // ════ LEFT: ROM list (icon + name + subtitle) ════
    FillRect(LEFT_X, CT_Y, LEFT_W, CT_H, menu_focusRight ? COL_PANEL_B : COL_PANEL_A);

    const int ROW_H    = 30;
    const int LIST_TOP = CT_Y + 4;
    const int VISIBLE  = (CT_H - 8) / ROW_H;
    const int LROW_ICON = 22;                 // per-row icon size
    const int LTEXT_X   = LEFT_X + 5 + LROW_ICON + 6;  // text column after the icon

    int firstRow = menu_romSel - VISIBLE / 2;
    if (firstRow < 0) firstRow = 0;
    if (firstRow > total - VISIBLE) firstRow = (total > VISIBLE) ? total - VISIBLE : 0;

    for (int r = 0; r < VISIBLE; r++) {
        int idx = firstRow + r;
        if (idx >= total) break;
        int ry = LIST_TOP + r * ROW_H;
        bool sel = (idx == menu_romSel);
        int slot = idx - page * TOTAL_ICONS;

        if (sel) {
            // Full-width selection bar + accent edge on the left.
            FillRect(LEFT_X + 2, ry, LEFT_W - 6, ROW_H - 2,
                     menu_focusRight ? accentDim : accent5551);
            FillRect(LEFT_X, ry, 3, ROW_H - 2, accent5551);
        }
        // Per-row icon for every loaded ROM except the selected one (that one
        // is shown big in the detail card). Drawn before the deferred text.
        else if (slot >= 0 && slot < N_Roms) {
            DrawIconAt(slot, LEFT_X + 5, ry + (ROW_H - 2 - LROW_ICON) / 2, LROW_ICON);
        }

        char namebuf[48];
        snprintf(namebuf, sizeof(namebuf), "%.20s", list->fname[idx].name);
        QT(LTEXT_X, ry + 13, 0.46f, sel ? C_ONACC : C_NAME, namebuf);

        // Subtitle: internal NDS title for loaded rows (dim).
        if (slot >= 0 && slot < N_Roms && menu[slot].GetTitle()[0]) {
            char subbuf[48];
            snprintf(subbuf, sizeof(subbuf), "%.24s", menu[slot].GetTitle());
            QT(LTEXT_X, ry + 25, 0.36f, sel ? C_ONACC : C_SUB, subbuf);
        }
    }

    // Scrollbar
    if (total > VISIBLE) {
        int trackH = VISIBLE * ROW_H;
        int knobH  = trackH * VISIBLE / total; if (knobH < 12) knobH = 12;
        int knobY  = LIST_TOP + (trackH - knobH) * firstRow / (total - VISIBLE);
        FillRect(LEFT_X + LEFT_W - 4, LIST_TOP, 3, trackH, COL_TRACK_C);
        FillRect(LEFT_X + LEFT_W - 4, knobY, 3, knobH, accent5551);
    }

    // ════ RIGHT: detail card (small left-centered icon) ════
    const int DET_Y = CT_Y, DET_H = 104;

    FillRect(RIGHT_X, DET_Y, RIGHT_W, DET_H, COL_PANEL_A);
    FillRect(RIGHT_X, DET_Y, RIGHT_W, 22, accentDark);
    FillRect(RIGHT_X, DET_Y + DET_H - 1, RIGHT_W, 1, COL_DIVIDER);
    QT(RIGHT_X + 10, DET_Y + 15, 0.42f, accent8888, "GAME DETAILS");

    if (haveSel) {
        // Small icon, centered inside a left column of the card.
        const int COL_W = 60;
        const int base  = 48;
        int sz = (int)(base * pulseScale);
        int ix = RIGHT_X + 8 + (COL_W - sz) / 2;
        int iy = DET_Y + 28 + (base - sz) / 2 + 6;

        FillRect(ix - 2, iy - 2, sz + 4, sz + 4, accent5551);
        DrawIconAt(selSlot, ix, iy, sz);

        int textX = RIGHT_X + 8 + COL_W + 12;
        char tbuf[48];
        snprintf(tbuf, sizeof(tbuf), "%.16s", menu[selSlot].GetTitle());
        QT(textX, DET_Y + 44, 0.55f, C_TITLE, tbuf);

        if (menu[selSlot].GetDevName()[0]) {
            snprintf(tbuf, sizeof(tbuf), "%.18s", menu[selSlot].GetDevName());
            QT(textX, DET_Y + 64, 0.42f, accent8888, tbuf);
        }
        snprintf(tbuf, sizeof(tbuf), "%.22s", menu[selSlot].GetFileName());
        QT(textX, DET_Y + 82, 0.38f, C_FAINT, tbuf);
    }

    // ════ RIGHT: settings panel (clean list, value badges) ════
    const int SET_Y = DET_Y + DET_H + 6;
    const int SET_H = (CT_Y + CT_H) - SET_Y;

    FillRect(RIGHT_X, SET_Y, RIGHT_W, SET_H, menu_focusRight ? COL_PANEL_A : COL_PANEL_B);
    FillRect(RIGHT_X, SET_Y, RIGHT_W, 20, accentDark);
    QT(RIGHT_X + 10, SET_Y + 14, 0.42f, menu_focusRight ? accent8888 : C_SUB, "OPTIONS");

    const int SROW_H = 18;
    const int SVIS   = (SET_H - 24) / SROW_H;
    int sFirst = menu_setSel - SVIS / 2;
    if (sFirst < 0) sFirst = 0;
    if (sFirst > totalSettings - SVIS) sFirst = (totalSettings > SVIS) ? totalSettings - SVIS : 0;

    for (int r = 0; r < SVIS; r++) {
        int idx = sFirst + r;
        if (idx >= totalSettings) break;
        int ry = SET_Y + 22 + r * SROW_H;
        bool sel = (idx == menu_setSel) && menu_focusRight;

        if (sel)
            FillRect(RIGHT_X + 2, ry, RIGHT_W - 4, SROW_H - 1, accent5551);

        char valbuf[24];
        if (strcmp(settings[idx].name, "Language") == 0)
            snprintf(valbuf, sizeof(valbuf), "%s", GetLanguageText(settings[idx].value));
        else if (strcmp(settings[idx].name, "FPS Cap") == 0)
            snprintf(valbuf, sizeof(valbuf), "%s", GetFPSCapText(settings[idx].value));
        else if (settings[idx].value == 0 || settings[idx].value == 1)
            snprintf(valbuf, sizeof(valbuf), "%s", settings[idx].value ? "ON" : "OFF");
        else
            snprintf(valbuf, sizeof(valbuf), "%d", settings[idx].value);

        char nbuf[48];
        snprintf(nbuf, sizeof(nbuf), "%.18s", settings[idx].name);
        QT(RIGHT_X + 10, ry + SROW_H - 5, 0.42f, sel ? C_ONACC : C_NAME, nbuf);

        char vshow[32];
        snprintf(vshow, sizeof(vshow), "< %s >", valbuf);
        QT(RIGHT_X + RIGHT_W - 78, ry + SROW_H - 5, 0.42f, sel ? C_ONACC : accent8888, vshow);
    }

    // ════ Footer hint bar ════
    FillRect(0, 246, 480, 26, accentDark);
    FillRect(0, 246, 480, 2, accent5551);
    QT(10, 263, 0.42f, C_NAME,
       menu_focusRight ? "L/R Panel   U/D Setting   </> Change   X Launch"
                       : "L/R Panel   U/D Game   X Launch   Tri JIT   HOME Exit");

    // ── Flush all queued text now that every icon is drawn ───────────────────
    for (int i = 0; i < qN; i++) {
        intraFontSetStyle(Font, qtext[i].size, qtext[i].color, 0, 0, 0);
        intraFontPrint(Font, qtext[i].x, qtext[i].y, qtext[i].str);
    }

    sceGuFinish();
    sceGuSync(GU_SYNC_FINISH, GU_SYNC_WHAT_DONE);
    sceGuSwapBuffers();
}

void drawmenu()
{
    sceGuSync(0, 0);
    sceGuStart(GU_DIRECT, gulist);

    sceGuDisable(GU_DEPTH_TEST);
    sceGuDepthMask(GU_TRUE);
    sceGuDisable(GU_STENCIL_TEST);
    sceGuDisable(GU_CULL_FACE);
    sceGuDisable(GU_SCISSOR_TEST);
    sceGuDisable(GU_ALPHA_TEST);
    sceGuEnable(GU_BLEND);
    sceGuBlendFunc(GU_ADD, GU_SRC_ALPHA, GU_ONE_MINUS_SRC_ALPHA, 0, 0);
    sceGuEnable(GU_TEXTURE_2D);

    sceGuClearColor(0x10404047);
    sceGuClear(GU_COLOR_BUFFER_BIT);

    for (int y = 50, romX = 0; y <= 220; y += 32)
        for (int x = 30; x < 410; x += 32, romX++) {
            if (N_Roms <= romX) goto done;
            DrawIcon(x, y, romX, (curr_posX == romX));
            if (curr_posX == romX)
                intraFontPrintf(Font, 20, 240, "ROM: %s::%s",
                    menu[romX].GetIconName(), menu[romX].GetFileName());
        }
done:
    intraFontPrint(Font, 210, 15, "Release V3.1");
    intraFontPrintf(Font, 390, 15, "Battery:%d%%", scePowerGetBatteryLifePercent());

    sceGuFinish();
    sceGuSync(GU_SYNC_FINISH, GU_SYNC_WHAT_DONE);
    sceGuSwapBuffers();
}

// ─── Screen vertex buffers ────────────────────────────────────────────────────

static const float sw    = 511.0f;
static const int   scale = (int)(sw * (float)SLICE_SIZE) / (int)512;

// FIX: allocate from system RAM, not VRAM. Sized for the maximum slice count.
// Maximum slices = ceil(512 / SLICE_SIZE) = 32, each needs 2 vertices.
#define MAX_SLICES 32
int n_slices = 0;

// PSPDisplay.cpp — SetupDisplay: build separate slice arrays per screen half

#define MAX_SLICES_HALF 32

// Separate slice arrays: left half and right half of PSP screen
static struct DispVertex slices_left[MAX_SLICES_HALF  * 2] __attribute__((aligned(16)));
static struct DispVertex slices_right[MAX_SLICES_HALF * 2] __attribute__((aligned(16)));
int n_slices_left  = 0;
int n_slices_right = 0;

void SetupDisplay()
{
    n_slices_left  = 0;
    n_slices_right = 0;

    // NDS framebuffer: 512 wide, left 256px = left NDS screen, right 256px = right NDS screen
    // PSP display: left 240px = left half, right 240px = right half
    // Scale: 256 texels → 240 pixels

    const int texHalfW = 256;   // each NDS screen is 256px wide in texture
    const int dispHalfW = 240;  // each screen occupies 240 PSP pixels
    const int texH  = 192;
    const int startY = 40;
    const int endY   = 232;

    // Left half: texture U=[0,256], screen X=[0,240]
    for (int start = 0, idx = 0; start < texHalfW; start += SLICE_SIZE, idx += 2) {
        int width = std::min(SLICE_SIZE, texHalfW - start);
        int sx0 = (start         * dispHalfW) / texHalfW;
        int sx1 = ((start+width) * dispHalfW) / texHalfW;

        slices_left[idx  ].u = start;         slices_left[idx  ].v = 0;
        slices_left[idx  ].x = sx0;           slices_left[idx  ].y = startY; slices_left[idx  ].z = 0;
        slices_left[idx+1].u = start + width; slices_left[idx+1].v = texH;
        slices_left[idx+1].x = sx1;           slices_left[idx+1].y = endY;   slices_left[idx+1].z = 0;
        n_slices_left++;
    }

    // Right half: texture U=[256,512], screen X=[240,480]
    for (int start = 0, idx = 0; start < texHalfW; start += SLICE_SIZE, idx += 2) {
        int width = std::min(SLICE_SIZE, texHalfW - start);
        int sx0 = 240 + (start         * dispHalfW) / texHalfW;
        int sx1 = 240 + ((start+width) * dispHalfW) / texHalfW;

        slices_right[idx  ].u = 256 + start;         slices_right[idx  ].v = 0;
        slices_right[idx  ].x = sx0;                 slices_right[idx  ].y = startY; slices_right[idx  ].z = 0;
        slices_right[idx+1].u = 256 + start + width; slices_right[idx+1].v = texH;
        slices_right[idx+1].x = sx1;                 slices_right[idx+1].y = endY;   slices_right[idx+1].z = 0;
        n_slices_right++;
    }

    sceKernelDcacheWritebackRange(slices_left,  n_slices_left  * 2 * sizeof(DispVertex));
    sceKernelDcacheWritebackRange(slices_right, n_slices_right * 2 * sizeof(DispVertex));

    // Background quads unchanged
    bck_gpuvtx[0].x = 0;   bck_gpuvtx[0].y = startY; bck_gpuvtx[0].z = 0;
    bck_gpuvtx[1].x = 240; bck_gpuvtx[1].y = endY;   bck_gpuvtx[1].z = 0;
    bck_gpuvtx[2].x = 240; bck_gpuvtx[2].y = startY; bck_gpuvtx[2].z = 0;
    bck_gpuvtx[3].x = 480; bck_gpuvtx[3].y = endY;   bck_gpuvtx[3].z = 0;
    bck_gpuvtx[0].color = bck_gpuvtx[1].color = 0xFFFF;
    bck_gpuvtx[2].color = bck_gpuvtx[3].color = 0xFFFF;
    sceKernelDcacheWritebackRange(bck_gpuvtx, 4 * sizeof(DispVertexColoredNT));
}

void draw_mouse(int x, int y)
{
    mouse_vtx[0].u = 0; mouse_vtx[0].v = 0;
    mouse_vtx[0].x = 240 + x; mouse_vtx[0].y = 40 + y; mouse_vtx[0].z = 0;

    mouse_vtx[1].u = 8; mouse_vtx[1].v = 8;
    mouse_vtx[1].x = 240 + x + 8; mouse_vtx[1].y = 40 + y + 8; mouse_vtx[1].z = 0;
}

// ─── Touch cursor (crosshair) ─────────────────────────────────────────────────
//
// A 16x16 RGBA5551 crosshair generated once: a thin ring + a 4-way tick cross +
// a centre dot, each with a dark outline so it stays visible over any game
// colour. Drawn centred on the touch point and scaled up a bit so it reads well
// on the small bottom screen. Replaces the old 8x8 square outline.
#define CURSOR_TEX 16
#define CURSOR_DRAW 14   // on-screen size in pixels
static u16 cursorTex[CURSOR_TEX * CURSOR_TEX] __attribute__((aligned(16)));
static bool cursorBuilt = false;

static void BuildCursorTex()
{
    const u16 CLEAR = 0x0000;
    const u16 WHITE = 0xFFFF;
    const int mid = CURSOR_TEX / 2;

    for (int y = 0; y < CURSOR_TEX; y++)
        for (int x = 0; x < CURSOR_TEX; x++) {
            bool onV = (x == mid - 1 || x == mid);
            bool onH = (y == mid - 1 || y == mid);
            cursorTex[x + y * CURSOR_TEX] = (onV || onH) ? WHITE : CLEAR;
        }

    sceKernelDcacheWritebackRange(cursorTex, sizeof(cursorTex));
    cursorBuilt = true;
}

void DrawTouchCursor(int x, int y)
{
    if (!cursorBuilt) BuildCursorTex();

    sceGuEnable(GU_TEXTURE_2D);
    sceGuColor(0xffffffff);
    sceGuEnable(GU_BLEND);
    sceGuBlendFunc(GU_ADD, GU_SRC_ALPHA, GU_ONE_MINUS_SRC_ALPHA, 0, 0);
    sceGuEnable(GU_ALPHA_TEST);
    sceGuAlphaFunc(GU_GREATER, 0, 0xFF);   // discard fully-transparent texels
    sceGuTexMode(GU_PSM_5551, 0, 0, 0);
    sceGuTexFunc(GU_TFX_REPLACE, GU_TCC_RGBA);
    sceGuTexFilter(GU_LINEAR, GU_LINEAR);
    sceGuTexWrap(GU_CLAMP, GU_CLAMP);
    sceGuTexImage(0, CURSOR_TEX, CURSOR_TEX, CURSOR_TEX, cursorTex);

    const int half = CURSOR_DRAW / 2;
    struct DispVertex* v = (struct DispVertex*)sceGuGetMemory(2 * sizeof(struct DispVertex));
    v[0].u = 0;          v[0].v = 0;          v[0].x = 240 + x - half;           v[0].y = 40 + y - half;            v[0].z = 0;
    v[1].u = CURSOR_TEX; v[1].v = CURSOR_TEX; v[1].x = 240 + x - half + CURSOR_DRAW; v[1].y = 40 + y - half + CURSOR_DRAW; v[1].z = 0;
    sceGuDrawArray(GU_SPRITES, TEXTURE_FLAGS, 2, NULL, v);

    sceGuDisable(GU_ALPHA_TEST);
}

void DrawSprite()
{
    icons_gpuvtx[0].u = 0;   icons_gpuvtx[0].v = 0;
    icons_gpuvtx[0].x = 0;   icons_gpuvtx[0].y = 45;        icons_gpuvtx[0].z = 0;
    icons_gpuvtx[1].u = 512; icons_gpuvtx[1].v = 192;
    icons_gpuvtx[1].x = 470; icons_gpuvtx[1].y = 192 + 45;  icons_gpuvtx[1].z = 0;

    sceGuDrawArray(GU_SPRITES, TEXTURE_FLAGS, 2, NULL, icons_gpuvtx);
}

// ─── EMU_SCREEN ───────────────────────────────────────────────────────────────

u32 old_color_main = 0;
u32 old_color_sub  = 0;


static inline u16 getNDSBackdropColor(int core)
{
    // BG palette RAM layout in ARM9_VMEM (0x05000000):
    //   Main engine BG palette: offset 0x000 (512 bytes)
    //   Sub  engine BG palette: offset 0x200 (512 bytes)  ← FIX: was 0x400
    // Color index 0 = backdrop color.
    u32 offset = (core == 0) ? 0x000 : 0x200;
    u16 ndsColor = T1ReadWord(MMU.ARM9_VMEM, offset) & 0x7FFF;
    // Force alpha=1 (opaque) in GU_PSM_5551
    return ndsColor | 0x8000;
}


static bool emuFrameStarted = false;

// Funzione reale per disegnare il buffer 2D
struct DispVertexTextured {
    unsigned short u, v; 
    short x, y, z;       
};

static void Draw2DTexture(volatile u8* texture_buffer, int screen_x, int screen_y, int screen_w, int screen_h, int u_start = -1) {
    sceGuEnable(GU_TEXTURE_2D);
    sceGuTexMode(GU_PSM_5551, 0, 0, 0);
    // Explicit filter/wrap: the 3D pass leaves arbitrary state here (varies per
    // polygon). Without this the framebuffer texture can be sampled with
    // GU_LINEAR (blurry) or REPEAT (wraparound tile garbage at edges).
    sceGuTexFilter(GU_NEAREST, GU_NEAREST);
    sceGuTexWrap(GU_CLAMP, GU_CLAMP);
    sceGuTexImage(0, 512, 256, 512, (void*)texture_buffer);
    // Texture cache must be flushed AFTER binding a new image — otherwise the
    // GU samples stale texels left over from the last 3D polygon's texture.
    sceGuTexFlush();
    sceGuTexFunc(GU_TFX_REPLACE, GU_TCC_RGBA);
    sceGuTexScale(1.0f, 1.0f);
    sceGuTexOffset(0.0f, 0.0f);

    sceGuEnable(GU_ALPHA_TEST);          // transparent NDS pixels (alpha bit 0)
    sceGuAlphaFunc(GU_GREATER, 0, 0xFF); // pass only if alpha != 0

    sceGuDrawArray(GU_SPRITES, TEXTURE_FLAGS, n_slices_left  * 2, NULL, slices_left);
    sceGuDrawArray(GU_SPRITES, TEXTURE_FLAGS, n_slices_right * 2, NULL, slices_right);
}

// No-argument form used by the baseline's NDS_exec()/main(): draw both screens.
void EMU_SCREEN()
{
	EMU_SCREEN(true, true);
}

void EMU_SCREEN(bool skip2d, bool skip3d)
{
    const bool do_3d = (((REG_DISPx*)&MMU.ARM9_REG[0])->dispx_DISPCNT.bits.BG0_3D)
                       && !skip3d;
    if (skip2d && !do_3d) return;

    const u16 _color_main = getNDSBackdropColor(MainScreen.gpu->core);
    const u16 _color_sub  = getNDSBackdropColor(SubScreen.gpu->core);
    const bool _3dOnLeft  = (MainScreen.offset == 0);

    struct DispVertex* slices_3d = _3dOnLeft ? slices_left  : slices_right;
    struct DispVertex* slices_2d = _3dOnLeft ? slices_right : slices_left;
    int n_3d = _3dOnLeft ? n_slices_left  : n_slices_right;
    int n_2d = _3dOnLeft ? n_slices_right : n_slices_left;

    sceGuSync(0, 0);
    sceGuStart(GU_DIRECT, gulist);

    // FIX problema 2: pulisci l'intero framebuffer a nero prima di ogni frame.
    // Questo elimina il grigio del menu che persiste fuori dall'area NDS (y<40, y>232).
    sceGuDisable(GU_SCISSOR_TEST);
    sceGuDisable(GU_DEPTH_TEST);
    sceGuDepthMask(GU_TRUE);
    sceGuClearColor(0xFF000000);
    sceGuClearDepth(0);
    sceGuClear(GU_COLOR_BUFFER_BIT | GU_DEPTH_BUFFER_BIT);

    sceGuDisable(GU_ALPHA_TEST);
    sceGuDisable(GU_STENCIL_TEST);
    sceGuDisable(GU_CULL_FACE);
    sceGuEnable(GU_BLEND);
    sceGuBlendFunc(GU_ADD, GU_SRC_ALPHA, GU_ONE_MINUS_SRC_ALPHA, 0, 0);

   if (do_3d)
    {
        if (old_color_main != _color_main || old_color_sub != _color_sub) {
            if (_3dOnLeft) {
                bck_gpuvtx[0].color = bck_gpuvtx[1].color = _color_main;
                bck_gpuvtx[2].color = bck_gpuvtx[3].color = _color_sub;
            } else {
                bck_gpuvtx[0].color = bck_gpuvtx[1].color = _color_sub;
                bck_gpuvtx[2].color = bck_gpuvtx[3].color = _color_main;
            }
            old_color_main = _color_main;
            old_color_sub  = _color_sub;
            sceKernelDcacheWritebackRange(bck_gpuvtx, 4 * sizeof(DispVertexColoredNT));
        }

        int v_idx = 0;
        int screen_x = bck_gpuvtx[v_idx].x;
        int screen_y = bck_gpuvtx[v_idx].y;
        int screen_w = bck_gpuvtx[v_idx + 1].x - screen_x;
        int screen_h = bck_gpuvtx[v_idx + 1].y - screen_y;

        int u_start = _3dOnLeft ? 0 : 256;

        // 0. Sfondo tinta unita
        sceGuDisable(GU_TEXTURE_2D);
        sceGuDrawArray(GU_SPRITES, TEXTURE_FLAGS_COLOR_NT, 4, NULL, bck_gpuvtx);


        // 1. BACK LAYER (Sfondo 2D dietro al 3D)
        sceGuDisable(GU_DEPTH_TEST); // Forza la scrittura sotto il 3D
        sceGuEnable(GU_TEXTURE_2D); 
        sceGuDisable(GU_BLEND);     
        Draw2DTexture(DISP_POINTER, screen_x, screen_y, screen_w, screen_h); 

        // 2. MIDDLE LAYER (Hardware 3D)
        sceGuEnable(GU_DEPTH_TEST);  // Riattiva la profondità per i modelli
        sceGuDepthMask(GU_TRUE); 
        gpu3D->NDS_3D_Render();

        // 3. FRONT LAYER (HUD, Sprite 2D davanti al 3D)
        sceGuDisable(GU_DEPTH_TEST); // Non fa tagliare l'HUD dal 3D
        sceGuDisable(GU_BLEND);
        // Read from our managed display buffer (system RAM) instead of the
        // DMA destination at DISP_POINTER_FRONT. The DMA still copies to that
        // VRAM address but its result is no longer consulted — the double-
        // buffer logic in EMU_SCREEN_Finish maintains front_bufs[display_idx]
        // with the most recent complete MAIN render.
        volatile u8* front_src = front_bufs[front_display_idx]
                                 ? (volatile u8*)front_bufs[front_display_idx]
                                 : (volatile u8*)DISP_POINTER_FRONT; // fallback before init
        Draw2DTexture(front_src, screen_x, screen_y, screen_w, screen_h, u_start);
    }
    else
    {
        sceGuDisable(GU_DEPTH_TEST);
        sceGuDepthMask(GU_TRUE);
        Draw2DTexture(DISP_POINTER, 0, 40, 480, 192);
    }

    if (my_config.cur)
        DrawTouchCursor(mouse.x % 256, mouse.y % 192);

    // FIX problema 3: NON fare sceGuSwapBuffers() qui.
    // Il menu HUD (configurazione) viene disegnato DOPO EMU_SCREEN
    // nella stessa chiamata sceGuStart/Finish, quindi deve restare
    // sullo stesso buffer. Lo swap avviene solo alla fine del frame completo.
    sceGuTexFlush();
    sceGuFinish();

     
    emuFrameStarted = true;
}

void EMU_SCREEN_Finish()
{
    if (!emuFrameStarted) return;

    // sceGuSync waits for the display list to finish — once it returns the GU
    // has finished sampling the FRONT texture, so the display buffer is free
    // to be reused.
    sceGuSync(0, 0);
    sceGuSwapBuffers();
    emuFrameStarted = false;

    // ─── FRONT double-buffer swap ───────────────────────────────────────────
    //
    // renderScreenFull() alternates MAIN/SUB each frame. We mirror that
    // alternation with a static toggle. The initial state matches
    // renderScreenFull's `static bool upScreen = true;` so the first time we
    // run, MAIN has just been rendered.
    //
    // On MAIN-render frames:
    //   - The write buffer now holds this frame's complete FRONT data
    //     (sparse, but starting from zero because we cleared it on the
    //     previous swap), so swap it to the display side.
    //   - Clear the new write buffer for the next MAIN render — this is what
    //     stops ghost trails from accumulating.
    //   - Repoint GPU_Screen_extra so the GPU's next FRONT writes land in
    //     the freshly-cleared buffer.
    //
    // On SUB-render frames:
    //   - No FRONT writes happened, so do nothing. The display keeps showing
    //     the most recent MAIN render (no flicker between MAIN frames).
    if (front_bufs[0] && front_bufs[1]) {
        static bool justRenderedMain = false;
        justRenderedMain = !justRenderedMain;   // starts true on first call

        if (justRenderedMain) {
            const int new_display = front_write_idx;
            const int new_write   = front_display_idx;

            // Flush GPU's CPU-side writes to RAM before the GU samples them
            // next frame.
            sceKernelDcacheWritebackRange(front_bufs[new_display], FRONT_BUF_SIZE);

            // Wipe the new write target so next MAIN starts ghost-free.
            memset(front_bufs[new_write], 0, FRONT_BUF_SIZE);
            sceKernelDcacheWritebackRange(front_bufs[new_write], FRONT_BUF_SIZE);

            front_display_idx = new_display;
            front_write_idx   = new_write;
            GPU_Screen_extra  = (volatile u8*)front_bufs[new_write];
        }
    }
}

// ─── Init ─────────────────────────────────────────────────────────────────────

void Init_PSP_DISPLAY_FRAMEBUFF()
{
    static bool inited = false;

    sceGuInit();
    sceGuStart(GU_DIRECT, gulist);

    sceGuDrawBuffer(GU_PSM_5551, doubleBuffer, GU_VRAM_WIDTH);
    sceGuDispBuffer(SCR_WIDTH, SCR_HEIGHT, frameBuffer, GU_VRAM_WIDTH);
    sceGuDepthBuffer(depthBuffer, GU_VRAM_WIDTH);

    sceGuDepthRange(65535, 0);
    sceGuDisable(GU_DEPTH_TEST);
    sceGuDepthMask(GU_TRUE);
    sceGuDisable(GU_SCISSOR_TEST);
    sceGuDisable(GU_STENCIL_TEST);
    sceGuDisable(GU_CULL_FACE);
    sceGuDisable(GU_ALPHA_TEST);

    sceGuTexWrap(GU_CLAMP, GU_CLAMP);
    sceGuTexMode(GU_PSM_5551, 0, 0, 0);
    sceGuEnable(GU_TEXTURE_2D);
    sceGuDisable(GU_BLEND);
    sceGuBlendFunc(GU_ADD, GU_SRC_ALPHA, GU_ONE_MINUS_SRC_ALPHA, 0, 0);
    sceGuTexFunc(GU_TFX_REPLACE, GU_TCC_RGB);

    ScePspFMatrix4 identity = {
        {1.f,0,0,0},{0,1.f,0,0},{0,0,1.f,0},{0,0,0,1.f}
    };
    sceGuSetMatrix(GU_PROJECTION, &identity);
    sceGuSetMatrix(GU_TEXTURE,    &identity);
    sceGuSetMatrix(GU_MODEL,      &identity);
    sceGuSetMatrix(GU_VIEW,       &identity);

    sceGuViewport(0, 0, 480, 272);

    // Clear both buffers
    sceGuClearColor(0xFF000000);
    sceGuClearDepth(0);
    sceGuClear(GU_COLOR_BUFFER_BIT | GU_DEPTH_BUFFER_BIT);
    sceGuFinish();
    sceGuSync(GU_SYNC_FINISH, GU_SYNC_WHAT_DONE);

    sceGuSwapBuffers();
    sceGuStart(GU_DIRECT, gulist);
    sceGuClearColor(0xFF000000);
    sceGuClear(GU_COLOR_BUFFER_BIT | GU_DEPTH_BUFFER_BIT);
    sceGuFinish();
    sceGuSync(GU_SYNC_FINISH, GU_SYNC_WHAT_DONE);
    sceGuSwapBuffers();

    sceGuDisplay(GU_TRUE);

    if (inited) return;
    inited = true;

    intraFontInit();
    Font = intraFontLoad("flash0:/font/ltn1.pgf", INTRAFONT_CACHE_ASCII);
    intraFontActivate(Font);
    intraFontSetStyle(Font, 0.6f, 0xFFFFFFFF, 0, 0, 0);

    // Allocate our two FRONT source buffers and redirect the NDS GPU to
    // write into the "write" half. GPU.cpp's own memalign for GPU_Screen_extra
    // is leaked here (192 KB, one-time), which is preferable to fighting it
    // with a free() that could race with concurrent ME-side accesses on PSP.
    front_bufs[0] = (u8*)memalign(64, FRONT_BUF_SIZE);
    front_bufs[1] = (u8*)memalign(64, FRONT_BUF_SIZE);
    if (front_bufs[0] && front_bufs[1]) {
        memset(front_bufs[0], 0, FRONT_BUF_SIZE);
        memset(front_bufs[1], 0, FRONT_BUF_SIZE);
        sceKernelDcacheWritebackRange(front_bufs[0], FRONT_BUF_SIZE);
        sceKernelDcacheWritebackRange(front_bufs[1], FRONT_BUF_SIZE);
        GPU_Screen_extra = (volatile u8*)front_bufs[front_write_idx];
    }

    SetupDisplay();
}


// ─── ROM icon loading ─────────────────────────────────────────────────────────

Header header;

int readBanner(char* filename, tNDSBanner* banner)
{
    FILE* romF = fopen(filename, "rb");
    if (!romF) return 1;
    fread(&header, sizeof(header), 1, romF);
    fseek(romF, header.banner_offset, SEEK_SET);
    fread(banner, sizeof(*banner), 1, romF);
    fclose(romF);
    return 0;
}

void loadImage(unsigned short* image, unsigned short* palette, unsigned char* tileData)
{
    for (int tile = 0; tile < 16; tile++) {
        for (int pixel = 0; pixel < 32; pixel++) {
            unsigned char a = tileData[(tile << 5) + pixel];
            int px = ((tile & 3) << 3) + ((pixel << 1) & 7);
            int py = ((tile >> 2) << 3) + (pixel >> 2);

            unsigned short upper = (a & 0xF0) >> 4;
            unsigned short lower = (a & 0x0F);

            image[(px+1) + py * 32] = upper ? (palette[upper] | 0x8000) : 0;
            image[ px    + py * 32] = lower ? (palette[lower] | 0x8000) : 0;
        }
    }
}

bool CreateRomIcon(char* file, f_list* list)
{
    N_Roms = 0;
    tNDSBanner banner;

    for (int c = 0; c < TOTAL_ICONS; c++) {
        if (list->cnt <= c) break;

        int index = c + (curr_page * TOTAL_ICONS);
        if (list->cnt <= index) break;

        char rompath[256];
        strncpy(rompath, file, sizeof(rompath) - 1);
        rompath[sizeof(rompath)-1] = 0;
        strncat(rompath, list->fname[index].name, sizeof(rompath) - strlen(rompath) - 1);

        if (readBanner(rompath, &banner)) {
            // No readable banner (homebrew / new rom without a valid header):
            // don't abort the whole page — every slot after this one would be
            // left unloaded (no icon, and haveSel false hides the detail card).
            // Fill a fallback slot from the filename so the rom still shows.
            menu[c].ClearIcon(0x0000);
            menu[c].SetIconName(list->fname[index].name);
            menu[c].SetTitle(list->fname[index].name);
            menu[c].SetFileName(list->fname[index].name);
            menu[c].SetDevName("");
            menu[c].ComputeAccent();
            N_Roms++;
            continue;
        }

        loadImage(menu[c].GetIconData(), banner.palette, banner.icon);
        menu[c].SetIconName(header.title);
        menu[c].SetTitle(header.title);
        menu[c].SetFileName(list->fname[index].name);

        // Extract the developer line: NDS banner titles are multi-line
        // (title / subtitle / developer), newline-separated. title_en is UTF-16;
        // grab the text after the last newline as the developer string.
        {
            const u16* t = banner.titles ? (const u16*)banner.titles[1] : NULL; // English
            char dev[64]; int dn = 0; int lastNL = -1;
            char buf[128]; int bn = 0;
            if (t) {
                for (int i = 0; i < 128 && t[i] != 0; i++) {
                    char ch = (t[i] < 0x80) ? (char)t[i] : '?';
                    if (ch == '\n') lastNL = bn;
                    buf[bn++] = ch;
                }
                buf[bn] = 0;
                if (lastNL >= 0 && lastNL + 1 < bn) {
                    for (int i = lastNL + 1; i < bn && dn < 63; i++)
                        if (buf[i] != '\n') dev[dn++] = buf[i];
                }
            }
            dev[dn] = 0;
            menu[c].SetDevName(dn ? dev : "");
        }

        menu[c].ComputeAccent();
        N_Roms++;
    }
    return true;
}

void DrawRom(char* file, f_list* list, int pos, bool reload)
{
    char rompath[256];
    strncpy(rompath, file, sizeof(rompath) - 1);
    rompath[sizeof(rompath)-1] = 0;

    curr_page = pos / TOTAL_ICONS;
    curr_posX = pos % TOTAL_ICONS;

    if (old_page != curr_page) {
        CreateRomIcon(rompath, list);
        old_page = curr_page;
    }
    drawmenu();
}

// ─── GU-based in-game pause menu ─────────────────────────────────────────────

extern intraFont* Font;

static const char* s_pause_items[] = {
    "Resume", "Change Rom", "Reset Rom", "Save State", "Load State", "Rom Settings", "Exit"
};
static const int s_pause_item_count = 7;

void DrawPauseMenu(int selected)
{
    sceGuSync(0, 0);
    sceGuStart(GU_DIRECT, gulist);

    sceGuDisable(GU_DEPTH_TEST);
    sceGuDepthMask(GU_TRUE);
    sceGuDisable(GU_STENCIL_TEST);
    sceGuDisable(GU_CULL_FACE);
    sceGuDisable(GU_SCISSOR_TEST);
    sceGuDisable(GU_ALPHA_TEST);
    sceGuEnable(GU_BLEND);
    sceGuBlendFunc(GU_ADD, GU_SRC_ALPHA, GU_ONE_MINUS_SRC_ALPHA, 0, 0);
    sceGuEnable(GU_TEXTURE_2D);

    sceGuClearColor(0xFF000000);
    sceGuClear(GU_COLOR_BUFFER_BIT);

    DrawBackgroundGradient();

    // Centered panel
    const int PW = 220, PH = 20 + s_pause_item_count * 24 + 28;
    const int PX = (480 - PW) / 2;
    const int PY = (272 - PH) / 2;

    FillRect(PX, PY, PW, PH, COL_PANEL);
    FillRect(PX, PY, PW, 22, COL_ACCENT);
    FillRect(PX, PY + PH - 22, PW, 22, COL_PANEL_DIM);

    intraFontSetStyle(Font, 0.55f, 0xFF101010, 0, 0, 0);
    intraFontPrint(Font, PX + 8, PY + 15, "PAUSE");
    intraFontSetStyle(Font, 0.45f, 0xFF808080, 0, 0, 0);
    intraFontPrintf(Font, PX + PW - 56, PY + 15, "BAT %d%%", scePowerGetBatteryLifePercent());

    const int ROW_H = 24;
    const int LIST_Y = PY + 26;
    for (int i = 0; i < s_pause_item_count; i++) {
        int ry = LIST_Y + i * ROW_H;
        if (i == selected)
            FillRect(PX + 4, ry + 1, PW - 8, ROW_H - 2, COL_ACCENT);
        intraFontSetStyle(Font, 0.52f,
            i == selected ? 0xFF101010 : 0xFFCFCFCF, 0, 0, 0);
        intraFontPrint(Font, PX + 14, ry + ROW_H - 7, s_pause_items[i]);
    }

    intraFontSetStyle(Font, 0.42f, 0xFF606060, 0, 0, 0);
    intraFontPrint(Font, PX + 8, PY + PH - 7, "X confirm   O back");

    sceGuFinish();
    sceGuSync(GU_SYNC_FINISH, GU_SYNC_WHAT_DONE);
    sceGuSwapBuffers();
}

// ─── GU-based in-game ROM settings editor ────────────────────────────────────
// settings[] / totalSettings / GetLanguageText / GetFPSCapText come from FrontEnd.h.

void DrawRomSettings(int sel, int tab)
{
    SETTINGS  *arr   = tab ? expSettings : settings;
    const int  count = tab ? totalExp    : totalSettings;
    const char *title = tab ? "EXPERIMENTAL" : "ROM SETTINGS";

    sceGuSync(0, 0);
    sceGuStart(GU_DIRECT, gulist);

    sceGuDisable(GU_DEPTH_TEST);
    sceGuDepthMask(GU_TRUE);
    sceGuDisable(GU_STENCIL_TEST);
    sceGuDisable(GU_CULL_FACE);
    sceGuDisable(GU_SCISSOR_TEST);
    sceGuDisable(GU_ALPHA_TEST);
    sceGuEnable(GU_BLEND);
    sceGuBlendFunc(GU_ADD, GU_SRC_ALPHA, GU_ONE_MINUS_SRC_ALPHA, 0, 0);
    sceGuEnable(GU_TEXTURE_2D);

    sceGuClearColor(0xFF000000);
    sceGuClear(GU_COLOR_BUFFER_BIT);

    DrawBackgroundGradient();

    const int ROW_H = 22;
    const int VIS   = 8;                       // visible rows
    const int PW = 380;
    const int PH = 34 + VIS * ROW_H + 24;
    const int PX = (480 - PW) / 2;
    const int PY = (272 - PH) / 2;

    // Panel + header/footer banners
    FillRect(PX + 5, PY + 5, PW, PH, 0xAA000000);   // shadow
    FillRect(PX, PY, PW, PH, COL_PANEL);
    FillRect(PX, PY, PW, 26, COL_ACCENT);
    FillRect(PX, PY + PH - 22, PW, 22, COL_PANEL_DIM);

    intraFontSetStyle(Font, 0.55f, 0xFF101010, 0, 0, 0);
    intraFontPrint(Font, PX + 10, PY + 18, title);
    intraFontSetStyle(Font, 0.42f, 0xFF101010, 0, 0, 0);
    intraFontPrintf(Font, PX + PW - 60, PY + 18, "%d/%d", sel + 1, count);

    // Scroll window centered on the selection
    int first = sel - VIS / 2;
    if (first < 0) first = 0;
    if (first > count - VIS) first = (count > VIS) ? count - VIS : 0;

    const int LIST_Y = PY + 30;
    for (int r = 0; r < VIS; r++) {
        int idx = first + r;
        if (idx >= count) break;
        int ry = LIST_Y + r * ROW_H;
        bool selrow = (idx == sel);

        if (selrow)
            FillRect(PX + 4, ry + 1, PW - 8, ROW_H - 2, COL_ACCENT);

        char valbuf[24];
        if (strcmp(arr[idx].name, "Language") == 0)
            snprintf(valbuf, sizeof(valbuf), "%s", GetLanguageText(arr[idx].value));
        else if (strcmp(arr[idx].name, "FPS Cap") == 0)
            snprintf(valbuf, sizeof(valbuf), "%s", GetFPSCapText(arr[idx].value));
        else if (arr[idx].value == 0 || arr[idx].value == 1)
            snprintf(valbuf, sizeof(valbuf), "%s", arr[idx].value ? "ON" : "OFF");
        else
            snprintf(valbuf, sizeof(valbuf), "%d", arr[idx].value);

        intraFontSetStyle(Font, 0.48f, selrow ? 0xFF101010 : 0xFFCFCFCF, 0, 0, 0);
        intraFontPrintf(Font, PX + 14, ry + ROW_H - 6, "%.28s", arr[idx].name);

        intraFontSetStyle(Font, 0.48f, selrow ? 0xFF101010 : 0xFF7FD0FF, 0, 0, 0);
        intraFontPrintf(Font, PX + PW - 90, ry + ROW_H - 6, "< %s >", valbuf);
    }

    intraFontSetStyle(Font, 0.42f, 0xFF808080, 0, 0, 0);
    intraFontPrint(Font, PX + 10, PY + PH - 7, "U/D select  </> change  L/R tab  X/O save");

    sceGuFinish();
    sceGuSync(GU_SYNC_FINISH, GU_SYNC_WHAT_DONE);
    sceGuSwapBuffers();
}

// ─── GU-based AOT progress screen ────────────────────────────────────────────

void DrawAOTProgress(const char* phase, int pct, int blocks)
{
    sceGuSync(0, 0);
    sceGuStart(GU_DIRECT, gulist);

    sceGuDisable(GU_DEPTH_TEST);
    sceGuDepthMask(GU_TRUE);
    sceGuDisable(GU_STENCIL_TEST);
    sceGuDisable(GU_CULL_FACE);
    sceGuDisable(GU_SCISSOR_TEST);
    sceGuDisable(GU_ALPHA_TEST);
    sceGuEnable(GU_BLEND);
    sceGuBlendFunc(GU_ADD, GU_SRC_ALPHA, GU_ONE_MINUS_SRC_ALPHA, 0, 0);
    sceGuEnable(GU_TEXTURE_2D);

    sceGuClearColor(0xFF000000);
    sceGuClear(GU_COLOR_BUFFER_BIT);

    DrawBackgroundGradient();

    const int PW = 300, PH = 90;
    const int PX = (480 - PW) / 2;
    const int PY = (272 - PH) / 2;

    FillRect(PX, PY, PW, PH, COL_PANEL);
    FillRect(PX, PY, PW, 22, COL_ACCENT);

    intraFontSetStyle(Font, 0.55f, 0xFF101010, 0, 0, 0);
    intraFontPrint(Font, PX + 8, PY + 15, "AOT Pre-compile");

    intraFontSetStyle(Font, 0.48f, 0xFFCFCFCF, 0, 0, 0);
    intraFontPrint(Font, PX + 8, PY + 38, phase);

    // Progress bar
    const int BAR_X = PX + 8, BAR_Y = PY + 48, BAR_W = PW - 16, BAR_H = 10;
    FillRect(BAR_X, BAR_Y, BAR_W, BAR_H, COL_TRACK);
    FillRect(BAR_X, BAR_Y, BAR_W * pct / 100, BAR_H, COL_ACCENT);

    intraFontSetStyle(Font, 0.42f, 0xFF808080, 0, 0, 0);
    intraFontPrintf(Font, PX + 8, PY + 74, "%d%%  %d blocks compiled", pct, blocks);

    sceGuFinish();
    sceGuSync(GU_SYNC_FINISH, GU_SYNC_WHAT_DONE);
    sceGuSwapBuffers();
}