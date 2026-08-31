/*
	Copyright (C) 2006 yopyop
	Copyright (C) 2006-2007 shash
	Copyright (C) 2008-2015 DeSmuME team

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

#ifndef _TEXCACHE_H_
#define _TEXCACHE_H_

#include <map>
#include <cstdlib>

#include "types.h"

enum TexCache_TexFormat
{
	TexFormat_None, //used when nothing yet is cached
	TexFormat_32bpp, //used by ogl renderer
	TexFormat_15bpp //used by rasterizer
};

class TexCacheItem;

typedef std::multimap<u32,TexCacheItem*> TTexCacheItemMultimap;

class TexCacheItem
{
public:
	TexCacheItem() 
		: decode_len(0)
		, decoded(NULL)
		, suspectedInvalid(false)
		, assumedInvalid(false)
		, deleteCallback(NULL)
		, cacheFormat(TexFormat_None)
	{}
	~TexCacheItem() {
		delete[] decoded;
		if(swizzled) free(swizzled);
		if(pal_indices) free(pal_indices);
		if(pal_indices_swizzled) free(pal_indices_swizzled);
		if(pal_clut) free(pal_clut);
		if(deleteCallback) deleteCallback(this);
	}
	
	bool suspectedInvalid;
	bool assumedInvalid;

	__attribute__((aligned(64))) u8* decoded; //decoded texture data (linear, 32bpp)

	// PSP GPU swizzled copy of `decoded` (16-byte x 8-row tiled), built lazily on
	// first use and reused. Feeding the GU a swizzled texture greatly reduces its
	// texture-cache thrashing. NULL until built.
	u8* swizzled = nullptr;

	// --- PSP palettized upload (GU_PSM_T8 / T4) ---
	// For simple indexed NDS formats (I8/I4) we keep the raw index data plus a
	// converted CLUT, so the GU can sample 1-byte (T8) or 4-bit (T4) texels with a
	// hardware palette instead of an expanded 32bpp texture (4-8x less bandwidth).
	// pal_indices is the (optionally swizzled) index buffer; pal_clut is the
	// palette in GU_PSM_8888; pal_count is the number of CLUT entries. pal_bpp is
	// 8 or 4. All NULL/0 when the format is not palettizable (A3I5/A5I3/4x4/16bpp).
	u8*  pal_indices = nullptr;
	u8*  pal_indices_swizzled = nullptr;
	u32* pal_clut = nullptr;
	u16  pal_count = 0;
	u8   pal_bpp = 0;            // 0 = not palettized, else 4 or 8

	void (*deleteCallback)(TexCacheItem*);

	u32 decode_len;
	u32 mode;
	u32 texformat, texpal;
	u32 sizeX, sizeY;

	u16 bufferWidth;

	float invSizeX, invSizeY;

	u64 texid; //used by ogl renderer for the texid

	TTexCacheItemMultimap::iterator iterator;

	int getTextureMode() const { return (int)((texformat>>26)&0x07); }

	TexCache_TexFormat cacheFormat;

	struct Dump {
		~Dump() {
			delete[] texture;
		}
		int textureSize, indexSize;
		static const int maxTextureSize=128*1024;
		u8* texture;
		u8 palette[256*2];
	} dump;
};

void TexCache_Invalidate();
void TexCache_Reset();
void TexCache_EvictFrame();

TexCacheItem* TexCache_SetTexture(TexCache_TexFormat TEXFORMAT, u32 format, u32 texpal);

#endif
