#include <iostream>
#include <vector>
#include <algorithm>
#include <unordered_map>

// Host registers usable to cache NDS registers.
// s4 is excluded: it is a scratch reg used by generate_condition_check().
// k0(RCPU), fp(PC), gp(flags) are also reserved elsewhere.
// s5/s6/s7 were previously unused; adding them grows the cache 4->7 slots,
// which sharply cuts spill/reload traffic for multi-operand ARM ops (LDM/STM,
// 3-operand ALU) that used to thrash the whole cache every instruction.
// DIAGNOSTIC (temporary): s5/s6/s7 removed from the pool.
//
// The LDM/STM codegen uses s5 (running address), s6 (rotate amount) and s7
// (writeback value) as scratch that must survive the calls to _MMU_read32 /
// JIT_WRITE32 -- it picked callee-saved regs for exactly that reason. But the
// comment above says the pool was later grown 4->7 by adding those same three
// registers, so the cache may now hand one of them to a guest register while
// LDM/STM is mid-sequence. Difftest reports LDM/STM 62/112 with the base ending
// at Rn-8n instead of Rn-4n, and The Sims 3 dies on the STMDB/LDMIA pair at
// 02004F20 / 02004F74. Decode, addresses and writeback all verified correct, so
// a scratch conflict is what is left.
//
// RESULT: shrinking the pool to {s0,s1,s2} did NOT fix it -- The Sims 3 still
// boots to black and LDM/STM still fails. So the cache is NOT stealing the
// LDM/STM scratch registers; hypothesis excluded, pool restored.
#define SRA_TEST_NO_S567 0
#if SRA_TEST_NO_S567
static const psp_gpr_t SRA_REGS[] = { psp_s0, psp_s1, psp_s2 };
#elif JIT_CYCLES_IN_REG
// s3 is reserved as RCYC (cycle accumulator held across the chain): 6 cache slots.
static const psp_gpr_t SRA_REGS[] = { psp_s0, psp_s1, psp_s2, psp_s5, psp_s6, psp_s7 };
#else
static const psp_gpr_t SRA_REGS[] = { psp_s0, psp_s1, psp_s2, psp_s3, psp_s5, psp_s6, psp_s7 };
#endif
static const int SRA_REG_COUNT = (int)(sizeof(SRA_REGS) / sizeof(SRA_REGS[0]));

class register_manager
{
   public:
      void reset(bool thumb = false)
      {
         memset(mapping, 0xFF, sizeof(mapping));
         memset(usage_tag, 0, sizeof(usage_tag));
         memset(dirty, 0, sizeof(dirty));
         memset(weak, 0, sizeof(weak));
         //memset(sp_reg_pushed, false, sizeof(sp_reg_pushed));
         next_usage_tag = 1;
         this->thumb = thumb;
      }

      bool is_usable(psp_gpr_t reg) const
      {
         static const uint32_t USE_MAP = 0x1F0000;
         return (USE_MAP & (1 << reg)) ? true : false;
      }

   private:
      
      bool thumb = false;

      psp_gpr_t find(uint32_t emu_reg_id)
      {
         for (int k = 0; k < SRA_REG_COUNT; k ++)
         {
            int i = SRA_REGS[k];
            if (mapping[i] == emu_reg_id)
            {
               usage_tag[i] = next_usage_tag ++;
               return (psp_gpr_t) i;
            }
         }

         return psp_null;
      }

      int32_t get_loaded(uint32_t emu_reg_id, bool no_read)
      {
         psp_gpr_t current = find(emu_reg_id);

         if (current >= 0)
         {
            if (weak[current] && !no_read)
            {
               read_emu(current, emu_reg_id);
               weak[current] = false;
            }
         }

         return current;
      }

      psp_gpr_t get_oldest()
      {
         psp_gpr_t result = psp_s0;
         uint32_t lowtag = 0xFFFFFFFF;

         for (int k = 0; k < SRA_REG_COUNT; k ++)
         {
            int i = SRA_REGS[k];
            if (usage_tag[i] < lowtag)
            {
               lowtag = usage_tag[i];
               result = (psp_gpr_t) i;
            }
         }

         return result;
      }

   public:

      bool is_mapped(uint32_t emu_reg_id)
      {
         for (int k = 0; k < SRA_REG_COUNT; k ++)
         {
            if (mapping[SRA_REGS[k]] == emu_reg_id)
            {
               return true;
            }
         }
         return false;
      }

      void map(uint32_t emu_reg_id, psp_gpr_t native_reg)
      {
         mapping[native_reg] = emu_reg_id;
         usage_tag[native_reg] = next_usage_tag ++;
         dirty[native_reg] = false;
         weak[native_reg] = false;
      }

      void get(uint32_t reg_count, int32_t* emu_reg_ids)
      {
         assert(reg_count < 5);
         bool found[5] = { false, false, false, false, false };

         // Find existing registers
         for (uint32_t i = 0; i < reg_count; i ++)
         {
            if (emu_reg_ids[i] < 0)
            {
               found[i] = true;
            }
            else
            {
               if (emu_reg_ids[i] == 15) {
                  // r15 is not kept in the LRU cache: psp_fp advances every
                  // instruction, so any cached value would be stale by the next op.
                  // Always recompute PC+8 (ARM) / PC+4 (Thumb) fresh from psp_fp.
                  emu_reg_ids[i] = psp_v1;
                  emit_addiu(psp_v1, psp_fp, thumb ? 2 : 4);
                  found[i] = true;
               } else {
                  int32_t current = get_loaded(emu_reg_ids[i] & 0xF, emu_reg_ids[i] & 0x10);
                  if (current >= 0)
                  {
                     emu_reg_ids[i] = current;
                     found[i] = true;
                  }
               }
            }
         }

         // Load new registers
         for (uint32_t i = 0; i < reg_count; i ++)
         {
            if (!found[i])
            {
               // Search register list again, in case the same register is used twice
               int32_t current = get_loaded(emu_reg_ids[i] & 0xF, emu_reg_ids[i] & 0x10);
               if (current >= 0)
               {
                  emu_reg_ids[i] = current;
                  found[i] = true;
               }
               else
               {
                  // Read the new register
                  psp_gpr_t result = get_oldest();
                  flush(result);

                  //printf("Loading register %d into %d at 0x%x\n", emu_reg_ids[i] & 0xF, result, emit_getCurrAdr());

                  if (!(emu_reg_ids[i] & 0x10))
                  {
                     read_emu(result, emu_reg_ids[i] & 0xF);
                  }

                  mapping[result] = emu_reg_ids[i] & 0xF;
                  usage_tag[result] = next_usage_tag ++;
                  weak[result] = (emu_reg_ids[i] & 0x10) ? true : false;

                  emu_reg_ids[i] = result;
                  found[i] = true;
               }
            }
         }
      }

      void mark_dirty(uint32_t native_reg)
      {
         dirty[native_reg] = true;
         weak[native_reg] = false;
      }

      void flush(psp_gpr_t native_reg, bool keep_dirty = false, bool force = false)
      {
         if ((dirty[native_reg] && !weak[native_reg]) || force)
         {
            write_emu(native_reg, mapping[native_reg]);
            dirty[native_reg] = keep_dirty;
            if (!keep_dirty)
            {
               mapping[native_reg] = 0xFF;
               usage_tag[native_reg] = 0;
            }
         }
      }

      void flush_nds(uint32_t emu_reg_id, bool keep_dirty = false, bool force = false)
      {
         for (int k = 0; k < SRA_REG_COUNT; k ++)
         {
            int i = SRA_REGS[k];
            if (mapping[i] == emu_reg_id)
            {
               flush((psp_gpr_t)i, keep_dirty, force);
            }
         }
      }

      void set_dirty(psp_gpr_t native_reg, bool _dirty)
      {
         dirty[native_reg] = _dirty;
      }


      void flush_all(bool keep_dirty = false)
      {
         for (int k = 0; k < SRA_REG_COUNT; k ++)
         {
            flush(SRA_REGS[k], keep_dirty);
         }
      }

      // Selective spill around an interpreter-fallback (OP_ITP) instruction.
      //
      // The old path was flush_all() + reset(): spill all six cached registers
      // and drop the whole mapping, for every interpreted instruction. With the
      // ITP profiler measuring ~37% fallback that is a lot of store/reload
      // traffic for instructions that usually touch one or two registers.
      //
      // Here only what the instruction actually uses is touched:
      //   read_mask    guest regs the interpreter READS  -> spill if dirty, so
      //                the armcpu_t copy it reads is current. Kept mapped: the
      //                value in the host register is still correct afterwards.
      //   write_mask   guest regs the interpreter WRITES -> the host copy goes
      //                stale, so drop the mapping. No store: whatever is in the
      //                host register is about to be overwritten in memory.
      //                (Spilled first if also read, hence read before write.)
      // Everything outside both masks keeps its mapping and its value.
      void flush_selective(u32 read_mask, u32 write_mask)
      {
         // KEEP-LIST, not a drop-list. The first version spilled the read set,
         // dropped the write set and left everything else mapped -- which was
         // wrong: an interpreter handler touches more than its operands (it
         // updates instruct_adr / next_instruction / R15 and the CPSR flags,
         // and some handlers write registers the encoding does not name), so
         // anything still mapped afterwards could be stale. Zookeeper crashed
         // on exactly that.
         //
         // So: spill every dirty register (the handler reads armcpu_t, and it
         // must be current), then keep the mapping ONLY for registers the
         // instruction provably does not write. Those keep their host copy and
         // do not need reloading. Everything else is unmapped.
         const u32 keep = ~write_mask;

         for (int k = 0; k < SRA_REG_COUNT; k++)
         {
            const psp_gpr_t r = SRA_REGS[k];
            const uint32_t emu = mapping[r];
            if (emu >= 16)
               continue;                       // slot unused

            const bool keep_mapped = (keep & (1u << emu)) != 0;

            // Spill first: the handler reads guest state out of armcpu_t.
            if (dirty[r] && !weak[r])
            {
               write_emu(r, emu);
               dirty[r] = false;
            }

            if (!keep_mapped)
            {
               mapping[r] = 0xFF;
               usage_tag[r] = 0;
               weak[r] = false;
            }
         }

         (void)read_mask;   // reads are covered by spilling every dirty reg
      }


      void push_sp_reg(uint32_t reg_count, int32_t* emu_reg_ids)
      {
         // Find existing registers
         for (uint32_t i = 0; i < reg_count; i ++)
         {
            if (emu_reg_ids[i] > 0)
            {
               if (emu_reg_ids[i] != 15){
                  int32_t current = get_loaded(emu_reg_ids[i] & 0xF, true);

                  if (current >= 0 && sp_reg_pushed[i] == false){
                     emit_mpush(1, (psp_s0 + i));
                     sp_reg_pushed[i] = true;
                  }
               }
            }
         }
      }

      void pull_sp_regs()
      {
         for (int i = 0; i < 30; i ++)
         {
            if (sp_reg_pushed[i])
            {
               emit_mpop(1, psp_s0 + i);
               sp_reg_pushed[i] = false;
            }
         }
      }

   private:
      void read_emu(psp_gpr_t native, int emu)
      {
        loadReg(native, emu);
      }

      void write_emu(psp_gpr_t native, int emu)
      {
        storeReg(native, emu);
      }

   private:
      uint32_t stack_pointer;
      //uint32_t sp_reg[30];
      bool sp_reg_pushed[30];
      uint32_t mapping[30]; // Mapping[native] = emu
      uint32_t usage_tag[30];
      bool dirty[30];
      bool weak[30];

      uint32_t next_usage_tag;
};
