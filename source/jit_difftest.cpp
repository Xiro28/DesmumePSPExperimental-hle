// Differential tester: runs each instruction through the JIT and the interpreter
// from an identical starting state and reports any divergence in registers,
// flags, or a memory window.
//
// It does NOT need a running game: it crafts CPU state and a scratch RAM window,
// then for each opcode in TEST_CASES it compares JIT vs interpreter output.
//
// Triggered from the in-emulator Menu -> "JIT DiffTest" (see ctrlssdl.cpp),
// which calls jit_difftest_run_force(). Output goes to the on-screen debug
// console (DTPRINT). To add coverage, append entries to TEST_CASES below
// (ARM opcodes, cond=AL=0xE). Each case is compared register/flag/memory-wise
// between the interpreter and the JIT; mismatches print [FAIL] lines.
//
// Limitations: R15/PC is not compared (JIT and interpreter advance PC by
// different conventions); the test op is wrapped with a trailing "B ." so the
// JIT block is exactly one instruction.

#include "types.h"
#include <cstdio>
#include <cstring>
#include <cstdarg>
#include <pspdebug.h>
#include <unistd.h>   // getcwd, for the ROMS/difftest.txt log

// Report to BOTH the on-screen debug console and stdout (printf -> PSP console /
// redirected log), so failures can be captured off-device.
// The full report is far larger than the on-screen log keeps, so mirror every
// line into ROMS/difftest.txt as well. Opened lazily on the first DTPRINT of a
// run (dt_log_open() truncates it) and closed by dt_log_close().
static FILE *g_dtlog = NULL;

static void dt_log_open(void)
{
   char path[256];
   if (getcwd(path, sizeof(path) - 32)) {
      strcat(path, "/ROMS/difftest.txt");
      g_dtlog = fopen(path, "w");
   }
}

static void dt_log_close(void)
{
   if (g_dtlog) { fclose(g_dtlog); g_dtlog = NULL; }
}

static void DTPRINT(const char *fmt, ...)
{
   char buf[256];
   va_list ap;
   va_start(ap, fmt);
   vsnprintf(buf, sizeof(buf), fmt, ap);
   va_end(ap);
   pspDebugScreenPrintf("%s", buf);
   printf("%s", buf);
   if (g_dtlog) { fputs(buf, g_dtlog); fflush(g_dtlog); }
}

#include "armcpu.h"
#include "MMU.h"
#include "blockdecoder.h"
#include "arm_jit.h"
#include "instructions.h"

// --- hooks into the JIT (defined in arm_jit.cpp / blockdecoder.cpp) ---
extern block currentBlock;
extern u32 base_adr;
extern "C" uint32_t run_block_chain(armcpu_t *armproc, uintptr_t first_block);
u32 GetFreeSpace();
template <int PROCNUM> u32 arm_jit_compile();
void arm_jit_reset(bool enable, bool suppress_msg);

// Interpreter op tables.
extern const OpFunc arm_instructions_set[2][4096];
extern const OpFunc thumb_instructions_set[2][1024];

// Global scheduler state (defined in NDSSystem.cpp). Declared OUTSIDE the
// anonymous namespace so they resolve to the real globals (not anon-mangled).
extern s32 arm9;
extern s32 s32next;

// Same address blockdecoder.cpp uses; setting it to 1 makes the chain exit
// after the first compiled block instead of chasing the trailing "B ." forever.
#define NDS_ReschedulePtr (*(bool *)(0x0010080))

// Defined in arm_jit.cpp (global linkage -- must be declared OUTSIDE the
// anonymous namespace below, or it mangles to an internal symbol).
extern int g_jitCmpPartial;
extern bool g_last_block_was_native;   // false if the block contained an OP_ITP
extern u32 g_ldm_wb_count;             // LDM/STM writeback executions
extern int g_pctrace_on;               // block-PC ring buffer (arm_jit.cpp)

namespace {

constexpr u32 SCRATCH_BASE = 0x02100000;   // inside main RAM (4MB region @ 0x02000000)
constexpr u32 SCRATCH_WORDS = 64;          // 256-byte window we diff

struct CpuSnapshot {
   u32 R[16];
   u32 cpsr_flags;   // NZCV in bits 31..28, plus we keep full val masked
   u32 mem[SCRATCH_WORDS];
};

// Simple deterministic PRNG so runs are reproducible.
u32 g_rng = 0x12345678u;
inline u32 xrand() { g_rng ^= g_rng << 13; g_rng ^= g_rng >> 17; g_rng ^= g_rng << 5; return g_rng; }

void seed_state(armcpu_t &cpu, u32 instr_adr, u32 flag_seed)
{
   // Random register contents (the same sequence is reused for the interp and
   // jit runs of a given case because the caller reseeds g_rng identically).
   for (int i = 0; i < 16; i++)
      cpu.R[i] = xrand();
   // Base registers for memory ops must point into the scratch window.
   cpu.R[0]  = SCRATCH_BASE + 0x40;
   cpu.R[1]  = SCRATCH_BASE + 0x40;
   cpu.R[13] = SCRATCH_BASE + 0x80;   // SP mid-window so push/pop both fit
   cpu.instruct_adr     = instr_adr;
   cpu.next_instruction = instr_adr + 4;
   cpu.R[15]            = instr_adr + 8;
   // Randomize NZCV (carry-in matters for ADC/SBC/RSC etc.).
   cpu.CPSR.val = (cpu.CPSR.val & ~0xF0000000u) | (flag_seed & 0xF0000000u);
   // continue_cpu_dispatch ends the block chain when any of idle_loop / freeze /
   // freezeBus is set. They carry whatever value the emulator had when the test
   // was launched, so a stale one made multi-block tests execute only their FIRST
   // block: the cross-block consumer never ran and every case whose condition was
   // TRUE reported a bogus register mismatch.
   cpu.idle_loop  = false;
   cpu.freeze     = 0;
   nds.freezeBus  = 0;
}

void seed_mem()
{
   for (u32 i = 0; i < SCRATCH_WORDS; i++)
      _MMU_ARM9_write32(SCRATCH_BASE + i * 4, 0xA5A50000u + i);
}

void take_snapshot(const armcpu_t &cpu, CpuSnapshot &s)
{
   memcpy(s.R, cpu.R, sizeof(s.R));
   s.cpsr_flags = cpu.CPSR.val & 0xF0000000u;
   for (u32 i = 0; i < SCRATCH_WORDS; i++)
      s.mem[i] = _MMU_ARM9_read32(SCRATCH_BASE + i * 4);
}

// Run one instruction through the pure interpreter.
void run_interpreter(armcpu_t &cpu, u32 op, bool thumb)
{
   cpu.instruction = op;
   if (thumb) {
      thumb_instructions_set[0][op >> 6](op);
      return;
   }
   // The ARM instruction functions assume the condition already passed (in the
   // real core, armcpu_exec checks it). For a faithful diff we must honor the
   // condition here too: a false condition makes the instruction a NOP, so the
   // JIT (which DOES honor conditions) can be validated against it.
   const u32 cond = CONDITION(op);
   if (cond != 0xE && !TEST_COND(cond, CODE(op), cpu.CPSR))
      return;   // condition false -> NOP
   arm_instructions_set[0][INSTRUCTION_INDEX(op)](op);
}

// Compile a single-instruction block at base and run it via the chain.
// Returns true if the JIT actually COMPILED the op (false = it fell back to the
// interpreter, in which case JIT and interpreter trivially agree).
bool run_jit(armcpu_t &cpu, u32 op, bool thumb, u32 instr_adr)
{
   // Place the opcode in scratch *code* memory the JIT will read, followed by a
   // branch so the block is exactly one real instruction (block build ends on a
   // branch). 0xEAFFFFFE = "B ." (branch to self), an unconditional branch.
   _MMU_ARM9_write32(instr_adr,     op);
   _MMU_ARM9_write32(instr_adr + 4, 0xEAFFFFFE);

   cpu.instruct_adr     = instr_adr;
   cpu.next_instruction = instr_adr + 4;
   cpu.R[15]            = instr_adr + 8;

   JIT_COMPILED_FUNC(instr_adr, 0) = 0; // force fresh compile of this opcode
   arm_jit_compile<0>();
   uintptr_t blk = JIT_COMPILED_FUNC(instr_adr, 0);
   if (!blk) return false;

   const bool saved_resched = NDS_ReschedulePtr;
   NDS_ReschedulePtr = 1;               // force chain to exit after this one block
   run_block_chain(&cpu, blk);
   NDS_ReschedulePtr = saved_resched;

   // Report whether the op was actually compiled to native code, or fell back to
   // the interpreter inside the block (OP_ITP). Previously this returned plain
   // `true` for "a block was produced", which is true even for a pure OP_ITP
   // block -- so the report printed native=1 for opcodes that the JIT never
   // touched, and their failures looked like codegen bugs.
   return g_last_block_was_native;
}

// ── Multi-op sequence test ───────────────────────────────────────────────────
// The single-instruction tests above cannot catch bugs that only appear when an
// op runs IN CONTEXT: stale condition predicates reused across ops, flags that a
// later op consumes, register-cache interactions. That is exactly how the CMP-imm
// partial-block bug slips through (difftest green, in-game colour corruption).
// Here we compile a SEQUENCE of ops as one block and diff the final state.
constexpr int MAX_SEQ = 8;

// Run a whole sequence through the interpreter, honouring conditions per op.
void run_interpreter_seq(armcpu_t &cpu, const u32 *ops, int n, u32 base)
{
   for (int k = 0; k < n; k++) {
      cpu.instruct_adr     = base + k * 4;
      cpu.next_instruction = base + (k + 1) * 4;
      cpu.R[15]            = base + (k + 2) * 4;
      run_interpreter(cpu, ops[k], false);
   }
}

// Compile+run the sequence as a single JIT block (terminated by "B .").
// Compile with the partial-block CMP gate forced ON. arm_jit_compile() runs with
// full_block=false, where CMP/CMN/TST-immediate are gated on g_jitCmpPartial; with
// the flag off they fall back to INTERPRET, so the "JIT" side of the diff is the
// interpreter itself and the test passes while testing nothing.
static void jit_compile_cmp_on()
{
   const int saved_cmp = g_jitCmpPartial;
   g_jitCmpPartial = 1;
   arm_jit_compile<0>();
   g_jitCmpPartial = saved_cmp;
}

bool run_jit_seq(armcpu_t &cpu, const u32 *ops, int n, u32 base)
{
   for (int k = 0; k < n; k++)
      _MMU_ARM9_write32(base + k * 4, ops[k]);
   _MMU_ARM9_write32(base + n * 4, 0xEAFFFFFEu);   // B .  → ends the block

   for (int k = 0; k <= n; k++)
      JIT_COMPILED_FUNC(base + k * 4, 0) = 0;      // force fresh compile

   cpu.instruct_adr     = base;
   cpu.next_instruction = base + 4;
   cpu.R[15]            = base + 8;

   jit_compile_cmp_on();
   uintptr_t blk = JIT_COMPILED_FUNC(base, 0);
   if (!blk) return false;

   const bool saved = NDS_ReschedulePtr;
   NDS_ReschedulePtr = 1;            // exit the chain after this block
   run_block_chain(&cpu, blk);
   NDS_ReschedulePtr = saved;
   return true;
}

// (check_seq lives further down, after the per-family tracking globals it uses.)

// Did the JIT emit native code for this op (vs interpret-fallback)? We detect
// it by checking whether the compiled block contains an OP_ITP marker: rather
// than parse, we re-run the front-end decoder. Simpler: probe currentBlock after
// compile. We approximate by: an op is "JIT-native" if any ALU/LDR/STR/LDM/STM
// op type was produced. For reporting we just track per-family counts instead.

// --- per-family result tracking ---
struct FamStat {
   char name[24];
   u32 tested, failed, native;
};
constexpr int MAX_FAMS = 512;
FamStat g_fams[MAX_FAMS];
int     g_nfams = 0;
int     g_cur_fam = -1;

int begin_family(const char *name)
{
   if (g_nfams >= MAX_FAMS) { g_cur_fam = MAX_FAMS - 1; return g_cur_fam; }
   int f = g_nfams++;
   strncpy(g_fams[f].name, name, sizeof(g_fams[f].name) - 1);
   g_fams[f].name[sizeof(g_fams[f].name) - 1] = 0;
   g_fams[f].tested = g_fams[f].failed = g_fams[f].native = 0;
   g_cur_fam = f;
   return f;
}

// Distinct failing opcodes (so the same buggy form across seeds prints once).
struct FailRec { u32 op; const char *fam; };
constexpr int MAX_FAILS = 64;
FailRec g_fail_list[MAX_FAILS];
// Separate from g_nfail_list, which saturates at MAX_FAILS(64): the detail dump
// needs to keep going past that so late families (CMP reg-shift, TEQ) get lines.
int g_ndump = 0;
int     g_nfail_list = 0;

armcpu_t *g_cpu = nullptr;
u32       g_adr = 0;

constexpr int SEEDS_PER_OP = 8;            // random register sets tried per opcode

// Run one opcode across several random states, both ways, and diff. The first
// seed that diverges is reported in full; the op is recorded once.
bool check_op(u32 op, bool thumb, const char *name)
{
   armcpu_t &cpu = *g_cpu;
   FamStat &fam = g_fams[g_cur_fam];
   fam.tested++;

   bool native_seen = false;

   for (int seed = 0; seed < SEEDS_PER_OP; seed++)
   {
      const u32 reg_seed  = 0xC0FFEEu * (seed + 1) + op;
      const u32 flag_seed = (op << 3) ^ (seed * 0x40000000u);

      // Snapshot the INPUT state (both runs use the same seeded state).
      g_rng = reg_seed; seed_mem();
      seed_state(cpu, g_adr, flag_seed);
      CpuSnapshot in; take_snapshot(cpu, in);

      run_interpreter(cpu, op, thumb);
      CpuSnapshot itp; take_snapshot(cpu, itp);

      g_rng = reg_seed; seed_mem();
      seed_state(cpu, g_adr, flag_seed);
      g_ldm_wb_count = 0;
      bool native = run_jit(cpu, op, thumb, g_adr);
      CpuSnapshot jit; take_snapshot(cpu, jit);
      native_seen = native_seen || native;

      bool ok = true;
      for (int r = 0; r < 15; r++)          // skip R15: PC handling differs by design
         if (itp.R[r] != jit.R[r]) ok = false;
      if (itp.cpsr_flags != jit.cpsr_flags) ok = false;
      for (u32 m = 0; m < SCRATCH_WORDS; m++)
         if (itp.mem[m] != jit.mem[m]) ok = false;

      if (!ok) {
         fam.failed++;
         if (g_nfail_list < MAX_FAILS)
            g_fail_list[g_nfail_list++] = { op, name };
         // Detailed dump for the first few distinct failures: show the inputs
         // each operand register actually held, then expected (interp) vs jit.
         // Cap raised from 30: ADCS/SBCS/RSCS alone produce ~60 failures and
         // consumed the whole budget, so families listed after them (CMP
         // register-shift, TEQ) reached the summary with no detail lines to
         // diagnose from.
         if (++g_ndump <= 200) {
            DTPRINT("[FAIL] %s op=%08X native=%d seed=%d wb_exec=%u\n",
                    name, op, native, seed, (unsigned)g_ldm_wb_count);
            // inputs: only the registers this opcode reads, plus flags-in.
            const u32 rn = (op >> 16) & 0xF, rm = op & 0xF,
                      rs = (op >> 8) & 0xF, rd = (op >> 12) & 0xF;
            DTPRINT("   in: Rn(%u)=%08X Rm(%u)=%08X Rs(%u)=%08X Rd(%u)=%08X flg=%08X\n",
                    rn, in.R[rn], rm, in.R[rm], rs, in.R[rs], rd, in.R[rd], in.cpsr_flags);
            // differing result registers: expected vs jit.
            for (int r = 0; r < 15; r++)
               if (itp.R[r] != jit.R[r])
                  DTPRINT("   R%-2d expect=%08X jit=%08X\n", r, itp.R[r], jit.R[r]);
            if (itp.cpsr_flags != jit.cpsr_flags) {
               DTPRINT("   flg expect=%08X jit=%08X\n", itp.cpsr_flags, jit.cpsr_flags);
               // Break NZCV out so we see EXACTLY which flag diverges (bit31..28).
               DTPRINT("   N=%d/%d Z=%d/%d C=%d/%d V=%d/%d (expect/jit)\n",
                       (itp.cpsr_flags>>31)&1, (jit.cpsr_flags>>31)&1,
                       (itp.cpsr_flags>>30)&1, (jit.cpsr_flags>>30)&1,
                       (itp.cpsr_flags>>29)&1, (jit.cpsr_flags>>29)&1,
                       (itp.cpsr_flags>>28)&1, (jit.cpsr_flags>>28)&1);
            }
            for (u32 m = 0; m < SCRATCH_WORDS; m++)
               if (itp.mem[m] != jit.mem[m])
                  DTPRINT("   m+%02X(in=%08X) expect=%08X jit=%08X\n",
                          m*4, in.mem[m], itp.mem[m], jit.mem[m]);
         }
         if (native_seen) fam.native++;
         return false;
      }
   }
   if (native_seen) fam.native++;
   return true;
}

// Diff a SEQUENCE compiled as one block: same seeded state into interpreter and
// JIT, compare R0-R14, NZCV and the scratch window. Catches in-context bugs the
// single-instruction check_op cannot see (stale condition predicates, flag reuse).
bool check_seq(const u32 *ops, int n, const char *name)
{
   armcpu_t &cpu = *g_cpu;
   FamStat &fam = g_fams[g_cur_fam];
   fam.tested++;
   bool native_seen = false;

   for (int seed = 0; seed < SEEDS_PER_OP; seed++)
   {
      const u32 reg_seed  = 0xC0FFEEu * (seed + 1) + ops[0] + n;
      const u32 flag_seed = (ops[0] << 3) ^ (seed * 0x40000000u);

      g_rng = reg_seed; seed_mem(); seed_state(cpu, g_adr, flag_seed);
      CpuSnapshot in; take_snapshot(cpu, in);
      run_interpreter_seq(cpu, ops, n, g_adr);
      CpuSnapshot itp; take_snapshot(cpu, itp);

      g_rng = reg_seed; seed_mem(); seed_state(cpu, g_adr, flag_seed);
      const bool native = run_jit_seq(cpu, ops, n, g_adr);
      native_seen = native_seen || native;
      CpuSnapshot jit; take_snapshot(cpu, jit);

      bool ok = true;
      for (int r = 0; r < 15; r++) if (itp.R[r] != jit.R[r]) ok = false;
      if (itp.cpsr_flags != jit.cpsr_flags) ok = false;
      for (u32 m = 0; m < SCRATCH_WORDS; m++) if (itp.mem[m] != jit.mem[m]) ok = false;

      if (!ok) {
         fam.failed++;
         if (g_nfail_list < MAX_FAILS) g_fail_list[g_nfail_list++] = { ops[0], name };
         DTPRINT("[SEQFAIL] %s native=%d seed=%d\n", name, native, seed);
         for (int k = 0; k < n; k++) DTPRINT("   op%d=%08X\n", k, ops[k]);
         DTPRINT("   flg in=%08X expect=%08X jit=%08X\n",
                 in.cpsr_flags, itp.cpsr_flags, jit.cpsr_flags);
         if (itp.cpsr_flags != jit.cpsr_flags)
            DTPRINT("   N=%d/%d Z=%d/%d C=%d/%d V=%d/%d (expect/jit)\n",
                    (itp.cpsr_flags>>31)&1, (jit.cpsr_flags>>31)&1,
                    (itp.cpsr_flags>>30)&1, (jit.cpsr_flags>>30)&1,
                    (itp.cpsr_flags>>29)&1, (jit.cpsr_flags>>29)&1,
                    (itp.cpsr_flags>>28)&1, (jit.cpsr_flags>>28)&1);
         for (int r = 0; r < 15; r++)
            if (itp.R[r] != jit.R[r])
               DTPRINT("   R%-2d in=%08X expect=%08X jit=%08X\n", r, in.R[r], itp.R[r], jit.R[r]);
         for (u32 m = 0; m < SCRATCH_WORDS; m++)
            if (itp.mem[m] != jit.mem[m])
               DTPRINT("   m+%02X expect=%08X jit=%08X\n", m*4, itp.mem[m], jit.mem[m]);
         if (native_seen) fam.native++;
         return false;
      }
   }
   if (native_seen) fam.native++;
   return true;
}

// Build an ARM data-processing opcode (immediate or register form).
inline u32 dp(u32 cond, u32 opc, u32 s, u32 rn, u32 rd, u32 operand2, bool imm)
{
   // TST/TEQ/CMP/CMN (opc 8..11) have no destination and exist ONLY with S=1:
   // that same encoding with S=0 is the MRS/MSR space, a completely different
   // instruction. The sweep loops s over {0,1} for all 16 opcodes, so the s==0
   // half was emitting MSR/MRS and comparing it as if it were a compare -- both
   // interpreter and JIT wrote registers (visible as R2/R3 changing on a "CMP"),
   // and the mismatch was reported against the CMP families. Force S on.
   if (opc >= 8 && opc <= 11)
      s = 1;
   return (cond << 28) | (imm ? (1u << 25) : 0) | (opc << 21) | (s << 20) |
          (rn << 16) | (rd << 12) | (operand2 & 0xFFF);
}

// Sweep all data-processing opcodes. Each (opcode x operand-form) is its own
// reported family, so the summary says exactly e.g. "ADD imm" passes but
// "ADD ror" / "ADD rrx" fails. Forms:
//   imm   : rotated 8-bit immediate
//   lsl   : Rm, LSL #n   (n in {0,1,31})
//   lsr   : Rm, LSR #n
//   asr   : Rm, ASR #n
//   ror   : Rm, ROR #n   (n in {1,31}; n=0 is RRX, tested separately)
//   rrx   : Rm, ROR #0  == RRX (rotate right through carry)
//   lslr  : Rm, LSL Rs   (register-specified shift amount)
//   lsrr/asrr/rorr similarly
void sweep_dataproc()
{
   static const char *opc_names[16] = {
      "AND","EOR","SUB","RSB","ADD","ADC","SBC","RSC",
      "TST","TEQ","CMP","CMN","ORR","MOV","BIC","MVN" };
   const u32 rn = 3, rd = 2, rm = 4, rs = 6;

   for (u32 opc = 0; opc < 16; opc++)
   for (u32 s = 0; s <= 1; s++)
   {
      // Skip families that are ARM_OP_UNDEF in arm_jit.cpp and therefore always
      // interpreted in-game: ADC_S(5) SBC_S(6) RSC_S(7) TEQ(9) MOV_S(13) MVN_S(15)
      // and CMN(11). Their ~70 failures are real codegen bugs but in DEAD code,
      // and they were crowding the detail dump out for the families that are live.
      // Flip g_dt_skip_dead to 0 to test them again.
      static const int g_dt_skip_dead = 1;
      if (g_dt_skip_dead && s == 1 &&
          (opc == 5 || opc == 6 || opc == 7 || opc == 13 || opc == 15))
         continue;
      if (g_dt_skip_dead && s == 0 && (opc == 9))
         continue;
      const char *suf = s ? "S" : "";
      char fam[24];

      // immediate
      snprintf(fam, sizeof(fam), "%s%s imm", opc_names[opc], suf);
      begin_family(fam);
      for (u32 imm8 : {0u, 1u, 0xFFu, 0x80u})
      for (u32 rot  : {0u, 4u, 8u})            // rotate field too
         check_op(dp(0xE, opc, s, rn, rd, (rot << 8) | imm8, true), false, fam);

      // immediate-shifted register: LSL/LSR/ASR/ROR by #1 and #31
      static const char *shn[4] = { "lsl", "lsr", "asr", "ror" };
      for (u32 sh = 0; sh < 4; sh++) {
         snprintf(fam, sizeof(fam), "%s%s %s", opc_names[opc], suf, shn[sh]);
         begin_family(fam);
         for (u32 amt : {1u, 5u, 31u}) {
            u32 op2 = (amt << 7) | (sh << 5) | rm;
            check_op(dp(0xE, opc, s, rn, rd, op2, false), false, fam);
         }
      }

      // special encodings where shift-amount #0 means something else:
      // LSL #0 = plain Rm ; LSR #0 = LSR #32 ; ASR #0 = ASR #32 ; ROR #0 = RRX
      snprintf(fam, sizeof(fam), "%s%s lsl0", opc_names[opc], suf); begin_family(fam);
      check_op(dp(0xE, opc, s, rn, rd, (0u<<5)|rm, false), false, fam);     // LSL #0
      snprintf(fam, sizeof(fam), "%s%s lsr32", opc_names[opc], suf); begin_family(fam);
      check_op(dp(0xE, opc, s, rn, rd, (1u<<5)|rm, false), false, fam);     // LSR #0(=32)
      snprintf(fam, sizeof(fam), "%s%s asr32", opc_names[opc], suf); begin_family(fam);
      check_op(dp(0xE, opc, s, rn, rd, (2u<<5)|rm, false), false, fam);     // ASR #0(=32)
      snprintf(fam, sizeof(fam), "%s%s rrx", opc_names[opc], suf); begin_family(fam);
      check_op(dp(0xE, opc, s, rn, rd, (3u<<5)|rm, false), false, fam);     // ROR #0 = RRX

      // register-specified shift amount: LSL/LSR/ASR/ROR by Rs
      static const char *shr[4] = { "lslr", "lsrr", "asrr", "rorr" };
      for (u32 sh = 0; sh < 4; sh++) {
         snprintf(fam, sizeof(fam), "%s%s %s", opc_names[opc], suf, shr[sh]);
         begin_family(fam);
         u32 op2 = (rs << 8) | (sh << 5) | (1u << 4) | rm;   // bit4=1 => reg shift
         check_op(dp(0xE, opc, s, rn, rd, op2, false), false, fam);
      }
   }
}

// Sweep CONDITIONAL data-processing S-ops. The main sweep only uses cond=AL, so
// it can't see bugs in the conditional commit path (result OR flags must stay
// unchanged when the condition is false). Here we use EQ (0x0) and NE (0x1):
// seed_state randomizes NZCV per seed, so across SEEDS_PER_OP roughly half the
// runs have the condition false — both interp and jit must then leave Rd and the
// flags exactly as they were on entry.
void sweep_dataproc_cond()
{
   static const char *opc_names[16] = {
      "AND","EOR","SUB","RSB","ADD","ADC","SBC","RSC",
      "TST","TEQ","CMP","CMN","ORR","MOV","BIC","MVN" };
   // Only the opcodes we JIT with flags are interesting here.
   static const u32 opcs[] = { 0,2,4,12 };   // AND, SUB, ADD, ORR (+ EOR via 1)
   static const u32 conds[] = { 0x0 /*EQ*/, 0x1 /*NE*/ };
   const u32 rn = 3, rd = 2, rm = 4;
   for (u32 opc : { 0u,1u,2u,4u,12u })       // AND EOR SUB ADD ORR
   for (u32 cond : conds)
   {
      char fam[24];
      // immediate form
      snprintf(fam, sizeof(fam), "%sS %s imm", opc_names[opc], cond ? "NE" : "EQ");
      begin_family(fam);
      for (u32 imm8 : {1u, 0x80u, 0xFFu})
         check_op(dp(cond, opc, 1, rn, rd, imm8, true), false, fam);
      // a register-shift form (LSL #3) to exercise the shifted path too
      snprintf(fam, sizeof(fam), "%sS %s lsl3", opc_names[opc], cond ? "NE" : "EQ");
      begin_family(fam);
      check_op(dp(cond, opc, 1, rn, rd, (3u<<7)|(0u<<5)|rm, false), false, fam);
   }
   (void)opcs; (void)rn;
}

// Branch tester. Unlike check_op (which skips R15 and appends a trailing "B ."),
// this compares the FINAL control-flow state after the branch: next_instruction
// (where execution actually continues), R15, R14 (link), and CPSR/flags. With
// NDS_ReschedulePtr forced to 1 inside run_jit, the branch helper does NOT recurse
// into the target block — it only sets the registers — which is exactly what we
// want to validate against the interpreter.
bool check_branch(u32 op, const char *name)
{
   armcpu_t &cpu = *g_cpu;
   FamStat &fam = g_fams[g_cur_fam];
   fam.tested++;
   bool native_seen = false;

   for (int seed = 0; seed < SEEDS_PER_OP; seed++)
   {
      const u32 reg_seed  = 0xC0FFEEu * (seed + 1) + op;
      const u32 flag_seed = (op << 3) ^ (seed * 0x40000000u);

      g_rng = reg_seed; seed_mem(); seed_state(cpu, g_adr, flag_seed);
      run_interpreter(cpu, op, false);
      const u32 i_next = cpu.next_instruction, i_r15 = cpu.R[15],
                i_r14 = cpu.R[14], i_cpsr = cpu.CPSR.val;

      g_rng = reg_seed; seed_mem(); seed_state(cpu, g_adr, flag_seed);
      const bool native = run_jit(cpu, op, false, g_adr);
      native_seen = native_seen || native;
      const u32 j_next = cpu.next_instruction, j_r15 = cpu.R[15],
                j_r14 = cpu.R[14], j_cpsr = cpu.CPSR.val;

      if (i_next != j_next || i_r14 != j_r14 || i_cpsr != j_cpsr)
      {
         fam.failed++;
         DTPRINT("[FAIL] %s op=%08X native=%d seed=%d wb_exec=%u\n",
                    name, op, native, seed, (unsigned)g_ldm_wb_count);
         DTPRINT("   next exp=%08X jit=%08X  R15 exp=%08X jit=%08X\n", i_next, j_next, i_r15, j_r15);
         DTPRINT("   R14  exp=%08X jit=%08X  cpsr exp=%08X jit=%08X\n", i_r14, j_r14, i_cpsr, j_cpsr);
         if (native_seen) fam.native++;
         return false;
      }
   }
   if (native_seen) fam.native++;
   return true;
}

// Chaining test: does a JIT branch actually transfer control to the target block
// AND execute it? Single-instruction check_branch only validates register setup;
// this runs a real multi-block chain (NDS_ReschedulePtr=0) bounded by a cycle
// budget. branch at g_adr -> target g_adr+0x40 which is [MOV r5,#0x42 ; B .].
// If r5 becomes 0x42, the branch chained to and ran the target.
void check_branch_chain(u32 branch_op, const char *name)
{
   armcpu_t &cpu = *g_cpu;
   const u32 target = g_adr + 0x40;

   seed_mem();
   _MMU_ARM9_write32(g_adr,      branch_op);
   _MMU_ARM9_write32(g_adr + 4,  0xEAFFFFFEu);   // guard (not reached if branch taken)
   _MMU_ARM9_write32(target,     0xE3A05042u);   // MOV r5, #0x42
   _MMU_ARM9_write32(target + 4, 0xEAFFFFFEu);   // B . (self loop -> exits on cycle budget)

   seed_state(cpu, g_adr, 0);
   cpu.R[5] = 0;

   JIT_COMPILED_FUNC(g_adr, 0)      = 0;
   JIT_COMPILED_FUNC(target, 0)     = 0;
   JIT_COMPILED_FUNC(target + 4, 0) = 0;

   // Diagnostic: can we even compile a block AT the target address?
   cpu.CPSR.bits.T = 0;
   cpu.instruct_adr = target; cpu.next_instruction = target + 4; cpu.R[15] = target + 8;
   arm_jit_compile<0>();
   DTPRINT("[CHAIN] %s: target precompile -> %08X\n", name, (u32)JIT_COMPILED_FUNC(target, 0));

   // Reset to the branch and run the chain.
   cpu.instruct_adr = g_adr; cpu.next_instruction = g_adr + 4; cpu.R[15] = g_adr + 8;
   cpu.R[5] = 0;
   JIT_COMPILED_FUNC(g_adr, 0) = 0;

   arm9 = 0; s32next = 100000;
   const bool saved = NDS_ReschedulePtr;
   NDS_ReschedulePtr = 0;                         // ALLOW chaining (the real in-game path)

   // Mimic exec_cpu's outer loop: look up the block at instruct_adr, compile if
   // needed, run it; the chain may exit (idle/event) — then re-enter. A correct
   // branch must eventually transfer control so the target's MOV runs (r5=0x42).
   for (int k = 0; k < 8 && cpu.R[5] != 0x42; k++)
   {
      uintptr_t b = JIT_COMPILED_FUNC(cpu.instruct_adr, 0);
      if (!b) { jit_compile_cmp_on(); b = JIT_COMPILED_FUNC(cpu.instruct_adr, 0); }
      if (!b) { DTPRINT("[CHAIN] %s: instr_adr=%08X not compiled\n", name, cpu.instruct_adr); break; }
      run_block_chain(&cpu, b);
   }
   NDS_ReschedulePtr = saved;

   const bool ok = (cpu.R[5] == 0x42);
   DTPRINT("[CHAIN] %s: r5=%08X(want 42) next=%08X %s\n",
           name, cpu.R[5], cpu.next_instruction, ok ? "OK" : "FAIL");
}

// Conditional-branch chaining test. Validates BOTH outcomes of a conditional
// branch chain to the correct target block — the path conditional-branch slot
// linking touches. taken_flags makes the condition TRUE, nottaken_flags FALSE.
//   layout: g_adr   = <cond branch> target +0x40
//           g_adr+4 = MOV r6,#0x99 ; B .   (not-taken fall-through block)
//           g_adr+0x40 = MOV r5,#0x42 ; B . (taken target block)
// taken run: r5 must become 0x42 (and r6 stay 0). not-taken: r6 -> 0x99, r5 stay 0.
void check_cond_chain(u32 branch_op, const char *name, u32 taken_flags, u32 nottaken_flags)
{
   armcpu_t &cpu = *g_cpu;
   const u32 a = g_adr, target = a + 0x40;

   for (int taken = 0; taken <= 1; taken++)
   {
      seed_mem();
      _MMU_ARM9_write32(a,        branch_op);
      _MMU_ARM9_write32(a + 4,    0xE3A06099u);   // MOV r6,#0x99 (not-taken block)
      _MMU_ARM9_write32(a + 8,    0xEAFFFFFEu);   // B .
      _MMU_ARM9_write32(target,   0xE3A05042u);   // MOV r5,#0x42 (taken block)
      _MMU_ARM9_write32(target+4, 0xEAFFFFFEu);   // B .

      JIT_COMPILED_FUNC(a, 0)      = 0;
      JIT_COMPILED_FUNC(a + 4, 0)  = 0;
      JIT_COMPILED_FUNC(a + 8, 0)  = 0;
      JIT_COMPILED_FUNC(target, 0) = 0;
      JIT_COMPILED_FUNC(target+4,0) = 0;

      seed_state(cpu, a, taken ? taken_flags : nottaken_flags);
      cpu.R[5] = 0; cpu.R[6] = 0;
      cpu.instruct_adr = a; cpu.next_instruction = a + 4; cpu.R[15] = a + 8;

      arm9 = 0; s32next = 100000;
      const bool saved = NDS_ReschedulePtr;
      NDS_ReschedulePtr = 0;
      for (int k = 0; k < 8 && cpu.R[5] != 0x42 && cpu.R[6] != 0x99; k++)
      {
         uintptr_t b = JIT_COMPILED_FUNC(cpu.instruct_adr, 0);
         if (!b) { jit_compile_cmp_on(); b = JIT_COMPILED_FUNC(cpu.instruct_adr, 0); }
         if (!b) break;
         run_block_chain(&cpu, b);
      }
      NDS_ReschedulePtr = saved;

      const bool ok = taken ? (cpu.R[5] == 0x42 && cpu.R[6] == 0)
                            : (cpu.R[6] == 0x99 && cpu.R[5] == 0);
      DTPRINT("[CCHAIN] %s %s: r5=%08X r6=%08X next=%08X %s\n",
              name, taken ? "taken" : "nottk", cpu.R[5], cpu.R[6],
              cpu.next_instruction, ok ? "OK" : "FAIL");
   }
}

// Fall-through chaining test: a block that ends WITHOUT a branch must transfer
// control to the sequentially-next block. This exercises the self-healing slot
// dispatch on the fall-through path (the hottest transition). We force the first
// block to end early by compiling a 1-instruction block whose epilogue falls
// through to g_adr+4, where a [MOV r5,#0x42 ; B .] lives. The key part: the target
// block is compiled AFTER the first one (mimicking the real compile-later case the
// self-healing slot must handle), so we compile block1 first, then let the chain
// resolve the fall-through target lazily.
void check_fallthrough_chain(const char *name)
{
   armcpu_t &cpu = *g_cpu;
   const u32 a = g_adr;

   seed_mem();
   _MMU_ARM9_write32(a,      0xE1A00000u);   // NOP (MOV r0,r0) — no branch, falls through
   _MMU_ARM9_write32(a + 4,  0xE3A05042u);   // MOV r5,#0x42  (the "next" block)
   _MMU_ARM9_write32(a + 8,  0xEAFFFFFEu);   // B .

   JIT_COMPILED_FUNC(a, 0)     = 0;
   JIT_COMPILED_FUNC(a + 4, 0) = 0;
   JIT_COMPILED_FUNC(a + 8, 0) = 0;

   // Pre-compile the block at a+4 so that exp_early_term makes the block at `a`
   // STOP after the NOP and fall through to a+4 — creating two real adjacent blocks
   // with a sequential transition, which is exactly the fall-through path we test.
   cpu.CPSR.bits.T = 0;
   cpu.instruct_adr = a + 4; cpu.next_instruction = a + 8; cpu.R[15] = a + 12;
   arm_jit_compile<0>();
   DTPRINT("[FALLT] %s: a+4 precompile -> %08X\n", name, (u32)JIT_COMPILED_FUNC(a + 4, 0));

   seed_state(cpu, a, 0);
   cpu.R[5] = 0;
   cpu.instruct_adr = a; cpu.next_instruction = a + 4; cpu.R[15] = a + 8;

   arm9 = 0; s32next = 100000;
   const bool saved = NDS_ReschedulePtr;
   NDS_ReschedulePtr = 0;
   for (int k = 0; k < 8 && cpu.R[5] != 0x42; k++)
   {
      uintptr_t b = JIT_COMPILED_FUNC(cpu.instruct_adr, 0);
      if (!b) { jit_compile_cmp_on(); b = JIT_COMPILED_FUNC(cpu.instruct_adr, 0); }
      if (!b) { DTPRINT("[FALLT] %s: instr_adr=%08X not compiled\n", name, cpu.instruct_adr); break; }
      run_block_chain(&cpu, b);
   }
   NDS_ReschedulePtr = saved;

   const bool ok = (cpu.R[5] == 0x42);
   DTPRINT("[FALLT] %s: r5=%08X(want 42) next=%08X %s\n",
           name, cpu.R[5], cpu.next_instruction, ok ? "OK" : "FAIL");
}

void sweep_branch()
{
   // pc at g_adr; R15 = g_adr+8 at execute. B target = R15 + (signext(off24)<<2).
   begin_family("B fwd");   check_branch(0xEA00000Eu, "B fwd");    // +0x40
   begin_family("B back");  check_branch(0xEAFFFFF8u, "B back");   // -0x18
   begin_family("BL fwd");  check_branch(0xEB00000Eu, "BL fwd");
   begin_family("BL back"); check_branch(0xEBFFFFF8u, "BL back");
   // Conditional branches (EQ/NE): seeds vary flags so both taken & not-taken hit.
   begin_family("Beq");     check_branch(0x0A00000Eu, "Beq");
   begin_family("Bne");     check_branch(0x1A00000Eu, "Bne");
   begin_family("BLeq");    check_branch(0x0B00000Eu, "BLeq");
   begin_family("BLne");    check_branch(0x1B00000Eu, "BLne");

   // Real chaining (NDS_ReschedulePtr=0): branch must reach+run the target block.
   check_branch_chain(0xEA00000Eu, "B->target");
   check_branch_chain(0xEB00000Eu, "BL->target");
   // Fall-through chaining: block with no branch must reach the next block.
   check_fallthrough_chain("fallthrough->next");
   // Conditional-branch chaining: both taken and not-taken must reach the right block.
   // BEQ taken when Z=1 (0x40000000); BNE taken when Z=0.
   check_cond_chain(0x0A00000Eu, "Beq", 0x40000000u, 0x00000000u);
   check_cond_chain(0x1A00000Eu, "Bne", 0x00000000u, 0x40000000u);
}

// Sweep LDR/STR (word & byte) immediate and register offset, pre/post, up/down,
// with and without writeback.
void sweep_ldrstr()
{
   begin_family("LDR/STR");
   for (u32 L = 0; L <= 1; L++)             // 0=STR 1=LDR
   for (u32 B = 0; B <= 1; B++)             // byte
   for (u32 P = 0; P <= 1; P++)             // pre/post
   for (u32 U = 0; U <= 1; U++)             // up/down
   for (u32 W = 0; W <= 1; W++)             // writeback (only valid with P, or post)
   for (u32 I = 0; I <= 1; I++)             // 0=imm offset 1=reg offset
   {
      const u32 rn = 0 /*base=scratch+0x40*/, rd = 2;
      u32 off = I ? 4 /*rm=r4*/ : 0x10;     // imm12 = 16 bytes
      u32 op = (0xEu << 28) | (1u << 26) | (I << 25) | (P << 24) | (U << 23) |
               (B << 22) | (W << 21) | (L << 20) | (rn << 16) | (rd << 12) | off;
      check_op(op, false, "ldr/str");
   }
}

// Sweep LDM/STM across P/U/W/L and several register-list patterns.
void sweep_ldmstm()
{
   begin_family("LDM/STM");
   const u32 lists[] = { 0x000C, 0x001C, 0x003C, 0x0006, 0x00FF, 0x8004 /*incl PC*/, 0x4004 /*incl LR*/ };
   for (u32 P = 0; P <= 1; P++)
   for (u32 U = 0; U <= 1; U++)
   for (u32 W = 0; W <= 1; W++)
   for (u32 L = 0; L <= 1; L++)
   for (u32 rlist : lists)
   {
      const u32 rn = 0; // base = scratch+0x40
      u32 op = (0xEu << 28) | (1u << 27) | (P << 24) | (U << 23) |
               (W << 21) | (L << 20) | (rn << 16) | rlist;
      check_op(op, false, "ldm/stm");
   }
}

// Sweep realistic CMP-in-context sequences — the pattern that breaks in-game but
// passes the single-instruction tests. Each case is a CMP followed by conditional
// ops that CONSUME the flags it produced. Same-condition runs also exercise the
// condition-merge ($s4 reuse) path.
void sweep_cmp_sequences()
{
   const u32 rn = 3, rd = 2, rd2 = 5;
   // helpers: dp(cond, opc, s, rn, rd, operand2, imm)
   //   opc: 4=ADD 13=MOV 10=CMP 0=AND 12=ORR ; cond: E=AL 0=EQ 1=NE B=LT A=GE

   struct Case { const char *name; int n; u32 ops[MAX_SEQ]; };
   Case cases[] = {
      // CMP then ONE conditional op (the classic cmp/moveq)
      { "cmp;moveq", 2, {
          dp(0xE, 10, 1, rn, 0, 0x10, true),          // CMP rn,#0x10
          dp(0x0, 13, 0, 0, rd, 0x01, true) } },      // MOVEQ rd,#1
      // CMP then two ops with the SAME condition → condition-merge reuse of $s4
      { "cmp;moveq;addeq", 3, {
          dp(0xE, 10, 1, rn, 0, 0x10, true),
          dp(0x0, 13, 0, 0, rd, 0x01, true),
          dp(0x0,  4, 0, rn, rd2, 0x02, true) } },
      // CMP then opposite conditions (both paths must see the same flags)
      { "cmp;moveq;movne", 3, {
          dp(0xE, 10, 1, rn, 0, 0x10, true),
          dp(0x0, 13, 0, 0, rd, 0x01, true),
          dp(0x1, 13, 0, 0, rd2, 0x02, true) } },
      // Two CMPs in a row: the second must not reuse the first's predicate
      { "cmp;movlt;cmp;movge", 4, {
          dp(0xE, 10, 1, rn, 0, 0x10, true),
          dp(0xB, 13, 0, 0, rd, 0x01, true),          // MOVLT
          dp(0xE, 10, 1, rn, 0, 0x20, true),          // CMP rn,#0x20
          dp(0xA, 13, 0, 0, rd2, 0x02, true) } },     // MOVGE
      // CMP then a flag-setting op then a conditional (flags must come from the S-op)
      { "cmp;addseq;movcs", 3, {
          dp(0xE, 10, 1, rn, 0, 0x10, true),
          dp(0x0,  4, 1, rn, rd, 0xFF, true),         // ADDEQS rd,rn,#0xFF
          dp(0x2, 13, 0, 0, rd2, 0x03, true) } },     // MOVCS
      // conditional CMP (its own flags update is predicated)
      { "cmpeq;movne", 2, {
          dp(0x0, 10, 1, rn, 0, 0x10, true),          // CMPEQ
          dp(0x1, 13, 0, 0, rd, 0x01, true) } },
      // longer same-condition run
      { "cmp;4x eq-ops", 5, {
          dp(0xE, 10, 1, rn, 0, 0x10, true),
          dp(0x0, 13, 0, 0, rd, 0x01, true),
          dp(0x0,  4, 0, rn, rd2, 0x02, true),
          dp(0x0, 12, 0, rn, rd, 0x04, true),
          dp(0x0,  0, 0, rn, rd2, 0x0F, true) } },
   };

   for (auto &c : cases) {
      begin_family(c.name);
      check_seq(c.ops, c.n, c.name);
   }
}


// ── Cross-BLOCK flag test ────────────────────────────────────────────────────
// The sequence test proves CMP is correct WITHIN one block. In-game the consumer
// of a CMP usually lives in the NEXT block (the block ends at a branch). Flags
// must survive the boundary via store_flags -> ARMPROC.CPSR -> load_flags. Layout:
//    block A @ base   : <flag-setting op> ; B (to base+8)   -> ends block A
//    block B @ base+8 : <conditional consumer> ; B .        -> self loop, chain exits
void check_xblock(u32 setter, u32 consumer, const char *name)
{
   armcpu_t &cpu = *g_cpu;
   FamStat &fam = g_fams[g_cur_fam];
   fam.tested++;

   for (int seed = 0; seed < SEEDS_PER_OP; seed++)
   {
      const u32 reg_seed  = 0xC0FFEEu * (seed + 1) + setter;
      const u32 flag_seed = (setter << 3) ^ (seed * 0x40000000u);
      const u32 a = g_adr;
      // B at a+4: its PC is (a+4)+8 = a+12, so offset 0 lands on a+12.
      // Layout: a=setter, a+4=B(->a+12), a+8=unused, a+12=consumer, a+16=B .
      const u32 bfwd = 0xEA000000u;

      g_rng = reg_seed; seed_mem(); seed_state(cpu, a, flag_seed);
      CpuSnapshot in; take_snapshot(cpu, in);
      cpu.instruct_adr = a;     cpu.next_instruction = a + 4;  cpu.R[15] = a + 8;
      run_interpreter(cpu, setter, false);
      cpu.instruct_adr = a + 12; cpu.next_instruction = a + 16; cpu.R[15] = a + 20;
      run_interpreter(cpu, consumer, false);
      CpuSnapshot itp; take_snapshot(cpu, itp);

      g_rng = reg_seed; seed_mem(); seed_state(cpu, a, flag_seed);
      _MMU_ARM9_write32(a,      setter);
      _MMU_ARM9_write32(a + 4,  bfwd);
      _MMU_ARM9_write32(a + 8,  0xE1A00000u);   // NOP (never executed)
      _MMU_ARM9_write32(a + 12, consumer);
      _MMU_ARM9_write32(a + 16, 0xEAFFFFFEu);   // B .
      for (int k = 0; k < 5; k++) JIT_COMPILED_FUNC(a + k*4, 0) = 0;

      cpu.instruct_adr = a; cpu.next_instruction = a + 4; cpu.R[15] = a + 8;
      arm9 = 0; s32next = 100000;
      const bool saved = NDS_ReschedulePtr;
      NDS_ReschedulePtr = 0;
      int trace_n = 0; u32 trace_pc[8] = {0};
      for (int k = 0; k < 6; k++) {
         // Re-clear per hop: a compiled block can set idle_loop itself, and the
         // dispatcher would then refuse to chain into the next block.
         cpu.idle_loop = false; cpu.freeze = 0; nds.freezeBus = 0;
         if (trace_n < 8) trace_pc[trace_n++] = cpu.instruct_adr;
         uintptr_t b = JIT_COMPILED_FUNC(cpu.instruct_adr, 0);
         if (!b) { jit_compile_cmp_on(); b = JIT_COMPILED_FUNC(cpu.instruct_adr, 0); }
         if (!b) break;
         const u32 rbc = run_block_chain(&cpu, b);
         if (trace_n <= 8)
            DTPRINT("   [hop%d] blk=%08X -> pc=%08X rbc=%u free=%u\n",
                    k, (u32)b, cpu.instruct_adr, rbc, (unsigned)GetFreeSpace());
         if (cpu.instruct_adr == a + 16) break;
      }
      NDS_ReschedulePtr = saved;
      CpuSnapshot jit; take_snapshot(cpu, jit);

      bool ok = true;
      for (int r = 0; r < 15; r++) if (itp.R[r] != jit.R[r]) ok = false;
      if (itp.cpsr_flags != jit.cpsr_flags) ok = false;

      if (!ok) {
         fam.failed++;
         DTPRINT("[XBLK] %s seed=%d set=%08X cons=%08X\n", name, seed, setter, consumer);
         DTPRINT("   base=%08X final_pc=%08X steps=%d\n", a, cpu.instruct_adr, trace_n);
         DTPRINT("   blk_a=%08X blk_a12=%08X resched=%d idle=%d frz=%d fbus=%d\n",
                 (u32)JIT_COMPILED_FUNC(a, 0), (u32)JIT_COMPILED_FUNC(a + 12, 0),
                 (int)NDS_ReschedulePtr, (int)cpu.idle_loop, (int)cpu.freeze,
                 (int)nds.freezeBus);
         for (int t = 0; t < trace_n; t++) DTPRINT("   step%d pc=%08X\n", t, trace_pc[t]);
         DTPRINT("   flg in=%08X expect=%08X jit=%08X\n",
                 in.cpsr_flags, itp.cpsr_flags, jit.cpsr_flags);
         DTPRINT("   N=%d/%d Z=%d/%d C=%d/%d V=%d/%d (expect/jit)\n",
                 (itp.cpsr_flags>>31)&1, (jit.cpsr_flags>>31)&1,
                 (itp.cpsr_flags>>30)&1, (jit.cpsr_flags>>30)&1,
                 (itp.cpsr_flags>>29)&1, (jit.cpsr_flags>>29)&1,
                 (itp.cpsr_flags>>28)&1, (jit.cpsr_flags>>28)&1);
         for (int r = 0; r < 15; r++)
            if (itp.R[r] != jit.R[r])
               DTPRINT("   R%-2d in=%08X expect=%08X jit=%08X\n", r, in.R[r], itp.R[r], jit.R[r]);
         return;
      }
   }
   fam.native++;
}

void sweep_cmp_xblock()
{
   const u32 rn = 3, rd = 2;
   struct XC { const char *name; u32 set; u32 cons; };
   XC cases[] = {
      { "x:cmp|moveq",  dp(0xE,10,1,rn,0,0x10,true), dp(0x0,13,0,0,rd,0x01,true) },
      { "x:cmp|movne",  dp(0xE,10,1,rn,0,0x10,true), dp(0x1,13,0,0,rd,0x02,true) },
      { "x:cmp|movlt",  dp(0xE,10,1,rn,0,0x10,true), dp(0xB,13,0,0,rd,0x03,true) },
      { "x:cmp|movge",  dp(0xE,10,1,rn,0,0x10,true), dp(0xA,13,0,0,rd,0x04,true) },
      { "x:cmp|movcs",  dp(0xE,10,1,rn,0,0x10,true), dp(0x2,13,0,0,rd,0x05,true) },
      { "x:cmp|movvs",  dp(0xE,10,1,rn,0,0x10,true), dp(0x6,13,0,0,rd,0x06,true) },
      { "x:adds|moveq", dp(0xE, 4,1,rn,rd,0x10,true),dp(0x0,13,0,0,rd,0x07,true) },
   };
   for (auto &c : cases) { begin_family(c.name); check_xblock(c.set, c.cons, c.name); }
}

// Sweep MUL / MLA across register choices.
void sweep_mul()
{
   begin_family("MUL/MLA");
   for (u32 s = 0; s <= 1; s++) {
      // MUL rd, rm, rs : rd=2 rm=3 rs=4
      check_op((0xEu<<28)|(s<<20)|(2u<<16)|(4u<<8)|(9u<<4)|3u, false, "mul");
      // MLA rd, rm, rs, rn : rd=2 rn=5 rs=4 rm=3
      check_op((0xEu<<28)|(1u<<21)|(s<<20)|(2u<<16)|(5u<<12)|(4u<<8)|(9u<<4)|3u, false, "mla");
   }
}

} // namespace

// Force-run: invoked from the ROM-select menu. Tests every instruction family
// the JIT handles, sweeping operand/addressing/condition variations, and diffs
// each against the interpreter.
extern "C" void jit_difftest_run_force()
{
   DTPRINT("=== JIT vs interpreter difftest ===\n");

   // Ensure JIT memory is initialized (the ROM-select menu may run before any
   // ROM has been loaded / MMU_Reset has set the JIT up).
   arm_jit_reset(true, true);

   armcpu_t &cpu = NDS_ARM9;
   armcpu_t saved = cpu;          // restore real CPU afterwards

   g_cpu = &cpu;
   g_adr = SCRATCH_BASE + 0x200;
   g_nfams = 0; g_nfail_list = 0; g_ndump = 0; g_cur_fam = -1;
   dt_log_open();
   // The difftest compiles thousands of one-instruction blocks at a single
   // address, which floods the PC ring buffer and hides the game's trace.
   // Turn recording off for the duration of the suite.
   const int saved_pctrace = g_pctrace_on;
   g_pctrace_on = 0;

   DTPRINT("=== DIFFTEST BUILD MARKER 1924 ===\n");
   // ORDER MATTERS: the per-case [FAIL] dump is capped at 30 entries
   // (g_nfail_list <= 30), and sweep_ldrstr alone produces 48 failures. Running
   // it first consumed the whole budget, so the rrx / carry-consumer failures
   // were reported in the summary with NO detail lines to diagnose them from.
   // Put the data-processing sweeps first while those are the ones under study.
   sweep_dataproc();
   sweep_dataproc_cond();
   sweep_cmp_sequences();   // multi-op within ONE block
   sweep_cmp_xblock();      // flags across a BLOCK BOUNDARY (the in-game pattern)
   sweep_branch();
   sweep_mul();
   sweep_ldmstm();
   sweep_ldrstr();

   cpu = saved;

   // --- per-family summary (only families with native JIT codegen are
   //     interesting; interpret-fallback families trivially pass) ---
   u32 tot_t = 0, tot_f = 0;
   DTPRINT("family        tested fail native\n");
   for (int f = 0; f < g_nfams; f++) {
      FamStat &s = g_fams[f];
      tot_t += s.tested; tot_f += s.failed;
      // Print only families that were JIT-native or had failures (skip noise).
      if (s.native || s.failed)
         DTPRINT("%-13s %5u %5u %5u%s\n", s.name, s.tested, s.failed, s.native,
                 s.failed ? "  <-- FAIL" : "");
   }
   DTPRINT("TOTAL tested=%u failed=%u\n", tot_t, tot_f);

   // --- distinct failing opcodes ---
   if (g_nfail_list) {
      DTPRINT("failing opcodes:\n");
      for (int i = 0; i < g_nfail_list; i++)
         DTPRINT("  %-10s %08X\n", g_fail_list[i].fam, g_fail_list[i].op);
      if (tot_f > (u32)g_nfail_list)
         DTPRINT("  ... (%u more)\n", tot_f - g_nfail_list);
   } else {
      DTPRINT("ALL PASS\n");
   }

   DTPRINT("(report also written to ROMS/difftest.txt)\n");
   g_pctrace_on = saved_pctrace;
   dt_log_close();
}
