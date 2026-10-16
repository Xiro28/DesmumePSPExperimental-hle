#include "blockdecoder.h"

// Single definition; declared extern in blockdecoder.h (see the note there).
uint32_t block_procnum = 0;

#include "mips_code_emiter.h"
#include "PSP/emit/psp_emit.h"
#include "armcpu.h"
#include "MMU.h"

#include "Disassembler.h"

#include <cstdarg>    // ldm_log diagnostic
#include <cstring>
#include <unistd.h>
#include <functional>
#include <cmath>

bool instr_does_prefetch(u32 opcode);

inline void loadReg(psp_gpr_t psp_reg, s32 nds_reg)
{
    if (nds_reg != -1)
        emit_lw(psp_reg, RCPU, _reg(nds_reg));
}
inline void storeReg(psp_gpr_t psp_reg, s32 nds_reg)
{
    if (nds_reg != -1)
        emit_sw(psp_reg, RCPU, _reg(nds_reg));
}

#include "psp_sra.h"

#define offsetBetween(x, x2) (((u32) & x) - ((u32) & x2[0]))
#define offsetBetween2(x, x2) (((u32) & x2[0]) - ((u32) & x))

u32 mem_off = 0;
u32 dtcm_addr_arr = 0;
int intr_instr = 0;

u32 start_block = 0;

block currentBlock;
register_manager regman;

// Spill every cached guest register to the armcpu_t struct and drop the mapping.
// Needed before handing control to a C handler that reads guest state directly
// (the NitroSDK microcode substitutions do: post_hle() takes the return address
// from NDS_ARM9.R[14]). regman lives here because psp_sra.h is only included in
// this translation unit.
void jit_flush_reg_cache()
{
    regman.flush_all();
    regman.reset();
}

bool flag_loaded = false;
bool use_flags = false;
bool islast_op = false;

bool flag_dirty = false;

// Diagnostic counter for the [LDM] decode log above. Set to 0 to silence.
int g_dumpLdmCode = 0;    // dump the first W=1 LDM/STM block's MIPS (real compile only:
                          // g_dumpCmpBlock is now 0, so no diagnostic pre-passes)
u32 g_ldm_wb_count = 0;   // see the LDM/STM writeback diagnostic below
int g_dumpLdmDecode = 0;  // >0 writes ROMS/ldm.txt; used to VERIFY that the
                           // JIT_ENABLE_LDM/STM switches actually took effect
                           // (an empty file proves nothing reached the codegen)

// printf() here only reaches stdout, and the difftest's DTPRINT is static to
// jit_difftest.cpp, so mirror the LDM decode log into its own file.
static void ldm_log(const char *fmt, ...)
{
    char path[256];
    if (!getcwd(path, sizeof(path) - 32)) return;
    strcat(path, "/ROMS/ldm.txt");
    FILE *f = fopen(path, "a");
    if (!f) return;
    va_list ap;
    va_start(ap, fmt);
    vfprintf(f, fmt, ap);
    va_end(ap);
    fclose(f);
}

void load_flags()
{
    if (use_flags /*&& !flag_loaded*/)
    {
        emit_lbu(psp_gp, RCPU, _flags + 3);
        flag_loaded = true;
    }
}

void store_flags()
{
    if (use_flags /*&& flag_loaded*/)
    {
        emit_sb(psp_gp, RCPU, _flags + 3);
        flag_loaded = false;
    }
}

// Load the ARM carry flag into `dst`, from wherever it currently lives.
//
// ADC/SBC/RSC used to read it straight out of CPSR in memory with an lbu. That is
// wrong whenever an earlier op in the SAME block already updated the flags: those
// updates go into psp_gp (see the emit_ins(psp_gp, ..., _flag_C8, _flag_C8) sites)
// and are only spilled to CPSR by store_flags() at the end of the block. So the
// lbu returned a STALE carry. Difftest caught it as failures confined to the
// carry-consuming S-variants -- SBCS imm failed 11 of 12 seeds -- while the
// non-S forms, which never read C, passed everywhere.
void load_carry(psp_gpr_t dst)
{
    if (use_flags && flag_loaded)
    {
        // Flags are live in psp_gp; CPSR in memory is behind.
        emit_ext(dst, psp_gp, _flag_C8, _flag_C8);
    }
    else
    {
        emit_lbu(dst, RCPU, _flags + 3);
        emit_ext(dst, dst, _flag_C8, _flag_C8);
    }
}

#define check_flags()       \
    if (flag_dirty)         \
    {                       \
        store_flags();      \
        flag_dirty = false; \
    }

#define force_flag_storage() \
    {                        \
        store_flags();       \
        flag_dirty = false;  \
    }

#define set_flag_dirty() flag_dirty = true;

const char *compiled_functions_hash =
    ">:1:05:8FDA19B2:C2AF4F28:998B99C6:F0596DB3:2FD2AA9B" // used by yoshi island ingame
    ">:1:07:88DD1AED:096B3BDA:7857790C:6D443C78:A4ADDB58" // a lot used in pokemon diamond
    ;

uint32_t conditional_label = 0;
uint32_t jump_sz = 0;

inline int constexpr getTypeIdx(opType op)
{
    switch (op)
    {
    case PRE_OP_LSL_IMM:
        return 0;
    case PRE_OP_LSR_IMM:
        return 1;
    case PRE_OP_ASR_IMM:
        return 2;
    case PRE_OP_ROR_IMM:
        return 3;
    case PRE_OP_LSL_REG:
        return 4;
    case PRE_OP_LSR_REG:
        return 5;
    case PRE_OP_ASR_REG:
        return 6;
    case PRE_OP_ROR_REG:
        return 7;
    default:
        printf("Is this alright? Return 0 as value to not make the emulator crash\n %d\n", op);
        return 0; // Default or error value
    }
}

// make an array of pointers to functions of type void
// and pass the opcode to the function
std::function<void(psp_gpr_t src, psp_gpr_t dst, opcode &op)> arm_preop[] = {
    [](psp_gpr_t src, psp_gpr_t dst, opcode &op) -> void { // PRE_OP_LSL_IMM
        // printf("PRE_OP_LSL_IMM %d \n", op.imm);
        emit_sll(dst, src, op.imm);
    },

    [](psp_gpr_t src, psp_gpr_t dst, opcode &op) -> void { // PRE_OP_LSR_IMM
        // printf("PRE_OP_LSR_IMM %d \n", op.imm);
        if (op.imm)
            emit_srl(dst, src, op.imm);
        else
            emit_move(dst, psp_zero);
    },

    [](psp_gpr_t src, psp_gpr_t dst, opcode &op) -> void { // PRE_OP_ASR_IMM
        emit_sra(dst, src, op.imm ? op.imm : 31);
    },

    [](psp_gpr_t src, psp_gpr_t dst, opcode &op) -> void { // PRE_OP_ROR_IMM
        if (op.imm == 0)
        {
            // RRX: dst = (C << 31) | (src >> 1).
            emit_srl(dst, src, 1);
            // Read C from wherever it currently lives. Reading it straight out of
            // CPSR was wrong for the S-variants: those run under
            // conditional_branchless_flags with the flags live in psp_gp, and CPSR
            // is only refreshed by store_flags() at end of block -- so RRX used a
            // STALE carry and got bit 31 of the operand wrong. Difftest showed the
            // non-S forms passing while every *S* rrx failed with the result
            // differing in bit 31 alone.
            load_carry(psp_t1);
            // emit_ins takes (msb, lsb), NOT (pos, size): the macro forwards _msb
            // straight through, unlike emit_ext which computes _msb-_lsb itself.
            // Passing (31, 1) therefore inserted 31 bits starting at bit 1 and
            // wrecked the whole register -- RRX failed in EVERY ALU family
            // (jit produced Rn op 1: the operand had been reduced to 1). A
            // single-bit insert needs msb == lsb, as the C-flag sites do.
            emit_ins(dst, psp_t1, 31, 31);             // insert 1 bit AT bit 31
            return;
        }

        emit_rotr(dst, src, op.imm);
    },
    [](psp_gpr_t src, psp_gpr_t dst, opcode &op) -> void { // PRE_OP_LSL_REG
        // ARM: amount = Rs & 0xFF; if amount >= 32 result is 0 (MIPS sllv only
        // uses low 5 bits, so we must zero the result for amount >= 32).
        // NOTE: dst is often psp_at, so amount/compare must use OTHER scratch.
        int32_t regs[1] = {op.imm};
        regman.get(1, regs);
        emit_andi(psp_v0, (psp_gpr_t)regs[0], 0xFF);   // v0 = amt8
        emit_sllv(dst, src, psp_v0);                   // src << (amt8 & 31)
        emit_sltiu(psp_v1, psp_v0, 32);                // v1 = (amt8 < 32)
        emit_movz(dst, psp_zero, psp_v1);              // amt8 >= 32 -> 0
    },

    [](psp_gpr_t src, psp_gpr_t dst, opcode &op) -> void { // PRE_OP_LSR_REG
        // ARM: amount = Rs & 0xFF; if amount >= 32 result is 0.
        int32_t regs[1] = {op.imm};
        regman.get(1, regs);
        emit_andi(psp_v0, (psp_gpr_t)regs[0], 0xFF);
        emit_srlv(dst, src, psp_v0);
        emit_sltiu(psp_v1, psp_v0, 32);
        emit_movz(dst, psp_zero, psp_v1);              // amt8 >= 32 -> 0
    },

    [](psp_gpr_t src, psp_gpr_t dst, opcode &op) -> void { // PRE_OP_ASR_REG
        // ARM: amount = Rs & 0xFF; if amount >= 32 result is sign-extension
        // (src >> 31, arithmetic). MIPS srav masks to 5 bits, so clamp.
        int32_t regs[1] = {op.imm};
        regman.get(1, regs);
        emit_andi(psp_v0, (psp_gpr_t)regs[0], 0xFF);   // v0 = amt8
        emit_srav(dst, src, psp_v0);                   // src >>a (amt8 & 31)
        emit_sra(psp_v1, src, 31);                     // sign fill
        emit_sltiu(psp_v0, psp_v0, 32);                // v0 = (amt8 < 32)
        emit_movz(dst, psp_v1, psp_v0);                // amt8 >= 32 -> sign fill
    },

    [](psp_gpr_t src, psp_gpr_t dst, opcode &op) -> void { // PRE_OP_ROR_REG
        // ARM: amount = Rs & 0xFF; rotate uses amount & 31. MIPS rotrv masks to
        // 5 bits, which matches ARM exactly (ROR by a multiple of 32 = no-op).
        int32_t regs[1] = {op.imm};
        regman.get(1, regs);
        emit_rotrv(dst, src, (psp_gpr_t)regs[0]);
    },
};

#define cpu (&ARMPROC)

template <int PROCNUM>
static u16 FASTCALL _LDRH(u32 adr)
{
    return READ16(cpu->mem_if->data, adr);
}

template <int PROCNUM>
static void FASTCALL _STRH(u32 regs, u32 imm)
{
    u32 adr = (u32)cpu->R[regs >> 8] + imm;
    u32 data = (u32)cpu->R[regs & 0xFF];
    WRITE16(cpu->mem_if->data, adr, data);
}

void emit_li(u32 reg, u32 data, u32 sz)
{
    if (data == 0 && sz == 0) {           // FAST PATH
        emit_move(reg, psp_zero);
        return;
    }
    if (is_u16(data) && sz != 2)
        emit_ori(reg, psp_zero, data & 0xFFFF);
    else if (is_s16(data) && sz != 2)
        emit_movi(reg, data & 0xFFFF);
    else {
        emit_lui(reg, data >> 16);
        if ((sz == 2) || (data & 0xFFFF))
            emit_ori(reg, reg, data & 0xFFFF);
    }
}

// load upper address
inline u32 emit_lua(u32 reg, u32 data)
{
    u32 hi = data >> 16;
    u32 lo = data & 0xFFFF;

    if (lo & 0x8000)
    {
        hi++;
        lo = data - (hi << 16);
    }

    emit_lui(reg, u16(hi));
    return lo;
}

void generate_condition_check(int cond)
{
    static const uint8 cond_bit[] = {0x40, 0x40, 0x20, 0x20, 0x80, 0x80, 0x10, 0x10};

    if (cond < 8)
    {
        emit_andi(psp_s4, psp_gp, cond_bit[cond]);
        return;
    }

    switch (cond)
    {

    case 8:
    case 9:
        emit_ext(psp_a1, psp_gp, 6, 5);
        emit_xori(psp_s4, psp_a1, 0b01);
        break;

    case 10:
    case 11:

        emit_ext(psp_a1, psp_gp, 7, 7);
        emit_ext(psp_at, psp_gp, 4, 4);

        emit_xor(psp_s4, psp_a1, psp_at);
        break;

    case 12:
    case 13:

        emit_ext(psp_a1, psp_gp, 7, 6);
        emit_ext(psp_at, psp_gp, 4, 3);

        emit_andi(psp_at, psp_at, 0b10);
        emit_xor(psp_s4, psp_a1, psp_at);
        break;
    }
}

uint32 emit_Halfbranch(int cond, bool generate_condition = true)
{
    if (generate_condition)
        generate_condition_check(cond);

    emit_nop();
    emit_nop();
    return emit_getPointAdr() - 8;
}

void CompleteCondition(u32 cond, u32 _addr, u32 label)
{
    if (cond < 8)
    {
        if (cond & 1)
            emit_bnelC(psp_s4, psp_zero, label, _addr);
        else
            emit_beqlC(psp_s4, psp_zero, label, _addr);
        return;
    }

    if (cond & 1)
        emit_beqlC(psp_s4, psp_zero, label, _addr);
    else
        emit_bnelC(psp_s4, psp_zero, label, _addr);
}
bool saved = false;

static int prefetch_skip = 1;
void emit_prefetch_reset() { prefetch_skip = 1; }

void emit_prefetch(const u8 isize, bool updateR15, bool store)
{

    if (updateR15 || store)
    {

        emit_addiu(psp_fp, psp_fp, isize * prefetch_skip);

        if (store)
            emit_sw(psp_fp, RCPU, _next_instr);

        saved = false;

        emit_addiu(psp_at, psp_fp, isize);
        if (store)
            emit_sw(psp_at, RCPU, _R15);

        emit_prefetch_reset();
    }
    else
        prefetch_skip++;
}

INLINE void emit_bic(u32 dst, u32 a0, u32 a1)
{
    emit_not(a1, a1);
    emit_and(dst, a0, a1);
}

INLINE void emit_bici(u32 dst, u32 a0, u32 a1)
{
    // op.imm arrives ALREADY complemented: ARM_OP_BIC_IMM_VAL rewrites
    // BIC Rd,Rn,#imm into OP_AND with ~imm. Complementing again here produced
    // andi(dst, a0, ~~imm) whose low 16 bits could be 0 (e.g. BIC #0xFFFF0000
    // -> op.imm=0x0000FFFF -> andi #0x0000 -> register zeroed instead of masked).
    emit_andi(dst, a0, a1);
}

#define END_OP(_rd)                               \
    {                                             \
        if (!op.dead_rd)                          \
            regman.mark_dirty((psp_gpr_t)_rd);    \
        if (!rd_allocated && !op.dead_rd)         \
            regman.flush((psp_gpr_t)_rd);         \
    }

#define END_OP_CHKR15(_rd)                                 \
    {                                                      \
        if (!op.dead_rd)                                   \
            regman.mark_dirty((psp_gpr_t)_rd);             \
        if (op.rd == 15)                                   \
        {                                                  \
            op.check_condition = false;                    \
            conditional(emit_sw(_rd, RCPU, _next_instr);); \
        }                                                  \
    }

#define HANDLE_CONDITIONAL(branchless_code, branch_code) \
    do                                                   \
    {                                                    \
        if (op.rd != 15)                                 \
        {                                                \
            conditional_branchless(regs[0], psp_at,      \
                                   {branchless_code});   \
            END_OP_CHKR15(regs[0]);                      \
        }                                                \
        else                                             \
        {                                                \
            conditional(                                 \
                branch_code                              \
                    END_OP_CHKR15(regs[0]););            \
        }                                                \
    } while (0)

#define HANDLE_CONDITIONAL_NR15(branchless_code, branch_code) \
    do                                                        \
    {                                                         \
        if ((rd_allocated) && (op).condition != -1)           \
        {                                                     \
            conditional_branchless(regs[0], psp_at,           \
                                   {branchless_code});        \
            END_OP(regs[0]);                                  \
        }                                                     \
        else                                                  \
        {                                                     \
            conditional(                                      \
                branch_code                                   \
                    END_OP(regs[0]););                        \
        }                                                     \
    } while (0)

// emit_subiu encodes -imm, not imm, so the 16-bit range check must be applied
// to the value that actually lands in the instruction. Gating on imm let
// op.imm==0xFFFF8000 (-32768) through: addiu then received +0x8000, which
// sign-extends back to -32768 -- off by 65536.
// is_sub16: subiu's immediate field receives -imm, so range-check that.
static INLINE bool is_sub16(u32 v) { return is_s16(0u - v); }
#define gen_nativeOP(opType, n_op, n_op_imm, sign)                                                          \
    template <bool imm, bool rev>                                                                           \
    void arm_##opType(opcode &op)                                                                           \
    {                                                                                                       \
        int32_t regs[3] = {(op.condition != -1) ? op.rd : (op.rd | 0x10), op.rs1, op.rs2};                  \
        const int reg_count = imm ? 2 : 3;                                                                  \
        regman.get(reg_count, regs);                                                                        \
        const psp_gpr_t dst = (op.condition != -1) ? psp_at : (psp_gpr_t)regs[0];                           \
        if (imm)                                                                                            \
        {                                                                                                   \
            if (!rev && is_##sign##16(op.imm))                                              \
            {                                                                                               \
                conditional_branchless(regs[0], dst,                                                        \
                                       {                                                                    \
                                           emit_##n_op_imm(dst, regs[1], op.imm);                           \
                                       });                                                                  \
            }                                                                                               \
            else                                                                                            \
            {                                                                                               \
                conditional_branchless(regs[0], dst,                                                        \
                                       {                                                                    \
                                           emit_li(psp_at, op.imm);                                         \
                                           if (!rev)                                                        \
                                               emit_##n_op(dst, regs[1], psp_at);                           \
                                           else                                                             \
                                               emit_##n_op(dst, psp_at, regs[1]);                           \
                                       });                                                                  \
            }                                                                                               \
        }                                                                                                   \
        else                                                                                                \
        {                                                                                                   \
            conditional_branchless(regs[0], dst,                                                            \
                                   {                                                                       \
                                       arm_preop[getTypeIdx(op.preOpType)]((psp_gpr_t)regs[2], psp_at, op); \
                                       if (!rev)                                                            \
                                           emit_##n_op(dst, regs[1], psp_at);                               \
                                       else                                                                 \
                                           emit_##n_op(dst, psp_at, regs[1]);                               \
                                   });                                                                      \
        }                                                                                                   \
        END_OP_CHKR15(regs[0]);                                                                             \
    }

gen_nativeOP(and, and, andi, u);
gen_nativeOP(or, or, ori, u);
gen_nativeOP(xor, xor, xori, u);
gen_nativeOP(add, addu, addiu, s);
gen_nativeOP(sub, subu, subiu, sub);
gen_nativeOP(bic, bic, bici, u);

// Set by constant propagation on a load/store whose address is a compile-time
// constant in ARM9 main RAM; op.imm then holds that address.
static const uint32_t EXTFL_CONSTADDR = 256;
// Set on a MUL/MLA whose Rs is a compile-time constant: its variable cycle cost
// was already added to the block's static count, so don't compute it at run time.
static const uint32_t EXTFL_MULCYC = 512;
extern uint32_t interpreted_cycles;

// Apply an IMM-shift to a compile-time constant.
// Returns false for RRX (shift=0 ROR), which needs the carry flag at runtime.
static bool apply_const_shift(opType preOp, uint32_t src, uint32_t shift, uint32_t &out)
{
    switch (preOp) {
    case PRE_OP_LSL_IMM:
        out = src << shift;
        return true;
    case PRE_OP_LSR_IMM:
        out = shift ? (src >> shift) : 0u;
        return true;
    case PRE_OP_ASR_IMM:
        out = (uint32_t)((int32_t)src >> (shift ? shift : 31));
        return true;
    case PRE_OP_ROR_IMM:
        if (!shift) return false; // RRX — depends on carry
        out = (src >> shift) | (src << (32u - shift));
        return true;
    default:
        return false;
    }
}

// Evaluate a binary IR opcode with two known constants.
// lhs = consts[rs1], rhs = op.imm.
// Returns false for ops whose result can't be folded (flag-setters, mem, etc.).
static bool eval_const_binop(op _op, uint32_t lhs, uint32_t rhs, uint32_t &out)
{
    switch (_op) {
    case OP_ADD: out = lhs + rhs;  return true;
    case OP_SUB: out = lhs - rhs;  return true;
    case OP_RSB: out = rhs - lhs;  return true; // RSB: rd = imm - rs1
    case OP_AND: out = lhs & rhs;  return true;
    case OP_ORR: out = lhs | rhs;  return true;
    case OP_EOR: out = lhs ^ rhs;  return true;
    case OP_BIC: out = lhs & ~rhs; return true;
    default:     return false;
    }
}

// Do you want speed? call this function if you want to see some black magic :D

void block::optimize_basicblock()
{
    // === Pass 1: Condition merging ===
    // Consecutive ops with the same condition can skip re-evaluating the flags.
    if (jit_opt_condmerge)
    {
        opcode *prev = nullptr;
        for (opcode &op : opcodes) {
            // Reuse the previous op's condition predicate ($s4) ONLY when prev neither
            // is interpreted NOR rewrites NZCV. generate_condition_check derives $s4
            // from psp_gp BEFORE the op body runs, so if prev updates the flags (CMP,
            // CMN, TST, TEQ, and every *_S op — all have _op >= OP_CMP) the cached $s4
            // is stale and a following same-condition op would test the OLD flags.
            // Previously the guard only excluded OP_ITP; when CMP was interpreted that
            // happened to cover it, but JITting CMP (OP_CMP) exposed the stale-predicate
            // bug (black screen). Plain ALU ops (OP_ITP < _op < OP_CMP) don't touch
            // flags, so merging across them is still safe. See perf-profile-findings.
            if (prev && prev->_op != OP_ITP && prev->_op < OP_CMP
                && op.condition == prev->condition)
                op.check_condition = false;
            prev = &op;
        }
    }

    // === Pass 2: Constant propagation ===
    // Forward pass: track registers whose values are compile-time constants and
    // fold arithmetic/shift expressions into immediate MOVs where possible.
    // Runtime toggle (jit_opt_constprop). Steps A/B are guarded to ALU ops only —
    // previously they rewrote memory ops, dropping GX-FIFO stores (3D vanished).
    if (jit_opt_constprop)
    {
        struct ConstVal { bool known; uint32_t val; };
        ConstVal consts[16] = {};

        for (opcode &op : opcodes)
        {
            // ITP is an interpreted instruction — it can read/write any register.
            if (op._op == OP_ITP) {
                memset(consts, 0, sizeof(consts));
                continue;
            }

            // LDM defines every register in its list; STM only its writeback base.
            // Invalidate all written regs so later folds don't use stale constants.
            if (op._op == OP_LDM || op._op == OP_STM) {
                if (op._op == OP_LDM) {
                    const uint16_t rlist = op.bytes & 0xFFFF;
                    for (int r = 0; r < 16; r++)
                        if (rlist & (1u << r)) consts[r].known = false;
                }
                if (op.rs1 >= 0 && op.rs1 < 16) consts[op.rs1].known = false; // writeback base
                continue;
            }

            const bool unconditional = (op.condition == (uint32_t)-1);

            // Folding (Steps A/B) is only valid for pure ALU ops. For memory ops
            // (OP_LDR/OP_STR/OP_LDM/...) preOpType encodes the ADDRESSING MODE and
            // op.imm is the offset, NOT an ALU operand — rewriting them corrupts
            // the access (and Step B would turn a store into a MOV, dropping it:
            // that silently kills GX-FIFO / 3D stores). ALU opcodes are the range
            // (OP_ITP, OP_LDR); compare/etc. (>= OP_CMP) also excluded.
            const bool is_alu_op = (op._op > OP_ITP && op._op < OP_LDR);

            // Step M: MUL/MLA with a known Rs (rs2). The ARM9 cost depends only on
            // Rs, so charge it statically (same count as emit_mul_cycles, also
            // charged when the condition fails). Rs == 0 or 2^k becomes a MOV/shift
            // (MUL) or MOV/ADD with shift (MLA; accumulator Rn lives in op.imm).
            if ((op._op == OP_MUL || op._op == OP_MLA) && op.rs2 < 15 && consts[op.rs2].known)
            {
                const uint32_t v = consts[op.rs2].val;
                uint32_t c = 4;
                for (int k = 1; k <= 3; k++)
                    if ((int32_t)v >> (8 * k) == (int32_t)v >> 31)
                        c--;
                interpreted_cycles += c;

                const uint32_t rm = op.rs1;
                if ((v & (v - 1)) == 0 && op.rd != 15)
                {
                    const bool mla = (op._op == OP_MLA);
                    const uint32_t rn = (uint32_t)op.imm;
                    if (v == 0 && !mla) {
                        op._op = OP_MOV; op.rs1 = -1; op.rs2 = -1;
                        op.imm = 0;      op.preOpType = PRE_OP_IMM;
                    } else if (v == 0) {
                        op._op = OP_MOV; op.rs1 = -1; op.rs2 = rn;
                        op.imm = 0;      op.preOpType = PRE_OP_LSL_IMM;
                    } else {
                        op._op = mla ? OP_ADD : OP_MOV;
                        op.rs1 = mla ? rn : (uint32_t)-1;
                        op.rs2 = rm;
                        op.imm = __builtin_ctz(v);
                        op.preOpType = PRE_OP_LSL_IMM;
                    }
                }
                else
                    op.extra_flags |= EXTFL_MULCYC;
            }

            // Step R: shift-by-register with a known amount (Rs in op.imm).
            // Rm known too -> the whole operand is a constant. Otherwise rewrite to
            // an immediate shift, only for amounts whose immediate encoding means
            // the same thing (0 -> LSL #0, 1..31). Non-S ALU ops ignore the
            // shifter carry, so this is exact.
            if (is_alu_op &&
                (op.preOpType == PRE_OP_LSL_REG || op.preOpType == PRE_OP_LSR_REG ||
                 op.preOpType == PRE_OP_ASR_REG || op.preOpType == PRE_OP_ROR_REG) &&
                op.imm >= 0 && op.imm < 15 && consts[op.imm].known)
            {
                const uint32_t n = consts[op.imm].val & 0xFF;
                if (op.rs2 < 15 && consts[op.rs2].known)
                {
                    const uint32_t v = consts[op.rs2].val;
                    uint32_t r;
                    switch (op.preOpType)
                    {
                    case PRE_OP_LSL_REG: r = n >= 32 ? 0 : v << n; break;
                    case PRE_OP_LSR_REG: r = n >= 32 ? 0 : v >> n; break;
                    case PRE_OP_ASR_REG: r = (uint32_t)((int32_t)v >> (n >= 32 ? 31 : n)); break;
                    default:             r = (n & 31) ? ((v >> (n & 31)) | (v << (32 - (n & 31)))) : v; break;
                    }
                    if (op._op == OP_MVN) {
                        op._op = OP_MOV;
                        r = ~r;
                    }
                    op.preOpType = PRE_OP_IMM;
                    op.imm       = (int32_t)r;
                    op.rs2       = -1;
                }
                else
                {
                    const uint32_t amt = (op.preOpType == PRE_OP_ROR_REG) ? (n & 31) : n;
                    if (amt == 0) {
                        op.preOpType = PRE_OP_LSL_IMM;
                        op.imm       = 0;
                    } else if (amt < 32) {
                        op.preOpType = (op.preOpType == PRE_OP_LSL_REG) ? PRE_OP_LSL_IMM :
                                       (op.preOpType == PRE_OP_LSR_REG) ? PRE_OP_LSR_IMM :
                                       (op.preOpType == PRE_OP_ASR_REG) ? PRE_OP_ASR_IMM : PRE_OP_ROR_IMM;
                        op.imm       = (int32_t)amt;
                    }
                }
            }

            // Step A: rs2 is const + IMM shift → fold the shift into an immediate.
            // e.g. "ADD rd, rn, rm LSL #n" → "ADD rd, rn, #(rm_val << n)"
            if (is_alu_op &&
                (op.preOpType == PRE_OP_LSL_IMM || op.preOpType == PRE_OP_LSR_IMM ||
                 op.preOpType == PRE_OP_ASR_IMM || op.preOpType == PRE_OP_ROR_IMM) &&
                op.rs2 >= 0 && op.rs2 < 16 && consts[op.rs2].known)
            {
                uint32_t shifted;
                if (apply_const_shift(op.preOpType, consts[op.rs2].val, (uint32_t)op.imm, shifted)) {
                    // The PRE_OP_IMM emitter treats MVN as a plain li, so invert here.
                    if (op._op == OP_MVN) {
                        op._op  = OP_MOV;
                        shifted = ~shifted;
                    }
                    op.preOpType = PRE_OP_IMM;
                    op.imm       = (int32_t)shifted;
                    op.rs2       = -1;
                }
            }

            // Step E: rs1 const, rs2 a plain register -> swap so the constant
            // becomes the immediate: "ADD rd, rK, rm" -> "ADD rd, rm, #K"
            // (SUB <-> RSB). Saves loading rK into a host register.
            if (is_alu_op &&
                op.preOpType == PRE_OP_LSL_IMM && op.imm == 0 &&
                op.rs1 < 15 && consts[op.rs1].known &&
                op.rs2 < 15 && !consts[op.rs2].known &&
                (op._op == OP_ADD || op._op == OP_AND || op._op == OP_ORR ||
                 op._op == OP_EOR || op._op == OP_SUB || op._op == OP_RSB))
            {
                if (op._op == OP_SUB)      op._op = OP_RSB;
                else if (op._op == OP_RSB) op._op = OP_SUB;
                op.imm       = (int32_t)consts[op.rs1].val;
                op.rs1       = op.rs2;
                op.rs2       = -1;
                op.preOpType = PRE_OP_IMM;
            }

            // Step B: rs1 is const + PRE_OP_IMM → fold to a pure MOV #imm.
            // e.g. "ADD rd, rn, #k" where rn==const → "MOV rd, #(rn_val + k)"
            // Conditional ops become a conditional MOV (never for rd==PC).
            if (is_alu_op &&
                (unconditional || op.rd != 15) &&
                op.preOpType == PRE_OP_IMM &&
                op._op != OP_MOV &&
                op.rs1 >= 0 && op.rs1 < 16 && consts[op.rs1].known)
            {
                uint32_t folded;
                if (eval_const_binop(op._op, consts[op.rs1].val, (uint32_t)op.imm, folded)) {
                    op._op  = OP_MOV;
                    op.imm  = (int32_t)folded;
                    op.rs1  = -1;
                    // preOpType stays PRE_OP_IMM — the emitter emits "li rd, folded"
                }
            }

            // Single loads/stores. IR convention: loads rd=dest rs1=base, stores
            // rd=base rs1=value. Only the plain offset form (PRE_OP_IMM) is free of
            // writeback; every other addressing form may update the base.
            const bool is_load  = (op._op == OP_LDR || op._op == OP_LDRH || op._op == OP_LDRSH || op._op == OP_LDRSB);
            const bool is_store = (op._op == OP_STR || op._op == OP_STRH);
            if (is_load || is_store)
            {
                const uint32_t base = is_store ? op.rd : op.rs1;
                const uint32_t data = is_store ? op.rs1 : op.rd;
                const bool plain = (op.preOpType == PRE_OP_IMM);

                // Step D: known base -> constant address; plain ARM9 main RAM
                // outside DTCM is then accessed directly (emit_const_mem).
                if (plain && unconditional && block_procnum == ARMCPU_ARM9 &&
                    base < 15 && data < 15 && consts[base].known)
                {
                    const uint32_t addr = consts[base].val + (uint32_t)op.imm;
                    const uint32_t size = (op._op == OP_LDR || op._op == OP_STR) ? 4 :
                                          (op._op == OP_LDRSB) ? 1 : 2;
                    if ((addr >> 24) == 0x02 && (addr & (size - 1)) == 0 &&
                        (addr & ~0x3FFFu) != MMU.DTCMRegion)
                    {
                        op.extra_flags |= EXTFL_CONSTADDR;
                        op.imm = (int32_t)addr;
                    }
                }

                if (!plain && base < 16) consts[base].known = false;
                if (is_load && op.rd < 16) consts[op.rd].known = false;
                continue;
            }

            // Step C: update const table for the destination register.
            // Other non-ALU ops may write rs1 too; UMULL/UMLAL write RdLo (imm).
            if (!is_alu_op && op.rs1 >= 0 && op.rs1 < 16) consts[op.rs1].known = false;
            if ((op._op == OP_UMUL || op._op == OP_UMLA) && op.imm >= 0 && op.imm < 16)
                consts[op.imm].known = false;
            if (op.rd >= 0 && op.rd < 15) { // never track r15 (PC)
                consts[op.rd].known = false;  // pessimistically invalidate first

                // Only a plain unconditional MOV #imm produces a known compile-time value.
                if (unconditional && op._op == OP_MOV && op.preOpType == PRE_OP_IMM) {
                    consts[op.rd] = {true, (uint32_t)op.imm};
                }
            }
        }
    }

    // === Pass 3: Dead store elimination ===
    // Backward liveness pass: if rd is overwritten before any subsequent read,
    // the writeback to RCPU is dead — suppress mark_dirty so flush emits no sw.
    // Conservative: all regs are live at block exit (next block may read anything).
    // We can only catch intra-block kills (reg written twice without a read between).
    {
        uint16_t live = 0xFFFF; // bits 0-15: live ARM registers

        for (int i = (int)opcodes.size() - 1; i >= 0; --i) {
            opcode &op = opcodes[i];

            if (op._op == OP_ITP) {
                live = 0xFFFF; // interpreted instr: conservatively uses everything
                continue;
            }

            if (op._op == OP_LDM || op._op == OP_STM) {
                const uint16_t rlist = op.bytes & 0xFFFF;
                if (op._op == OP_LDM) live &= ~rlist;   // loaded regs are defined
                else                  live |= rlist;     // stored regs are used
                if (op.rs1 >= 0 && op.rs1 < 16) live |= (1u << op.rs1); // base
                continue;
            }

            const bool unconditional = (op.condition == (uint32_t)-1);

            // Only pure ALU ops (no memory side-effects) can be dead-stored.
            const bool is_alu = (op._op > OP_ITP && op._op < OP_LDR);

            if (is_alu && unconditional && op.rd < 15 && op.rd >= 0) {
                if (!(live & (1u << op.rd))) {
                    op.dead_rd = true; // nobody reads this value before it's overwritten
                }
                // Definition kills liveness regardless of whether it's dead.
                live &= ~(1u << op.rd);
            }

            // Mark sources as live. For memory ops rd can be a read (a store's base).
            if (!is_alu && op.rd >= 0 && op.rd < 16) live |= (1u << op.rd);
            if (op._op == OP_SMLAXY) live |= (1u << (op.imm & 0xF));
            if (op.rs1 >= 0 && op.rs1 < 16) live |= (1u << op.rs1);
            if (op.rs2 >= 0 && op.rs2 < 16) live |= (1u << op.rs2);
            // REG-shift and MLA: shift-amount / accumulator register lives in imm field.
            if (op.preOpType == PRE_OP_LSL_REG || op.preOpType == PRE_OP_LSR_REG ||
                op.preOpType == PRE_OP_ASR_REG || op.preOpType == PRE_OP_ROR_REG ||
                op._op == OP_MLA || op._op == OP_UMLA) {
                if (op.imm >= 0 && op.imm < 16) live |= (1u << op.imm);
            }
        }
    }

    return;
}

void block::optimize_basicblockThumb()
{
    // EXTFL_NOFLAGS ops (PC/SP-relative ADD, folded literal LDR) must leave NZCV
    // alone, but only the OP_ADD emitter checks the flag -- OP_MOV would still
    // write N/Z and break a following conditional branch. Always applied, it is
    // a correctness rule, not an optimization.
    for (opcode &op : opcodes)
        if (op.extra_flags & EXTFL_NOFLAGS)
            op.saveN = op.saveZ = op.saveC = op.saveV = false;

    if (!jit_opt_thumbflags)
        return;

    opcode *prev_op = 0;
    opcode *prev_ITP = 0;

    bool savedN = false;
    bool savedZ = false;
    bool savedC = false;
    bool savedV = false;

    for (int i = opcodes.size() - 1; i >= 0; --i)
    {
        opcode &op = opcodes[i];

        if (op.extra_flags & EXTFL_NOFLAGS)
            continue;

        switch (op._op)
        {
        case OP_STR:
        case OP_STRH:
        case OP_LDR:
        case OP_LDRH:
        case OP_ITP:
            savedN = false;
            savedZ = false;
            savedC = false;
            savedV = false;
            break;

        case OP_AND:
        case OP_ADD:
        case OP_EOR:
        case OP_ORR:
        case OP_TST:
        case OP_MUL:
        case OP_MOV:
        case OP_MVN:

            if (!savedC)
            {
                savedC = true;
            }
            else
                op.saveC = false;

            if (!savedN)
            {
                savedN = true;
            }
            else
                op.saveN = false;

            if (!savedZ)
            {
                savedZ = true;
            }
            else
                op.saveZ = false;

            break;

        case OP_SUB:
        case OP_CMP:
            if (!savedN)
            {
                savedN = true;
            }
            else
                op.saveN = false;

            if (!savedZ)
            {
                savedZ = true;
            }
            else
                op.saveZ = false;

            if (!savedC)
            {
                savedC = true;
            }
            else
                op.saveC = false;

            if (!savedV)
            {
                savedV = true;
            }
            else
                op.saveV = false;

            break;

        default:
            break;
        }
    }
    /*if (noReadWriteOP && branch_addr == start_addr){
        for(opcode& op : opcodes){
            if (op._op == OP_SWI){

                idleLoop = true;
                break;
            }
        }
    } */

    // Dead store elimination (same pass as ARM mode).
    {
        uint16_t live = 0xFFFF;

        for (int i = (int)opcodes.size() - 1; i >= 0; --i) {
            opcode &op = opcodes[i];

            if (op._op == OP_ITP) { live = 0xFFFF; continue; }

            if (op._op == OP_LDM || op._op == OP_STM) {
                const uint16_t rlist = op.bytes & 0xFFFF;
                if (op._op == OP_LDM) live &= ~rlist;   // loaded regs are defined
                else                  live |= rlist;     // stored regs are used
                if (op.rs1 >= 0 && op.rs1 < 16) live |= (1u << op.rs1); // base
                continue;
            }

            const bool unconditional = (op.condition == (uint32_t)-1);
            const bool is_alu = (op._op > OP_ITP && op._op < OP_LDR);

            if (is_alu && unconditional && op.rd < 15 && op.rd >= 0) {
                if (!(live & (1u << op.rd)))
                    op.dead_rd = true;
                live &= ~(1u << op.rd);
            }

            if (op.rs1 >= 0 && op.rs1 < 16) live |= (1u << op.rs1);
            if (op.rs2 >= 0 && op.rs2 < 16) live |= (1u << op.rs2);
            if (op.preOpType == PRE_OP_LSL_REG || op.preOpType == PRE_OP_LSR_REG ||
                op.preOpType == PRE_OP_ASR_REG || op.preOpType == PRE_OP_ROR_REG ||
                op._op == OP_MLA || op._op == OP_UMLA) {
                if (op.imm >= 0 && op.imm < 16) live |= (1u << op.imm);
            }
        }
    }
}

extern "C" void set_sub_flags();
extern "C" void set_add_flags();
extern "C" void set_and_flags();
extern "C" void set_op_logic_flags();

#define cpu (&ARMPROC)

void lastBeforeCrash(int a0, int a1)
{
    printf("0x%x\n", a0);
}

void EmitReadFunction(u32 addr)
{
    unsigned o_ra;
    extern u8 *CodeCache;

    // Execute the patch at the end (overwrite ra addr inside the sp)
    asm volatile("addiu $2, $31, -16");
    asm volatile("sw $2, 0x14($29)");

    // Get the current ra
    asm volatile("sw $31, %0" : "=m"(o_ra));

    u32 _ptr = emit_Set((o_ra - 16) - 0x8400000);

    printf("EmitReadFunction: 0x%x 0x%x\n", o_ra, 0x8400000);

    if ((addr & (~0x3FFF)) == MMU.DTCMRegion)
    {
        emit_li(psp_t1, (u32)MMU.ARM9_DTCM);
        emit_andi(psp_a0, psp_a0, 0x3FFC);
    }
    else
    {
        emit_li(psp_t1, (u32)MMU.MAIN_MEM);
        emit_ins(psp_a0, psp_zero, 31, 22);
    }

    emit_addu(psp_a0, psp_t1, psp_a0);
    emit_lw(psp_v0, psp_a0, 0);

    emit_Set(_ptr);

    // make_address_range_executable(o_ra - 8, o_ra - 4);
    u32 addr_start = o_ra - 16;
    __builtin_allegrex_cache(0x1a, addr_start);
    __builtin_allegrex_cache(0x08, addr_start);
}

#define NDS_ReschedulePtr (*(bool *)(0x0010080))
// Non-recursive: set the target PC into BOTH next_instruction and instruct_adr
// (continue_cpu_check chains on instruct_adr), then return. Do NOT call code_block()
// — recursion grows the native stack one frame per taken branch (loops/calls
// overflow it and corrupt memory). The block epilogue + continue_cpu_check
// tail-jump to the target with no stack growth.
// These update the CPU that the block belongs to. They used to hardcode
// NDS_ARM9, which was correct only while the JIT was ARM9-only; an ARM7 block
// would have driven the ARM9's PC instead of its own.
template <bool execute, int PROCNUM>
u32 jump_to_linked_bc(u32 off)
{
    armcpu_t &proc = ARMPROC;
    proc.R[15] += off;
    proc.R[15] &= (0xFFFFFFFC | (proc.CPSR.bits.T << 1));
    proc.next_instruction = proc.R[15];
    proc.instruct_adr     = proc.R[15];
    return 3;
}

template <bool execute, int PROCNUM>
u32 jump_to_linked_blc(u32 off)
{
    armcpu_t &proc = ARMPROC;
    proc.R[14] = proc.next_instruction;   // link register = return address
    proc.R[15] += off;
    proc.R[15] &= (0xFFFFFFFC | (proc.CPSR.bits.T << 1));
    proc.next_instruction = proc.R[15];
    proc.instruct_adr     = proc.R[15];
    return 3;
}

template <bool execute, int PROCNUM>
u32 jump_to_linked(u32 addr)
{
    armcpu_t &proc = ARMPROC;
    proc.CPSR.bits.T = BIT0(addr);
    proc.R[15] = addr & (0xFFFFFFFC | (proc.CPSR.bits.T << 1));
    proc.next_instruction = proc.R[15];
    proc.instruct_adr     = proc.R[15];
    return 3;
}

template <bool execute, int PROCNUM>
u32 jump_to_linked_br(u32 addr)
{
    armcpu_t &proc = ARMPROC;
    proc.R[14] = proc.next_instruction;
    proc.CPSR.bits.T = BIT0(addr);
    proc.R[15] = addr & (0xFFFFFFFC | (proc.CPSR.bits.T << 1));
    proc.next_instruction = proc.R[15];
    proc.instruct_adr     = proc.R[15];
    return 3;
}
template <int PROCNUM>
u32 jump_to_linked_uncod_thumb(u32 addr)
{
    armcpu_t &proc = ARMPROC;
    proc.R[15] += addr;
    proc.next_instruction = proc.R[15];
    proc.instruct_adr     = proc.R[15];
    return 1;
}

u32 _read32(u32 addr, u32 addr_mem)
{
    if ((addr & (~0x3FFF)) == MMU.DTCMRegion)
        return *(u32 *)(MMU.ARM9_DTCM + (addr & 0x3FFC));

    return *(u32 *)(MMU.MAIN_MEM + addr_mem);
}

void _write16(u32 addr, u16 val)
{
    if ((addr & (~0x3FFF)) == MMU.DTCMRegion)
    {
        T1WriteWord(MMU.ARM9_DTCM, addr & 0x3FFE, (u16)val);
        return;
    }
    const WritePageEntry &e = mmu_write_lut_arm9[addr >> 24];
    if (e.base)
    {
        if (e.flags & WPE_JIT_INVAL)
            JIT_COMPILED_FUNC_KNOWNBANK(addr, MAIN_MEM, _MMU_MAIN_MEM_MASK16, 0) = 0;
        T1WriteWord(e.base, addr & (e.mask & ~1u), val);
        return;
    }
    _MMU_ARM9_write16(addr, val);
}

void _write32(u32 addr, u32 val)
{
    if ((addr & (~0x3FFF)) == MMU.DTCMRegion)
    {
        T1WriteLong(MMU.ARM9_DTCM, addr & 0x3FFC, val);
        return;
    }
    const WritePageEntry &e = mmu_write_lut_arm9[addr >> 24];
    if (e.base)
    {
        if (e.flags & WPE_JIT_INVAL)
        {
            JIT_COMPILED_FUNC_KNOWNBANK(addr, MAIN_MEM, _MMU_MAIN_MEM_MASK32, 0) = 0;
            JIT_COMPILED_FUNC_KNOWNBANK(addr, MAIN_MEM, _MMU_MAIN_MEM_MASK32, 1) = 0;
        }
        T1WriteLong(e.base, addr & (e.mask & ~3u), val);
        return;
    }
    _MMU_ARM9_write32(addr, val);
}

// ARM7 counterparts. The helpers above are ARM9-specific in three ways: DTCM
// exists only on the ARM9, mmu_write_lut_arm9 describes the ARM9's memory map,
// and the JIT invalidation they do is for ARM9 blocks. The ARM7 shares main RAM
// (and SIWRAM, depending on WRAMCNT) with the ARM9, so its writes still have to
// invalidate ARM9 blocks there -- but through the ARM7's own view.
u32 _read32_arm7(u32 addr, u32 addr_mem)
{
    return _MMU_ARM7_read32(addr);
}

void _write16_arm7(u32 addr, u16 val)
{
    if (JIT_MAPPED(addr, ARMCPU_ARM7))
        JIT_COMPILED_FUNC_PREMASKED(addr, ARMCPU_ARM7, 0) = 0;
    // Main RAM is shared: an ARM7 write there can invalidate ARM9 code.
    if ((addr >> 24) == 0x02)
        JIT_COMPILED_FUNC_KNOWNBANK(addr, MAIN_MEM, _MMU_MAIN_MEM_MASK16, 0) = 0;
    _MMU_ARM7_write16(addr, val);
}

void _write32_arm7(u32 addr, u32 val)
{
    if (JIT_MAPPED(addr, ARMCPU_ARM7))
    {
        JIT_COMPILED_FUNC_PREMASKED(addr, ARMCPU_ARM7, 0) = 0;
        JIT_COMPILED_FUNC_PREMASKED(addr, ARMCPU_ARM7, 1) = 0;
    }
    if ((addr >> 24) == 0x02)
    {
        JIT_COMPILED_FUNC_KNOWNBANK(addr, MAIN_MEM, _MMU_MAIN_MEM_MASK32, 0) = 0;
        JIT_COMPILED_FUNC_KNOWNBANK(addr, MAIN_MEM, _MMU_MAIN_MEM_MASK32, 1) = 0;
    }
    _MMU_ARM7_write32(addr, val);
}

// Pick the right helper for the block being compiled.
#define JIT_READ32(PN)  ((PN) == ARMCPU_ARM9 ? (u32)_read32  : (u32)_read32_arm7)
#define JIT_WRITE16(PN) ((PN) == ARMCPU_ARM9 ? (u32)_write16 : (u32)_write16_arm7)
#define JIT_WRITE32(PN) ((PN) == ARMCPU_ARM9 ? (u32)_write32 : (u32)_write32_arm7)

#include "FIFO.h"
#include "NDSSystem.h"

void write_3D_cmd(u32 addr, u32 data)
{
    extern void gfx3d_sendCommand(u32 cmd, u32 param);

    /*case 0x400044:
    case 0x400045:
    case 0x400046:
    case 0x400047:
    case 0x400048:
    case 0x400049:
    case 0x40004A:
    case 0x40004B:
    case 0x40004C:
    case 0x40004D:
    case 0x40004E:
    case 0x40004F:
    case 0x400050:
    case 0x400051:
    case 0x400052:
    case 0x400053:
    case 0x400054:
    case 0x400055:
    case 0x400056:
    case 0x400057:
    case 0x400058:
    case 0x400059:
    case 0x40005A:
    case 0x40005B:
    case 0x40005C:		// Individual Commands*/
    if (gxFIFO.size > 254)
        nds.freezeBus |= 1;

    ((u32 *)(MMU.MMU_MEM[ARMCPU_ARM9][0x40]))[(addr & 0xFFF) >> 2] = data;
    gfx3d_sendCommand(addr, data);
}

void write_3D_fifo(u32 addr, u32 data)
{
    extern void gfx3d_sendCommandToFIFO(u32 val);
    ((u32 *)(MMU.MMU_MEM[ARMCPU_ARM9][0x40]))[addr] = data;
    gfx3d_sendCommandToFIFO(data);
}

template <uint32_t size, int PROCNUM>
void write_dma(u32 val, u32 i, s32 shift_op, u32 type)
{

    extern u32 get_addr(u32 addr, s32 shift_op, u32 type, u32 i);

    u32 adr = get_addr(_ARMPROC.R[REG_POS(i, 16)], shift_op, type, i);
    // The -hle tree has a templated write_dma<PROCNUM,size>(DmaRegister) that
    // resolves the register at compile time. That is part of a wider MMU rework
    // which is not ported here, so route through this tree's runtime-dispatched
    // signature instead -- same underlying registers, just resolved at run time.
    MMU_new.write_dma(PROCNUM, size, adr, val);
}

template <int PROCNUM>
void arm_halt_hack()
{
    armcpu_t &proc = ARMPROC;
    proc.CPSR.bits.I = 0;
	proc.freeze |= CPU_FREEZE_WAIT_IRQ;
	NDS_Reschedule();
}

// ---------------------------------------------------------------------------
// Runtime cycle cost of an ARM multiply.
//
// On the ARM9 a multiply costs base+1 .. base+4 depending on how many
// significant bytes the Rs operand has -- a runtime property of the VALUE, not
// of the opcode, which is why instr_attributes marks these INSTR_CYCLES_V and
// instr_cycles() cannot price them (it sees only the opcode and returns 0).
// The interpreter charges the real cost in MUL_Mxx_END/MUL_UMxxL_END/
// MUL_SMxxL_END; a compiled block charged nothing, so every multiply made the
// ARM9 run ahead of the scheduler.
//
// The interpreter's test after each `v >>= 8` is:
//   signed   (MUL/MLA/SMULL/SMLAL):  (v == 0) || (v == 0xFF..F)
//   unsigned (UMULL/UMLAL):          (v == 0)
// "zero or all-ones" means the bits above that point are all copies of bit 31,
// so for the signed forms the test is `(Rs >> 8k) == (Rs >> 31)` arithmetic,
// which is branchless. The unsigned forms compare against zero directly.
//
// Emits into psp_t0 the extra cycles (1..4) and adds them to RCYC. Clobbers
// psp_t0/psp_t1/psp_t2 only -- all caller-save scratch here.
static void emit_mul_cycles(psp_gpr_t rs, bool is_signed)
{
    // psp_t2 = the value each shifted form is compared against:
    //   signed   -> sign propagation (Rs >> 31 arithmetic), i.e. 0 or ~0
    //   unsigned -> 0
    if (is_signed)
        emit_sra(psp_t2, rs, 31);
    else
        emit_move(psp_t2, psp_zero);

    // Start at 4 (the "all four bytes significant" case) and subtract for each
    // threshold the value falls under. Counting down keeps it branchless: the
    // three tests are independent and each contributes at most one step.
    emit_movi(psp_t0, 4);

    // k = 1, 2, 3  ->  shift by 8, 16, 24
    for (int k = 1; k <= 3; k++)
    {
        if (is_signed)
            emit_sra(psp_t1, rs, 8 * k);
        else
            emit_srl(psp_t1, rs, 8 * k);

        // psp_t1 = (shifted == compare) ? 1 : 0
        emit_xor(psp_t1, psp_t1, psp_t2);
        emit_sltiu(psp_t1, psp_t1, 1);

        // Each threshold met removes one cycle: 4 -> 3 -> 2 -> 1.
        emit_subu(psp_t0, psp_t0, psp_t1);
    }

#if JIT_CYCLES_IN_REG
    emit_addu(RCYC, RCYC, psp_t0);
#else
    emit_lui(psp_t9, DISP_HI(&executed_cycles));
    emit_lw(psp_t1, psp_t9, DISP_LO(&executed_cycles));
    emit_addu(psp_t1, psp_t1, psp_t0);
    emit_sw(psp_t1, psp_t9, DISP_LO(&executed_cycles));
#endif
}

// Direct MAIN_MEM access for EXTFL_CONSTADDR ops: no C call, so no register
// flush either. Stores clear the word's JIT slots exactly as _write16/_write32 do.
static void emit_const_mem(opcode &op)
{
    const uint32_t addr = (uint32_t)op.imm;
    const uint32_t off  = addr & _MMU_MAIN_MEM_MASK;
    const uint32_t host = (uint32_t)MMU.MAIN_MEM + off;
    const uint32_t hi   = ((host + 0x8000) >> 16) & 0xFFFF;
    const int32_t  lo   = (int16_t)(host & 0xFFFF);

    if (op._op == OP_STR || op._op == OP_STRH)
    {
        int32_t regs[1] = {(int32_t)op.rs1};
        regman.get(1, regs);
        emit_li(psp_t1, (uint32_t)&JIT.MAIN_MEM[off >> 1]);
        emit_sw(psp_zero, psp_t1, 0);
        if (op._op == OP_STR)
            emit_sw(psp_zero, psp_t1, 4);
        emit_lui(psp_t1, hi);
        if (op._op == OP_STR)
            emit_sw(regs[0], psp_t1, lo);
        else
            emit_sh(regs[0], psp_t1, lo);
        return;
    }

    int32_t regs[1] = {(int32_t)(op.rd | 0x10)};
    regman.get(1, regs);
    emit_lui(psp_t1, hi);
    switch (op._op)
    {
    case OP_LDR:   emit_lw(regs[0], psp_t1, lo);  break;
    case OP_LDRH:  emit_lhu(regs[0], psp_t1, lo); break;
    case OP_LDRSH: emit_lh(regs[0], psp_t1, lo);  break;
    default:       emit_lb(regs[0], psp_t1, lo);  break;   // OP_LDRSB
    }
    regman.mark_dirty((psp_gpr_t)regs[0]);
}

template <int PROCNUM>
void emitARMOP(opcode &op, const bool last_op)
{
    if (op.extra_flags & EXTFL_CONSTADDR)
    {
        emit_const_mem(op);
        return;
    }

    const bool rd_allocated = regman.is_mapped(op.rd);

    // printf("OP: %d\n", op._op);

    switch (op._op)
    {

    case OP_3D_CMD:
    {
        int32_t regs[1] = {op.rs1};

        conditional(

            regman.get(1, regs);

            emit_li(psp_a0, op.imm);
            emit_jal(write_3D_cmd);
            emit_move(psp_a1, regs[0]););
        break;
    }

    case OP_3D_FIFO:
    {
        int32_t regs[1] = {op.rs1};

        conditional(

            regman.get(1, regs);

            emit_movi(psp_a0, ((op.imm & 0xFFF) >> 2));
            emit_jal(write_3D_fifo);
            emit_move(psp_a1, regs[0]););
        break;
    }

    case OP_DMA:
    {

        int32_t regs[] = {op.rs1};

        printf("DMA: %d\n", op.imm);

        conditional(

            regman.get(1, regs);

            emit_li(psp_a1, op.op_pc);
            emit_li(psp_a2, op.imm);
            emit_li(psp_a3, op.preOpType);

            {
                emit_jal((write_dma<32, PROCNUM>));
            }

            emit_move(psp_a0, regs[0]););

        break;
    }

    case OP_ITP:
    {
        if (last_op)
        {
            check_flags();
        }

        conditional(

            if (!last_op) {
                check_flags();
            }

            emit_li(psp_a0, op.rs1);

            // PROFILER: count interpreter-fallback executions at runtime, plus a
            // per-opcode-class histogram so we know exactly which instructions
            // dominate the fallback. The instruction word (op.rs1) is known at
            // compile time, so the histogram bucket is a constant address.
            // t0/t1 are scratch and clobbered by the upcoming interpreter call.
#if PROFILER_ENABLED
            {
                extern u32 g_prof_itp_exec;
                extern u32 g_itp_hist[];
                // bucket: ARM class. cls=00 -> dataproc: opcode(4b)*2+S (0..31).
                //   cls=01 LDR/STR: 32 LDRi,33 LDRr,34 STRi,35 STRr.
                //   cls=10: 36 LDM,37 STM,38 B/BL. cls=11: 39 coproc/SWI.
                const u32 instr = op.rs1;
                const u32 cls = (instr >> 26) & 3;
                u32 bucket;
                if (cls == 0)
                    bucket = (((instr >> 21) & 0xF) << 1) | ((instr >> 20) & 1);
                else if (cls == 1)
                    bucket = 32 + (((instr >> 20) & 1) ? 0 : 2) + (((instr >> 25) & 1) ? 1 : 0);
                else if (cls == 2)
                    bucket = ((instr >> 25) & 1) ? 38 : (36 + (((instr >> 20) & 1) ? 0 : 1));
                else
                    bucket = 39;
                emit_li(psp_t0, (u32)&g_prof_itp_exec);
                emit_lw(psp_t1, psp_t0, 0);
                emit_addiu(psp_t1, psp_t1, 1);
                emit_sw(psp_t1, psp_t0, 0);
                emit_li(psp_t0, (u32)&g_itp_hist[bucket]);
                emit_lw(psp_t1, psp_t0, 0);
                emit_addiu(psp_t1, psp_t1, 1);
                emit_sw(psp_t1, psp_t0, 0);
            }
#endif // PROFILER_ENABLED

            uint32_t optmizeDelaySlot = emit_SlideDelay();

            emit_jal(arm_instructions_set[ARM9][INSTRUCTION_INDEX(op.rs1)]);
            emit_Write32(optmizeDelaySlot);

            if (!last_op) load_flags();)

            intr_instr++;
    }
    break;
    case OP_AND:
    {

        if (op.preOpType == PRE_OP_IMM)
        {
            arm_and<true, false>(op);
        }
        else
        {
            // printf("IMM: 0x%x\n", emit_getCurrAdr());
            arm_and<false, false>(op);
        }
    }
    break;

    case OP_BIC:
        if (op.preOpType == PRE_OP_IMM)
            arm_bic<true, false>(op);
        else
            arm_bic<false, false>(op);
        break;

    case OP_ORR:
    {

        if (op.preOpType == PRE_OP_IMM)
        {
            arm_or<true, false>(op);
        }
        else
        {
            arm_or<false, false>(op);
        }
        break;
    }

    case OP_EOR:
        if (op.preOpType == PRE_OP_IMM)
            arm_xor<true, false>(op);
        else
            arm_xor<false, false>(op);

        break;

    case OP_ADD:
    {
        if (op.preOpType == PRE_OP_IMM)
        {
            arm_add<true, false>(op);
        }
        else
        {
            /*if (op.rs1 == 15 && op.rd == 15 && op.preOpType == PRE_OP_LSL_IMM){

                int32_t regs[3] = {op.rd, op.rs1, op.rs2};
                conditional(
                    regman.get(3, regs);

                    arm_preop[op.preOpType]((psp_gpr_t)regs[2], psp_at, op);
                    emit_addu((psp_gpr_t)regs[0], (psp_gpr_t)regs[1], psp_at);

                    emit_sw((psp_gpr_t)regs[0], RCPU, _next_instr);
                );


                printf("Special ADD\n");
            }else*/
            arm_add<false, false>(op);
        }
        break;
    }
    case OP_SUB:
    {
        if (op.preOpType == PRE_OP_IMM)
        {
            arm_sub<true, false>(op);
        }
        else
        {
            arm_sub<false, false>(op);
        }
        break;
    }

    case OP_RSB:
        if (op.preOpType == PRE_OP_IMM)
            arm_sub<true, true>(op);
        else
            arm_sub<false, true>(op);
        break;

    case OP_ADC:
    {
        // rd = rn + op2 + C
        int32_t regs[3] = {(op.condition != -1) ? op.rd : (op.rd | 0x10), op.rs1, op.rs2};
        const int nregs = (op.preOpType == PRE_OP_IMM) ? 2 : 3;
        regman.get(nregs, regs);
        const psp_gpr_t dst = (op.condition != -1) ? psp_at : (psp_gpr_t)regs[0];
        conditional_branchless(regs[0], dst, {
            if (op.preOpType == PRE_OP_IMM) {
                emit_li(psp_t0, op.imm);
                emit_addu(dst, (psp_gpr_t)regs[1], psp_t0);
            } else {
                arm_preop[getTypeIdx(op.preOpType)]((psp_gpr_t)regs[2], psp_t0, op);
                emit_addu(dst, (psp_gpr_t)regs[1], psp_t0);
            }
            // Read C AFTER arm_preop: the register-shift preops use psp_v0/psp_v1
            // as scratch and would clobber the carry if it were loaded earlier.
            load_carry(psp_v1);
            emit_addu(dst, dst, psp_v1);
        });
        END_OP_CHKR15(regs[0]);
        break;
    }

    case OP_SBC:
    {
        // rd = rn - op2 + C - 1
        int32_t regs[3] = {(op.condition != -1) ? op.rd : (op.rd | 0x10), op.rs1, op.rs2};
        const int nregs = (op.preOpType == PRE_OP_IMM) ? 2 : 3;
        regman.get(nregs, regs);
        const psp_gpr_t dst = (op.condition != -1) ? psp_at : (psp_gpr_t)regs[0];
        conditional_branchless(regs[0], dst, {
            if (op.preOpType == PRE_OP_IMM) {
                emit_li(psp_t0, op.imm);
            } else {
                arm_preop[getTypeIdx(op.preOpType)]((psp_gpr_t)regs[2], psp_t0, op);
            }
            emit_subu(dst, (psp_gpr_t)regs[1], psp_t0);
            // Read C AFTER arm_preop (reg-shift preops clobber psp_v1).
            load_carry(psp_v1);
            emit_addu(dst, dst, psp_v1);
            emit_addiu(dst, dst, -1);
        });
        END_OP_CHKR15(regs[0]);
        break;
    }

    case OP_RSC:
    {
        // rd = op2 - rn + C - 1
        int32_t regs[3] = {(op.condition != -1) ? op.rd : (op.rd | 0x10), op.rs1, op.rs2};
        const int nregs = (op.preOpType == PRE_OP_IMM) ? 2 : 3;
        regman.get(nregs, regs);
        const psp_gpr_t dst = (op.condition != -1) ? psp_at : (psp_gpr_t)regs[0];
        conditional_branchless(regs[0], dst, {
            if (op.preOpType == PRE_OP_IMM) {
                emit_li(psp_t0, op.imm);
                emit_subu(dst, psp_t0, (psp_gpr_t)regs[1]);
            } else {
                arm_preop[getTypeIdx(op.preOpType)]((psp_gpr_t)regs[2], psp_t0, op);
                emit_subu(dst, psp_t0, (psp_gpr_t)regs[1]);
            }
            // Read C AFTER arm_preop (reg-shift preops clobber psp_v1).
            load_carry(psp_v1);
            emit_addu(dst, dst, psp_v1);
            emit_addiu(dst, dst, -1);
        });
        END_OP_CHKR15(regs[0]);
        break;
    }

    case OP_MUL:
    {
#ifdef JIT_OPT_PREDICATE_MUL
        // Always use movz/movn commit for conditional MUL — no branch needed.
        int32_t regs[3] = {(int32_t)op.rd, (int32_t)op.rs1, (int32_t)op.rs2};
        regman.get(3, regs);
        const psp_gpr_t dst = (op.condition != (uint32_t)(-1)) ? psp_at : (psp_gpr_t)regs[0];
        conditional_branchless((psp_gpr_t)regs[0], dst, {
            emit_mult((psp_gpr_t)regs[1], (psp_gpr_t)regs[2]);
            emit_mflo(dst);
        });
        END_OP(regs[0]);
#else
        int32_t regs[3] = {op.condition != -1 ? op.rd : (op.rd | 0x10), op.rs1, op.rs2};
        HANDLE_CONDITIONAL_NR15(
            { regman.get(3, regs); emit_mult(regs[1], regs[2]); emit_mflo(psp_at); },
            { regman.get(3, regs); emit_mult(regs[1], regs[2]); emit_mflo(regs[0]); });
#endif
        // Variable multiply cost, charged unconditionally: on ARM a failed
        // condition still burns the cycles. Rs is op.rs2; MUL is a signed form.
        if (!(op.extra_flags & EXTFL_MULCYC))
            emit_mul_cycles((psp_gpr_t)regs[2], true);
        break;
    }

    case OP_MLA:
    {
#ifdef JIT_OPT_MULACC
        // MLA: Rd = Rm*Rs + Rn (mod 2^32). Use madd to save one insn vs mult+mflo+addu.
        int32_t regs[4] = {(int32_t)op.rd, (int32_t)op.rs1, (int32_t)op.rs2, op.imm};
        regman.get(4, regs);
        const psp_gpr_t dst = (op.condition != (uint32_t)(-1)) ? psp_at : (psp_gpr_t)regs[0];
        conditional_branchless((psp_gpr_t)regs[0], dst, {
            emit_mtlo((psp_gpr_t)regs[3]);             // seed LO = Rn
            emit_madd((psp_gpr_t)regs[1], (psp_gpr_t)regs[2]); // LO += Rm*Rs (signed)
            emit_mflo(dst);                             // dst = (Rn + Rm*Rs) mod 2^32
        });
        END_OP(regs[0]);
#else
        int32_t regs[4] = {op.condition != -1 ? op.rd : (op.rd | 0x10), op.rs1, op.rs2, op.imm};
        HANDLE_CONDITIONAL_NR15(
            { regman.get(4, regs); emit_mult(regs[1], regs[2]); emit_mflo(psp_at); emit_addu(psp_at, psp_at, regs[3]); },
            { regman.get(4, regs); emit_mult(regs[1], regs[2]); emit_mflo(psp_at); emit_addu(regs[0], psp_at, regs[3]); });
#endif
        // MLA base is 2; the variable part depends on Rs (op.rs2), signed form.
        if (!(op.extra_flags & EXTFL_MULCYC))
            emit_mul_cycles((psp_gpr_t)regs[2], true);
        break;
    }

    case OP_UMUL:
    {
#ifdef JIT_OPT_MULACC
        // UMULL: {RdHi:RdLo} = Rm * Rs (unsigned 64-bit).
        // op.rd=RdHi, op.rs1=Rm, op.rs2=Rs, op.imm=RdLo
        const bool is_cond = (op.condition != (uint32_t)(-1));
        const int32_t rdhi_idx = (int32_t)op.rd  | (is_cond ? 0 : 0x10);
        const int32_t rdlo_idx = op.imm           | (is_cond ? 0 : 0x10);
        int32_t regs[4] = {rdhi_idx, rdlo_idx, (int32_t)op.rs1, (int32_t)op.rs2};
        regman.get(4, regs);

        emit_multu((psp_gpr_t)regs[2], (psp_gpr_t)regs[3]);

        if (is_cond) {
            generate_condition_check(op.condition);
            emit_mflo(psp_at);
            emit_mfhi(psp_v0);
            const bool isMovz = (op.condition < 8) ? (op.condition & 1) : !(op.condition & 1);
            if (isMovz) {
                emit_movz((psp_gpr_t)regs[1], psp_at, psp_s4);   // RdLo
                emit_movz((psp_gpr_t)regs[0], psp_v0, psp_s4);   // RdHi
            } else {
                emit_movn((psp_gpr_t)regs[1], psp_at, psp_s4);
                emit_movn((psp_gpr_t)regs[0], psp_v0, psp_s4);
            }
        } else {
            emit_mflo((psp_gpr_t)regs[1]);   // RdLo = LO
            emit_mfhi((psp_gpr_t)regs[0]);   // RdHi = HI
        }
        if (!op.dead_rd) {
            regman.mark_dirty((psp_gpr_t)regs[0]);
            regman.mark_dirty((psp_gpr_t)regs[1]);
        }
        // UMULL: unsigned form, so the interpreter tests (v == 0) only.
        // Rs is regs[3] in this case's layout.
        emit_mul_cycles((psp_gpr_t)regs[3], false);
#else
        // Without JIT_OPT_MULACC, ARM_OP_UMULL returns INTERPRET so this is dead.
#endif
        break;
    }

    case OP_UMLA:
    {
#ifdef JIT_OPT_MULACC
        // UMLAL: {RdHi:RdLo} += Rm * Rs (unsigned 64-bit accumulate).
        // op.rd=RdHi, op.rs1=Rm, op.rs2=Rs, op.imm=RdLo
        // Read current RdLo/RdHi first — do NOT use |0x10 (fresh alloc) since we need their values.
        const bool is_cond = (op.condition != (uint32_t)(-1));
        int32_t regs[4] = {(int32_t)op.rd, op.imm, (int32_t)op.rs1, (int32_t)op.rs2};
        regman.get(4, regs);

        emit_mtlo((psp_gpr_t)regs[1]);    // seed LO = current RdLo
        emit_mthi((psp_gpr_t)regs[0]);    // seed HI = current RdHi
        emit_maddu((psp_gpr_t)regs[2], (psp_gpr_t)regs[3]); // HI:LO += Rm*Rs (unsigned)

        if (is_cond) {
            generate_condition_check(op.condition);
            emit_mflo(psp_at);    // new RdLo
            emit_mfhi(psp_v0);    // new RdHi
            const bool isMovz = (op.condition < 8) ? (op.condition & 1) : !(op.condition & 1);
            if (isMovz) {
                emit_movz((psp_gpr_t)regs[1], psp_at, psp_s4);
                emit_movz((psp_gpr_t)regs[0], psp_v0, psp_s4);
            } else {
                emit_movn((psp_gpr_t)regs[1], psp_at, psp_s4);
                emit_movn((psp_gpr_t)regs[0], psp_v0, psp_s4);
            }
        } else {
            emit_mflo((psp_gpr_t)regs[1]);   // RdLo = new LO
            emit_mfhi((psp_gpr_t)regs[0]);   // RdHi = new HI
        }
        if (!op.dead_rd) {
            regman.mark_dirty((psp_gpr_t)regs[0]);
            regman.mark_dirty((psp_gpr_t)regs[1]);
        }
        // UMLAL: unsigned form; Rs is regs[3].
        emit_mul_cycles((psp_gpr_t)regs[3], false);
#else
        // Without JIT_OPT_MULACC, ARM_OP_UMLAL returns INTERPRET so this is dead.
#endif
        break;
    }

    case OP_CLZ:
    {
        int32_t regs[2] = {op.rd, op.rs1};
        regman.get(2, regs);

        const psp_gpr_t dst = (op.condition != -1) ? psp_at : (psp_gpr_t)regs[0];

        conditional_branchless(regs[0], dst, {
            emit_clz(dst, regs[1]);
        });

        regman.mark_dirty((psp_gpr_t)regs[0]);
        break;
    }

    case OP_QADD:
    {
#ifdef JIT_OPT_SATURATE
        // Rd = SAT_32(Rm + Rn). op.rd=Rd, op.rs1=Rm, op.rs2=Rn
        int32_t regs[3] = {(int32_t)op.rd, (int32_t)op.rs1, (int32_t)op.rs2};
        regman.get(3, regs);
        const psp_gpr_t rm = (psp_gpr_t)regs[1];
        const psp_gpr_t rn = (psp_gpr_t)regs[2];

        emit_addu(psp_t0, rm, rn);          // t0 = wrapping sum
        emit_xor(psp_t1, rm, rn);           // t1: bit31=1 if different signs
        emit_xor(psp_t2, rm, psp_t0);      // t2: bit31=1 if result sign != Rm sign
        emit_nor(psp_t1, psp_t1, psp_zero); // NOT: bit31=1 if SAME signs
        emit_and(psp_t1, psp_t1, psp_t2);  // overflow iff same signs AND sign changed
        emit_sra(psp_v0, rm, 31);           // v0 = 0 (Rm≥0) or -1 (Rm<0)
        emit_not(psp_v0, psp_v0);           // v0 = -1 (pos) or 0 (neg)
        emit_lui(psp_v1, 0x8000);           // v1 = 0x80000000 = INT32_MIN
        emit_xor(psp_v1, psp_v1, psp_v0);  // v1 = INT32_MAX (pos) or INT32_MIN (neg)
        emit_sra(psp_t1, psp_t1, 31);      // t1 = -1 (overflow) or 0 (no overflow)
        emit_movn(psp_t0, psp_v1, psp_t1); // if overflow: t0 = saturated value

        if (op.condition != (uint32_t)(-1)) {
            generate_condition_check(op.condition);
            const bool isMovz = (op.condition < 8) ? (op.condition & 1) : !(op.condition & 1);
            if (isMovz) emit_movz((psp_gpr_t)regs[0], psp_t0, psp_s4);
            else        emit_movn((psp_gpr_t)regs[0], psp_t0, psp_s4);
        } else {
            emit_move((psp_gpr_t)regs[0], psp_t0);
        }
        END_OP(regs[0]);
#endif
        break;
    }

    case OP_QSUB:
    {
#ifdef JIT_OPT_SATURATE
        // Rd = SAT_32(Rm - Rn). op.rd=Rd, op.rs1=Rm, op.rs2=Rn
        int32_t regs[3] = {(int32_t)op.rd, (int32_t)op.rs1, (int32_t)op.rs2};
        regman.get(3, regs);
        const psp_gpr_t rm = (psp_gpr_t)regs[1];
        const psp_gpr_t rn = (psp_gpr_t)regs[2];

        emit_subu(psp_t0, rm, rn);          // t0 = wrapping difference
        emit_xor(psp_t1, rm, rn);           // t1: bit31=1 if different signs
        emit_xor(psp_t2, rm, psp_t0);      // t2: bit31=1 if result sign != Rm sign
        emit_and(psp_t1, psp_t1, psp_t2);  // overflow iff diff signs AND sign changed
        emit_sra(psp_v0, rm, 31);
        emit_not(psp_v0, psp_v0);
        emit_lui(psp_v1, 0x8000);
        emit_xor(psp_v1, psp_v1, psp_v0);
        emit_sra(psp_t1, psp_t1, 31);
        emit_movn(psp_t0, psp_v1, psp_t1);

        if (op.condition != (uint32_t)(-1)) {
            generate_condition_check(op.condition);
            const bool isMovz = (op.condition < 8) ? (op.condition & 1) : !(op.condition & 1);
            if (isMovz) emit_movz((psp_gpr_t)regs[0], psp_t0, psp_s4);
            else        emit_movn((psp_gpr_t)regs[0], psp_t0, psp_s4);
        } else {
            emit_move((psp_gpr_t)regs[0], psp_t0);
        }
        END_OP(regs[0]);
#endif
        break;
    }

    case OP_SMULXY:
    {
#ifdef JIT_OPT_SIGNEDMUL16
        // Rd = sign16(Rm.x) * sign16(Rs.y). op.imm: bit0=Rm_top, bit1=Rs_top
        const int rm_top = (int32_t)op.imm & 1;
        const int rs_top = ((int32_t)op.imm >> 1) & 1;
        int32_t regs[3] = {(int32_t)op.rd, (int32_t)op.rs1, (int32_t)op.rs2};
        regman.get(3, regs);

        if (rm_top) emit_sra(psp_v0, (psp_gpr_t)regs[1], 16);
        else        emit_seh(psp_v0, (psp_gpr_t)regs[1]);

        if (rs_top) emit_sra(psp_v1, (psp_gpr_t)regs[2], 16);
        else        emit_seh(psp_v1, (psp_gpr_t)regs[2]);

        const psp_gpr_t dst = (op.condition != (uint32_t)(-1)) ? psp_at : (psp_gpr_t)regs[0];
        conditional_branchless((psp_gpr_t)regs[0], dst, {
            emit_mult(psp_v0, psp_v1);
            emit_mflo(dst);
        });
        END_OP(regs[0]);
#endif
        break;
    }

    case OP_SMLAXY:
    {
#ifdef JIT_OPT_SIGNEDMUL16
        // Rd = (sign16(Rm.x) * sign16(Rs.y)) + Rn (mod 2^32).
        // op.imm: [3:0]=Rn_reg, [4]=Rm_top, [5]=Rs_top
        const int rn_reg = (int32_t)op.imm & 0xF;
        const int rm_top = ((int32_t)op.imm >> 4) & 1;
        const int rs_top = ((int32_t)op.imm >> 5) & 1;
        int32_t regs[4] = {(int32_t)op.rd, (int32_t)op.rs1, (int32_t)op.rs2, rn_reg};
        regman.get(4, regs);

        if (rm_top) emit_sra(psp_v0, (psp_gpr_t)regs[1], 16);
        else        emit_seh(psp_v0, (psp_gpr_t)regs[1]);

        if (rs_top) emit_sra(psp_v1, (psp_gpr_t)regs[2], 16);
        else        emit_seh(psp_v1, (psp_gpr_t)regs[2]);

        const psp_gpr_t dst = (op.condition != (uint32_t)(-1)) ? psp_at : (psp_gpr_t)regs[0];
        conditional_branchless((psp_gpr_t)regs[0], dst, {
            emit_mtlo((psp_gpr_t)regs[3]);   // seed LO = Rn
            emit_madd(psp_v0, psp_v1);        // LO += Rm.x * Rs.y (signed)
            emit_mflo(dst);
        });
        END_OP(regs[0]);
#endif
        break;
    }

    case OP_SWI:
    {
        conditional(
            emit_jal(cpu->swi_tab[op.rs1]);

            if (flag_dirty) {
                store_flags();
                flag_dirty = false;
            } else emit_nop();)
        break;
    }

    case OP_MOV:
    case OP_MVN:
    {
        int32_t regs[2] = {op.condition != -1 ? op.rd : op.rd | 0x10, op.rs2};

        if (op.preOpType == PRE_OP_IMM)
        {
            regman.get(1, regs);
            const psp_gpr_t dst = (op.condition != -1) ? psp_at : (psp_gpr_t)regs[0];

            conditional_branchless(regs[0], ((op.imm == 0 && op.condition != -1) ? psp_zero : dst),
                                   {
                                       if (op.imm != 0)
                                           emit_li(dst, op.imm);
                                       else /*if (op.imm == 0 && op.condition == -1)*/
                                           emit_move(dst, psp_zero);
                                   });
        }
        else
        {
            regman.get(2, regs);
            const psp_gpr_t dst = (op.condition != -1) ? psp_at : (psp_gpr_t)regs[0];

            conditional_branchless(regs[0], dst,
                                   {
                                       arm_preop[getTypeIdx(op.preOpType)]((psp_gpr_t)regs[1], dst, op);

                                       if (op._op == OP_MVN)
                                           emit_not(dst, dst);
                                   });
        }
        END_OP_CHKR15(regs[0])
        break;
    }

    case OP_BXRC:
    {
        regman.flush_all(false);
        regman.reset();

        check_flags()

            emit_movi(psp_v0, 1);
        conditional(
            emit_jal((jump_to_linked_br<true, PROCNUM>));
            loadReg(psp_a0, op.rs1);) break;
    }

    case OP_BC:
    {
        // For an UNCONDITIONAL B the block always leaves here: flush+reset as before.
        // For a CONDITIONAL B the not-taken path falls through to the next op in this
        // same block, so flushing/resetting the SRA cache up front throws away regs
        // the fall-through still needs (very common: cmp;bne that continues).
        // Move the flush inside the taken path (keep_dirty=true keeps the cache live
        // for the fall-through while still spilling to ARMPROC for jump_to_linked_bc).
        const bool uncond = (op.condition == ((uint32_t)-1));
        if (uncond) {
            regman.flush_all(false);
            regman.reset();
        }

        check_flags()

            emit_movi(psp_v0, 1);
        conditional(
            if (!uncond) regman.flush_all(true);
            emit_li(psp_a0, op.imm);
            u32 op = emit_SlideDelay();
            emit_jal((jump_to_linked_bc<true, PROCNUM>));
            emit_Write32(op);)
    }
    break;

    case OP_BLC:
    {
        // Same fall-through optimization as OP_BC: only flush up front for the
        // unconditional case; conditional BL keeps the SRA cache live for fall-through.
        const bool uncond = (op.condition == ((uint32_t)-1));
        if (uncond) {
            regman.flush_all(false);
            regman.reset();
        }

        check_flags()

            // Mirror OP_BC but via jump_to_linked_blc, which also sets R14 (link
            // register = return address) and then directly runs the target block
            // if it's already compiled — keeping the chain alive across BL (very
            // common: every function call). Previously BL fell through to
            // continue_cpu_check, breaking the chain on every call.
            emit_movi(psp_v0, 1);
        conditional(
            if (!uncond) regman.flush_all(true);
            emit_li(psp_a0, op.imm);
            u32 op = emit_SlideDelay();
            emit_jal((jump_to_linked_blc<true, PROCNUM>));
            emit_Write32(op);)
    }
    break;

    case OP_BXC:
    {
        regman.flush_all(false);
        regman.reset();

        check_flags()

            // printf("0x%x, %d\n", (u32)emit_getCurrAdr(), op.rs1);

            emit_movi(psp_v0, 1);
        conditional(
            emit_jal((jump_to_linked<true, PROCNUM>));
            loadReg(psp_a0, op.rs1);)
        /*int32_t regs[1] = {op.rs1};

        emit_movi(psp_v0, 3);

        //printf("0x%x\n", (u32)emit_getCurrAdr());
        conditional(
            regman.get(1, regs);

            if (is_u16(op.imm)){
                emit_addiu(regs[0], regs[0], op.imm);
            }else{
                emit_li(psp_a0, op.imm);
                emit_addu(regs[0], regs[0], psp_a0);
            }

            printf("%d\n", (NDS_ARM9.CPSR.bits.T ));
            NDS_ARM9.CPSR.bits.T = BIT0(addr);
            NDS_ARM9.R[15] = addr & (0xFFFFFFFC|(NDS_ARM9.CPSR.bits.T<<1));

            emit_andi(psp_a0, regs[0], 0x1);

            emit_ins(psp_gp, psp_a0, _flag_C8, _flag_C8);
            emit_li(psp_a1, 0xFFFFFFFC);

            emit_sll(psp_a0, psp_a0, 1);
            emit_or(psp_a0, psp_a0, psp_a1);

            emit_and(regs[0], );

            emit_sw(psp_a0,  RCPU, );
            emit_sw(regs[0], RCPU, _instr_adr);
            emit_sw(regs[0], RCPU, _next_instr);
            regman.mark_dirty((psp_gpr_t)regs[0]);
        )*/
    }
    break;

    case OP_STR:
    case OP_STRH:
    {
        int32_t regs[3] = {op.rd, op.rs1, op.rs2};

        regman.flush_all(true);

        check_flags()

            conditional(
                regman.get(3, regs);

                const int special_type = op.preOpType >> 16;
                const bool is_special_type = (special_type == PRE_OP_LSL_IMM || special_type == PRE_OP_LSR_IMM ||
                                              special_type == PRE_OP_ASR_IMM || special_type == PRE_OP_ROR_IMM);
                // 0x1ff not 0xff: PRE_OP_IMM_POST_M==257 needs bit 8 (see OP_LDR).
                const int type = op.preOpType & 0x1ff;

                const bool positive = op.imm >= 0;

                if (is_special_type) {
                    const int idx = getTypeIdx((opType)special_type);

                    // Take |imm| for the barrel-shifter preop WITHOUT mutating
                    // op.imm: emitARMOP takes `opcode &op` by reference and the
                    // block is emitted more than once, so flipping the sign here
                    // made it stick. On the next emission the negative (downward)
                    // offset had become positive, and the post-index / write-back
                    // forms then ADDED where they should subtract -- difftest saw
                    // the offset applied with the wrong sign or twice (all 48
                    // failures were P=0 or W=1; plain pre-index never writes back
                    // and so never showed it).
                    op.imm = op.imm < 0 ? -op.imm : op.imm;

                    if (idx >= 0)
                        arm_preop[idx]((psp_gpr_t)regs[2], psp_t0, op);
                    else
                        printf("STR not encoded correctly: %d\n", idx);
                    // printf("Addr: %x\n", emit_getCurrAdr());
                    // printf("STR not encoded correctly: %d\n", idx);
                }

                if (type == PRE_OP_IMM_PRE_P || type == PRE_OP_IMM_PRE_M) {
                    if (!is_special_type)
                        emit_addiu(psp_a0, regs[0], op.imm);
                    else
                    {
                        if (positive)
                            emit_addu(psp_a0, regs[0], psp_t0);
                        else
                            emit_subu(psp_a0, regs[0], psp_t0);
                    }

                    storeReg(psp_a0, op.rd);
                } else if (type == PRE_OP_IMM_POST_P || type == PRE_OP_IMM_POST_M) {
                    emit_move(psp_a0, regs[0]);

                    if (!is_special_type)
                        emit_addiu(regs[0], regs[0], op.imm);
                    else
                    {
                        if (positive)
                            emit_addu(regs[0], regs[0], psp_t0);
                        else
                            emit_subu(regs[0], regs[0], psp_t0);
                    }

                    storeReg((psp_gpr_t)regs[0], op.rd);
                } else if (type == PRE_OP_IMM) {
                    if (is_special_type)
                    {
                        if (positive)
                            emit_addu(psp_a0, regs[0], psp_t0);
                        else
                            emit_subu(psp_a0, regs[0], psp_t0);
                    }
                    else
                    {
                        if (is_u16(op.imm))
                        {
                            emit_addiu(psp_a0, regs[0], op.imm);
                        }
                        else
                        {
                            emit_li(psp_a0, op.imm);
                            emit_addu(psp_a0, regs[0], psp_a0);
                        }
                    }
                }

                emit_move(psp_a1, regs[1]);

                if (OP_STRH == op._op) {
                    /*if (op.extra_flags & EXTFL_DIRECTMEMACCESS){
                        u32 op = emit_SlideDelay();
                        emit_jal(JIT_WRITE16(PROCNUM));
                        emit_Write32(op);
                    }else*/
                    {
                        emit_jal(JIT_WRITE16(PROCNUM));
                        emit_ins(psp_a0, psp_zero, 0, 0);
                    }
                } else {
                    if (jit_opt_fastmem && PROCNUM == ARMCPU_ARM9)
                    {
                        // Inline fastmem path for word STR to ARM9 main RAM.
                        // Mirrors the OP_LDR fastmem path but also invalidates the
                        // JIT block-cache entries for the written word (required for
                        // self-modifying code correctness).
                        // a0 = effective address (not yet aligned), a1 = value to store.
                        // Scratch: t1, t2, v0.  Layout (27 insns):
                        //   0–5  : DTCM check -> SLOW
                        //   6–10 : main-RAM check (addr>>24 == 0x02) -> SLOW
                        //   11–13: word-align offset in t2 = addr & 0x3FFFFC
                        //   14–19: JIT invalidation (clear two uintptr_t slots)
                        //   20–24: direct store to MAIN_MEM (store in delay slot)
                        //   25–26: SLOW: jal _write32 + align delay slot
                        //   27   : DONE
                        const u32 dtcmRegAddr = (u32)&MMU.DTCMRegion;
                        const u32 mainMemBase = (u32)MMU.MAIN_MEM;
                        const u32 jitBase     = (u32)&JIT.MAIN_MEM[0];
                        const u32 dtcmHi = ((dtcmRegAddr + 0x8000) >> 16) & 0xFFFF;
                        const u32 dtcmLo = (u32)(s16)dtcmRegAddr;
                        const u32 mmHi   = ((mainMemBase + 0x8000) >> 16) & 0xFFFF;
                        const u32 mmLo   = (u32)(s16)mainMemBase;
                        const u32 jitHi  = ((jitBase + 0x8000) >> 16) & 0xFFFF;
                        const u32 jitLo  = (u32)(s16)jitBase;
                        const u32 fmBase = (u32)emit_GetPtr();
                        const u32 SLOW   = fmBase + 25 * 4;
                        const u32 DONE   = fmBase + 27 * 4;

                        emit_srl(psp_t2, psp_a0, 14);          // 0
                        emit_sll(psp_t2, psp_t2, 14);          // 1: t2 = addr & ~0x3FFF
                        emit_lui(psp_t1, dtcmHi);              // 2
                        emit_lw(psp_t1, psp_t1, dtcmLo);      // 3: t1 = DTCMRegion
                        emit_beq(psp_t2, psp_t1, SLOW);       // 4: DTCM -> slow
                        emit_nop();                             // 5
                        emit_srl(psp_t2, psp_a0, 24);          // 6
                        emit_andi(psp_t2, psp_t2, 0x0F);       // 7
                        emit_movi(psp_t1, 0x02);                // 8
                        emit_bne(psp_t2, psp_t1, SLOW);       // 9: not main RAM -> slow
                        emit_nop();                             // 10
                        emit_sll(psp_t2, psp_a0, 10);          // 11
                        emit_srl(psp_t2, psp_t2, 10);          // 12: t2 = addr & 0x3FFFFF
                        emit_ins(psp_t2, psp_zero, 1, 0);      // 13: t2 = addr & 0x3FFFFC
                        emit_sll(psp_v0, psp_t2, 1);           // 14: v0 = JIT byte offset
                        emit_lui(psp_t1, jitHi);               // 15
                        emit_addiu(psp_t1, psp_t1, jitLo);    // 16: t1 = &JIT.MAIN_MEM[0]
                        emit_addu(psp_t1, psp_t1, psp_v0);    // 17
                        emit_sw(psp_zero, psp_t1, 0);          // 18: clear ARM9 JIT slot
                        emit_sw(psp_zero, psp_t1, 4);          // 19: clear ARM7 JIT slot
                        emit_lui(psp_t1, mmHi);                // 20
                        emit_addiu(psp_t1, psp_t1, mmLo);     // 21: t1 = MAIN_MEM base
                        emit_addu(psp_t1, psp_t1, psp_t2);    // 22
                        emit_beq(psp_zero, psp_zero, DONE);   // 23: -> DONE
                        emit_sw(psp_a1, psp_t1, 0);           // 24: delay slot: store word
                        // SLOW (25):
                        emit_jal(JIT_WRITE32(PROCNUM));                    // 25
                        emit_ins(psp_a0, psp_zero, 1, 0);     // 26: delay slot: align a0
                        // DONE (27):
                    }
                    else
                    {
                        emit_jal(JIT_WRITE32(PROCNUM));
                        emit_ins(psp_a0, psp_zero, 1, 0);
                    }
                }

                regman.reset();

                /*if (op.condition == -1) {
                    //optimization: mantain those regs since they aren't affected by the storing function
                    regman.map(op.rd, (psp_gpr_t)regs[0]);
                    regman.map(op.rs1, (psp_gpr_t)regs[1]);
                    //printf("Mapped %d to %d 0x%x\n", op.rd, regs[0], (u32)emit_getCurrAdr());
                }*/
            )
    }
    break;

    case OP_LDRSH:
    case OP_LDRSB:
    case OP_LDRH:
    case OP_LDR:
    {
        const int special_type = op.preOpType >> 16;
        const bool is_special_type = (special_type == PRE_OP_LSL_IMM || special_type == PRE_OP_LSR_IMM ||
                                      special_type == PRE_OP_ASR_IMM || special_type == PRE_OP_ROR_IMM);
        // NB: addressing mode lives in the low 16 bits (shift type is at >>16).
        // Must mask 0x1ff, not 0xff: PRE_OP_IMM_POST_M == (256|1) == 257, so 0xff
        // would drop bit 8 and mis-handle post-index as a plain offset load.
        const int type = op.preOpType & 0x1ff;

        const bool positive = op.imm >= 0;

        int32_t regs[3] = {op.rd | 0x10, op.rs1, is_special_type ? op.rs2 : -1};

        bool skip = false;

        /*if (op.extra_flags & EXTFL_DIRECTMEMACCESS && op.condition == -1 && op._op == OP_LDR && !special_type) {
            skip = true;
        } else {
            regman.flush_all(true);
        }*/

        regman.flush_all(true);

        check_flags()

            conditional(
                regman.get(3, regs);

                if (is_special_type) {
                    const int idx = getTypeIdx((opType)special_type);

                    // Take |imm| for the barrel-shifter preop WITHOUT mutating
                    // op.imm: emitARMOP takes `opcode &op` by reference and the
                    // block is emitted more than once, so flipping the sign here
                    // made it stick. On the next emission the negative (downward)
                    // offset had become positive, and the post-index / write-back
                    // forms then ADDED where they should subtract -- difftest saw
                    // the offset applied with the wrong sign or twice (all 48
                    // failures were P=0 or W=1; plain pre-index never writes back
                    // and so never showed it).
                    op.imm = op.imm < 0 ? -op.imm : op.imm;

                    if (idx >= 0)
                        arm_preop[idx]((psp_gpr_t)regs[2], psp_t0, op);
                    else
                        printf("LDR not encoded correctly: %d\n", idx);
                }

                if (type == PRE_OP_IMM) {
                    if (is_special_type)
                    {
                        if (positive)
                            emit_addu(psp_a0, regs[1], psp_t0);
                        else
                            emit_subu(psp_a0, regs[1], psp_t0);
                    }
                    else
                    {
                        if (is_u16(op.imm))
                        {
                            emit_addiu(psp_a0, regs[1], op.imm);
                        }
                        else
                        {
                            emit_li(psp_a0, op.imm);
                            emit_addu(psp_a0, regs[1], psp_a0);
                        }
                    }
                } else if (type == PRE_OP_IMM_PRE_P || type == PRE_OP_IMM_PRE_M) {
                    if (is_special_type)
                    {
                        if (positive)
                            emit_addu(psp_a0, regs[1], psp_t0);
                        else
                            emit_subu(psp_a0, regs[1], psp_t0);
                    }
                    else
                    {
                        emit_addiu(psp_a0, regs[1], op.imm);
                    }
                    storeReg(psp_a0, op.rs1);
                } else if (type == PRE_OP_IMM_POST_P || type == PRE_OP_IMM_POST_M) {
                    emit_move(psp_a0, regs[1]);

                    if (is_special_type)
                    {
                        if (positive)
                            emit_addu(psp_t1, psp_a0, psp_t0);
                        else
                            emit_subu(psp_t1, psp_a0, psp_t0);
                    }
                    else
                    {
                        emit_addiu(psp_t1, psp_a0, op.imm);
                    }

                    storeReg(psp_t1, op.rs1);
                }

                if (OP_LDR == op._op) {
                    if (jit_opt_fastmem && PROCNUM == ARMCPU_ARM9)
                    {
                        // Runtime fast/slow path for word LDR (ARM9). Inline the
                        // common main-RAM load; route DTCM and everything else to
                        // the slow _MMU_read32 (which handles DTCM correctly).
                        // a0 = effective address. regs[0] = dest. Scratch: t1,t2,v0.
                        // Instruction layout is fixed-size so branch offsets are exact.
                        const u32 dtcmRegAddr = (u32)&MMU.DTCMRegion;
                        const u32 mainMemBase = (u32)MMU.MAIN_MEM;
                        const u32 dtcmHi = ((dtcmRegAddr + 0x8000) >> 16) & 0xFFFF;
                        const u32 dtcmLo = (u32)(s16)dtcmRegAddr;
                        const u32 mmHi   = ((mainMemBase + 0x8000) >> 16) & 0xFFFF;
                        const u32 mmLo   = (u32)(s16)mainMemBase;
                        const u32 fmBase = (u32)emit_GetPtr();
                        const u32 SLOW = fmBase + 21 * 4;
                        const u32 HAVE = fmBase + 23 * 4;

                        emit_sll(regs[0], psp_a0, 3);                 // 0: rotate amount
                        emit_srl(psp_t2, psp_a0, 14);                 // 1
                        emit_sll(psp_t2, psp_t2, 14);                 // 2: t2 = addr & ~0x3FFF
                        emit_lui(psp_t1, dtcmHi);                     // 3
                        emit_lw(psp_t1, psp_t1, dtcmLo);             // 4: t1 = DTCMRegion
                        emit_beq(psp_t2, psp_t1, SLOW);               // 5: DTCM -> slow
                        emit_nop();                                   // 6
                        emit_srl(psp_t2, psp_a0, 24);                 // 7
                        emit_andi(psp_t2, psp_t2, 0x0F);              // 8
                        emit_movi(psp_t1, 0x02);                      // 9
                        emit_bne(psp_t2, psp_t1, SLOW);               // 10: not main RAM -> slow
                        emit_nop();                                   // 11
                        emit_sll(psp_t2, psp_a0, 10);                 // 12
                        emit_srl(psp_t2, psp_t2, 10);                 // 13: t2 = addr & 0x3FFFFF
                        emit_ins(psp_t2, psp_zero, 1, 0);             // 14: t2 = addr & 0x3FFFFC
                        emit_lui(psp_t1, mmHi);                       // 15
                        emit_addiu(psp_t1, psp_t1, mmLo);             // 16: t1 = MAIN_MEM base
                        emit_addu(psp_t1, psp_t1, psp_t2);            // 17
                        emit_lw(psp_v0, psp_t1, 0);                   // 18: v0 = word
                        emit_beq(psp_zero, psp_zero, HAVE);           // 19: -> have
                        emit_nop();                                   // 20
                        // SLOW (21):
                        emit_jal(_MMU_read32<PROCNUM>);               // 21
                        emit_ins(psp_a0, psp_zero, 1, 0);             // 22: delay slot: align a0
                        // HAVE (23):
                        emit_rotrv(regs[0], psp_v0, regs[0]);         // 23: ARM unaligned rotate
                    }
                    else if (op.extra_flags & EXTFL_DIRECTMEMACCESS)
                    {
                        // printf("DIRECT ACCEESS %x, %x, %x\n", MMU.ARM9_DTCM, MMU.MAIN_MEM, (u32)&MMU.DTCMRegion);
                        emit_sll(regs[0], psp_a0, 3);

                        emit_move(psp_a1, psp_a0);
                        emit_jal(JIT_READ32(PROCNUM));
                        emit_ins(psp_a1, psp_zero, 31, 22);

                        emit_rotrv(regs[0], psp_v0, regs[0]);

                        /*emit_lui(psp_v1, 0x9DF);
                        emit_or(psp_v1, RCPU, psp_v1);*/

                        /*emit_li(psp_v1, (u32)&MMU.MAIN_MEM);

                        emit_sll(regs[0], psp_a0, 3);

                        // Load the address of the memory
                        emit_ins(psp_a0, psp_zero, 1, 0);

                        // Load the address of the DTCM
                        emit_lw(psp_a1, psp_v1, dtcm_addr2);

                        emit_move(psp_a2, psp_a0);
                        emit_ins(psp_a0, psp_zero, 31, 22);

                        // Mask the last 14 bits for the DTCM
                        emit_ins(psp_a2, psp_zero, 13, 0);
                        emit_subu(psp_a2, psp_a1, psp_a2);

                        // If DTCM move the addr to a0
                        emit_andi(psp_a3, psp_a0, 0x3FFC);
                        emit_movz(psp_a0, psp_a3, psp_a2);

                        emit_addiu(psp_t3, psp_v1, dtcm_mem2);

                        emit_movz(psp_v1, psp_t3, psp_a2);

                        emit_addu(psp_v1, psp_v1, psp_a0);

                        emit_lw(psp_v0, psp_v1, 0);

                        emit_rotrv(regs[0], psp_v0, regs[0]);


                        /*emit_move(psp_a1, psp_a0);
                        emit_jal(JIT_READ32(PROCNUM));
                        emit_ins(psp_a1, psp_zero, 31, 22);

                        emit_li(psp_a1, JIT_READ32(PROCNUM));
                        emit_li(psp_a0, emit_getCurrAdr());
                        emit_jal(lastBeforeCrash);
                        emit_nop();*/
                    }
                    else
                    {
                        // printf("SLOW ACCEESS\n");

                        emit_sll(regs[0], psp_a0, 3);

                        emit_jal(_MMU_read32<PROCNUM>);
                        emit_ins(psp_a0, psp_zero, 1, 0);

                        emit_rotrv(regs[0], psp_v0, regs[0]);
                    }
                } else if (OP_LDRSB == op._op) {
                    u32 op = emit_SlideDelay();
                    emit_jal(_MMU_read08<PROCNUM>);
                    emit_Write32(op);
                    emit_seb(regs[0], psp_v0);
                } else {
                    emit_jal(_MMU_read16<PROCNUM>);
                    emit_ins(psp_a0, psp_zero, 0, 0);

                    if (op._op == OP_LDRSH)
                        emit_seh(regs[0], psp_v0);
                    else
                        emit_move(regs[0], psp_v0);
                }

                regman.reset();

                if (op.condition == -1) {
                    regman.map(op.rd, (psp_gpr_t)regs[0]);
                    regman.mark_dirty((psp_gpr_t)regs[0]);
                } else storeReg((psp_gpr_t)regs[0], op.rd);)
    }
    break;

    case OP_LDM:
    case OP_STM:
    {
        // op.bytes = full ARM opcode. Decode addressing form here.
        const u32 insn   = op.bytes;
        // Diagnostic: log what this LDM/STM was decoded as, plus the computed
        // start offset and writeback delta. Difftest shows the base register
        // ending at Rn-8n instead of Rn-4n (offset applied twice) and one LDM
        // reading all zeros, while the generator reads correct statically --
        // so print the actual decoded values.
        extern int g_dumpLdmDecode;
        if (g_dumpLdmDecode > 0 && ((insn >> 21) & 1)) {   // W=1 only: the failing forms
            g_dumpLdmDecode--;
            int dn = 0;
            for (u32 b = 0; b < 16; b++) if ((insn & 0xFFFF) & (1u << b)) dn++;
            const bool dP = (insn >> 24) & 1, dU = (insn >> 23) & 1, dW = (insn >> 21) & 1;
            ldm_log("[LDM] insn=%08X pc=%08X %s P=%d U=%d W=%d n=%d rlist=%04X "
                   "rs1=%d rd=%d imm=%d pre=%d cond=%X first_off=%d wb=%d\n",
                   (unsigned)insn, (unsigned)op.op_pc,
                   (op._op == OP_LDM) ? "LDM" : "STM",
                   (int)dP, (int)dU, (int)dW, dn, (unsigned)(insn & 0xFFFF),
                   (int)op.rs1, (int)op.rd, (int)op.imm, (int)op.preOpType,
                   (unsigned)op.condition,
                   dU ? (dP ? 4 : 0) : (dP ? -(dn * 4) : -((dn - 1) * 4)),
                   dU ? (dn * 4) : -(dn * 4));
        }
        const bool load  = (op._op == OP_LDM);
        const bool P     = (insn >> 24) & 1;   // pre/post
        const bool U     = (insn >> 23) & 1;   // up/down
        const bool W     = (insn >> 21) & 1;   // writeback
        const u32  rlist = insn & 0xFFFF;
        const u32  rn    = op.rs1;

        int n = 0;
        for (u32 b = 0; b < 16; b++)
            if (rlist & (1u << b)) n++;

        // DIAGNOSTIC: dump the MIPS emitted for the first W=1 block. wb_exec=1 in
        // difftest proves the writeback runs exactly once, yet the base ends at
        // Rn-8n instead of Rn-4n -- so something else in the emitted code also
        // moves rn. Read the actual instructions instead of the generator.
        extern int g_dumpLdmCode;
        const bool dump_this = (g_dumpLdmCode > 0) && (((insn >> 21) & 1) != 0);
        if (dump_this) { g_dumpLdmCode--; StartCodeDump(); }

        // psp_s5 holds the running effective address across the memory calls
        // (callee-saved, survives _read32/_write32). All NDS regs live in the
        // CPU struct here (flushed below), so we load/store them directly.
        regman.flush_all(false);
        regman.reset();

        check_flags()

        conditional(
            // base address into s5
            loadReg(psp_s5, rn);

            // Lowest register goes to lowest address. Compute address of the
            // lowest-numbered register's slot, then walk upward by +4.
            // IA: base ; IB: base+4 ; DA: base-(n-1)*4 ; DB: base-n*4
            int first_off;
            if (U)  first_off = P ? 4 : 0;
            else    first_off = P ? -(n * 4) : -((n - 1) * 4);

            if (first_off != 0)
                emit_addiu(psp_s5, psp_s5, first_off);

            // Writeback value computed up front (before any load can clobber rn).
            // n*4 <= 64, fits a signed 16-bit addiu immediate.
            if (W) {
                loadReg(psp_s7, rn);
                emit_addiu(psp_s7, psp_s7, U ? (n * 4) : -(n * 4));
                storeReg(psp_s7, rn);
            }

            int idx = 0;
            for (u32 b = 0; b < 16; b++) {
                if (!(rlist & (1u << b))) continue;

                if (load) {
                    // Mirror OP_LDR's proven read sequence (rotate is a no-op when aligned).
                    // Rotate amount must survive the call → use callee-saved s6.
                    emit_sll(psp_s6, psp_s5, 3);          // rotate amount = (addr&3)*8
                    emit_move(psp_a0, psp_s5);
                    emit_jal(_MMU_read32<PROCNUM>);
                    emit_ins(psp_a0, psp_zero, 1, 0);     // word-align addr (delay slot)
                    emit_rotrv(psp_v0, psp_v0, psp_s6);
                    storeReg(psp_v0, b);
                } else {
                    emit_move(psp_a0, psp_s5);
                    loadReg(psp_a1, b);
                    emit_jal(JIT_WRITE32(PROCNUM));
                    emit_ins(psp_a0, psp_zero, 1, 0);     // word-align addr (delay slot)
                }

                idx++;
                if (idx < n)
                    emit_addiu(psp_s5, psp_s5, 4);
            }

            regman.reset();
        )

        if (dump_this) {
            char path[256];
            if (getcwd(path, sizeof(path) - 32)) {
                strcat(path, "/ROMS/ldmcode.bin");
                CodeDump(path);
                printf("[LDMCODE] insn=%08X n=%d P=%d U=%d W=%d L=%d rn=%d dumped\n",
                       (unsigned)insn, n, (int)P, (int)U, (int)W, (int)load, (int)rn);
            }
        }
    }
    break;

    case OP_TST:
    {

        int32_t regs[2] = {op.rs1, op.rs2};

        conditional(
            regman.get(2, regs);

            if (op.preOpType == PRE_OP_IMM) {
                if (is_u16(op.imm))
                {
                    emit_andi(psp_v0, (psp_gpr_t)regs[0], op.imm);
                }
                else
                {
                    emit_li(psp_a0, op.imm);
                    emit_and(psp_v0, (psp_gpr_t)regs[0], psp_a0);
                }

                if ((op.rd >> 8) & 0xF)
                {
                    emit_ext(psp_t0, psp_a0, 31, 31);
                    emit_ins(psp_gp, psp_t0, _flag_C8, _flag_C8);
                }
            } else {
                arm_preop[getTypeIdx(op.preOpType)]((psp_gpr_t)regs[1], psp_t0, op);
                emit_and(psp_v0, (psp_gpr_t)regs[0], psp_t0);
            }

            emit_srl(psp_t0, psp_v0, 31);
            emit_ins(psp_gp, psp_t0, _flag_N8, _flag_N8);

            emit_sltiu(psp_t0, psp_v0, 1);
            emit_ins(psp_gp, psp_t0, _flag_Z8, _flag_Z8);

            // store_flags();

            set_flag_dirty();) 
            break;
    }

    case OP_CMN:
    {
        // CMN sets flags as Rn + op2 (compare with negative). The old codegen
        // wrongly subtracted (Rn - op2) and mixed add/sub flag formulas. Mirror
        // OP_CMP but use addition and set_add_flags (a0=left, a1=right, v0=res).
        int32_t regs[2] = {op.rs1, op.rs2};

        conditional(
            if (op.preOpType == PRE_OP_IMM) {
                regman.get(1, regs);
                emit_li(psp_a1, op.imm);
            } else {
                regman.get(2, regs);
                arm_preop[getTypeIdx(op.preOpType)]((psp_gpr_t)regs[1], psp_a1, op);
            }
            emit_addu(psp_v0, (psp_gpr_t)regs[0], psp_a1);
            emit_jal(set_add_flags);
            emit_move(psp_a0, (psp_gpr_t)regs[0]);
            set_flag_dirty();
        )
        break;
    }

    case OP_CMP:
    {

        int32_t regs[2] = {op.rs1, op.rs2};

           // printf("Cmp: 0x%x\n", op.op_pc);
        conditional(

            if (op.preOpType == PRE_OP_IMM) {
                regman.get(1, regs);
                emit_li(psp_a1, op.imm);
                emit_subu(psp_v0, (psp_gpr_t)regs[0], psp_a1);
            } else {
                regman.get(2, regs);
                arm_preop[getTypeIdx(op.preOpType)]((psp_gpr_t)regs[1], psp_a1, op);
                emit_subu(psp_v0, (psp_gpr_t)regs[0], psp_a1);
            }

            /*if (op.op_pc == 0x01FF8014){
                emit_move(psp_a0, psp_zero);
                emit_move(psp_a1, psp_zero);
                emit_subu(psp_v0, psp_a0, psp_a1);
                printf("PATCHED CMP\n");
            }*/

            // t0 = c
            // t1 = n
            // t2 = v
            // t3 = z

            /*emit_srl(psp_t1, psp_v0, 31);
            emit_sltiu(psp_t3, psp_v0, 1);
 

            {
                emit_sltu(psp_t0, (psp_gpr_t)regs[0], psp_a0);
                emit_xori(psp_t0, psp_t0, 1);

                // V
                emit_slt(psp_t2, (psp_gpr_t)regs[0], psp_a0);
                emit_xor(psp_t2, psp_t2, psp_t1);


                emit_ins(psp_gp, psp_t0, _flag_C8, _flag_C8);
                emit_ins(psp_gp, psp_t2, _flag_V8, _flag_V8);
            }



            emit_ins(psp_gp, psp_t1, _flag_N8, _flag_N8);
            emit_ins(psp_gp, psp_t3, _flag_Z8, _flag_Z8);*/

            emit_jal(set_sub_flags);
            emit_move(psp_a0, (psp_gpr_t)regs[0]);

            // store_flags();

            set_flag_dirty();
        ) break;
    }

    case OP_HALT_HACK:
    {
        emit_jal(arm_halt_hack<PROCNUM>);
        emit_nop(); 
        break;
    }

    case OP_ADD_S:
    {
        // rd = rn + op2, update NZCV
        int32_t regs[3] = {(op.condition != -1) ? op.rd : (op.rd | 0x10), op.rs1, op.rs2};
        const int nregs = (op.preOpType == PRE_OP_IMM) ? 2 : 3;
        regman.get(nregs, regs);
        const psp_gpr_t dst = (op.condition != -1) ? psp_at : (psp_gpr_t)regs[0];
        conditional_branchless_flags(regs[0], dst, psp_t3, {
            if (op.preOpType == PRE_OP_IMM) {
                emit_li(psp_a1, op.imm);
            } else {
                arm_preop[getTypeIdx(op.preOpType)]((psp_gpr_t)regs[2], psp_a1, op);
            }
            emit_addu(psp_v0, (psp_gpr_t)regs[1], psp_a1);
            emit_jal(set_add_flags);
            emit_move(psp_a0, (psp_gpr_t)regs[1]);
            emit_move(dst, psp_v0);
        });
        set_flag_dirty();
        END_OP_CHKR15(regs[0]);
        break;
    }

    case OP_SUB_S:
    {
        // rd = rn - op2, update NZCV
        int32_t regs[3] = {(op.condition != -1) ? op.rd : (op.rd | 0x10), op.rs1, op.rs2};
        const int nregs = (op.preOpType == PRE_OP_IMM) ? 2 : 3;
        regman.get(nregs, regs);
        const psp_gpr_t dst = (op.condition != -1) ? psp_at : (psp_gpr_t)regs[0];
        conditional_branchless_flags(regs[0], dst, psp_t3, {
            if (op.preOpType == PRE_OP_IMM) {
                emit_li(psp_a1, op.imm);
            } else {
                arm_preop[getTypeIdx(op.preOpType)]((psp_gpr_t)regs[2], psp_a1, op);
            }
            emit_subu(psp_v0, (psp_gpr_t)regs[1], psp_a1);
            emit_jal(set_sub_flags);
            emit_move(psp_a0, (psp_gpr_t)regs[1]);
            emit_move(dst, psp_v0);
        });
        set_flag_dirty();
        END_OP_CHKR15(regs[0]);
        break;
    }

    case OP_AND_S:
    case OP_EOR_S:
    case OP_ORR_S:
    {
        // rd = rn AND/EOR/ORR op2, update NZ (C,V from barrel shifter; keep existing for now)
        int32_t regs[3] = {(op.condition != -1) ? op.rd : (op.rd | 0x10), op.rs1, op.rs2};
        const int nregs = (op.preOpType == PRE_OP_IMM) ? 2 : 3;
        regman.get(nregs, regs);
        const psp_gpr_t dst = (op.condition != -1) ? psp_at : (psp_gpr_t)regs[0];
        const int logic_op = op._op;
        conditional_branchless_flags(regs[0], dst, psp_t3, {
            if (op.preOpType == PRE_OP_IMM) {
                emit_li(psp_t0, op.imm);
            } else {
                arm_preop[getTypeIdx(op.preOpType)]((psp_gpr_t)regs[2], psp_t0, op);
            }
            if (logic_op == OP_AND_S)
                emit_and(dst, (psp_gpr_t)regs[1], psp_t0);
            else if (logic_op == OP_EOR_S)
                emit_xor(dst, (psp_gpr_t)regs[1], psp_t0);
            else
                emit_or(dst, (psp_gpr_t)regs[1], psp_t0);
            emit_jal(set_op_logic_flags);
            emit_move(psp_v0, dst);
            // C from barrel shifter. set_op_logic_flags only writes N/Z, so the
            // carry stays whatever it was. For the immediate form the shifter
            // carry is a decode-time constant: rotate==0 keeps C, rotate!=0
            // sets C = bit31 of the rotated immediate.
            if (op.preOpType == PRE_OP_IMM) {
                const u32 rotate = (op.bytes >> 8) & 0xF;
                if (rotate) {
                    if ((op.imm >> 31) & 1) {
                        emit_li(psp_t1, 1);
                        // emit_ins(rt,rs,MSB,LSB) writes bits [LSB..MSB]. The old `1`
                        // as LSB made this write bits 1..5, clobbering V (bit4). Use
                        // _flag_C8 for BOTH msb and lsb → single-bit write to C only.
                        emit_ins(psp_gp, psp_t1, _flag_C8, _flag_C8);
                    } else {
                        emit_ins(psp_gp, psp_zero, _flag_C8, _flag_C8);
                    }
                }
            }
        });
        set_flag_dirty();
        END_OP_CHKR15(regs[0]);
        break;
    }

    case OP_SHIFT_S:
    {
        // Thumb LSL/LSR/ASR #imm, setting N,Z,C (V untouched). Mirrors
        // OP_LSL / OP_LSR / OP_ASR in thumb_instructions.cpp, including the
        // amount==0 forms which that file handles as separate opcodes:
        //   LSL #0 : Rd = Rm            ; C unchanged
        //   LSR #0 : Rd = 0             ; C = Rm>>31   (means LSR #32)
        //   ASR #0 : Rd = Rm>>31 ? ~0:0 ; C = Rm>>31   (means ASR #32)
        // Carry for a nonzero amount is the last bit shifted out:
        //   LSL: bit (32-amt) ; LSR/ASR: bit (amt-1)
        const uint32_t kind = SHIFT_S_KIND(op.imm);
        const uint32_t amt  = SHIFT_S_AMT(op.imm);

        int32_t regs[2] = {(op.condition != -1) ? op.rd : (op.rd | 0x10), op.rs1};
        regman.get(2, regs);
        const psp_gpr_t dst = (op.condition != -1) ? psp_at : (psp_gpr_t)regs[0];
        const psp_gpr_t src = (psp_gpr_t)regs[1];

        conditional_branchless_flags(regs[0], dst, psp_t3, {
            if (amt == 0) {
                if (kind == SHIFT_S_LSL) {
                    // Rd = Rm, C untouched.
                    emit_move(dst, src);
                } else {
                    // C = bit31 of Rm, computed BEFORE dst is written (dst may
                    // alias src).
                    emit_ext(psp_t0, src, 31, 31);
                    if (kind == SHIFT_S_LSR)
                        emit_move(dst, psp_zero);          // LSR #32 -> 0
                    else
                        emit_sra(dst, src, 31);            // ASR #32 -> sign
                    emit_ins(psp_gp, psp_t0, _flag_C8, _flag_C8);
                }
            } else {
                // Grab the carry bit first: dst can be the same register as src.
                const uint32_t cbit = (kind == SHIFT_S_LSL) ? (32u - amt) : (amt - 1u);
                emit_ext(psp_t0, src, cbit, cbit);
                if (kind == SHIFT_S_LSL)      emit_sll(dst, src, amt);
                else if (kind == SHIFT_S_LSR) emit_srl(dst, src, amt);
                else                          emit_sra(dst, src, amt);
                emit_ins(psp_gp, psp_t0, _flag_C8, _flag_C8);
            }
            // N and Z from the result.
            emit_jal(set_op_logic_flags);
            emit_move(psp_v0, dst);
        });
        set_flag_dirty();
        END_OP_CHKR15(regs[0]);
        break;
    }

    case OP_STMIA:
    case OP_NOP:
    case OP_LSR_0:
    case OP_MOV_S:
    case OP_MVN_S:
    case OP_NEG:
    default:
        printf("WTF? No opcode defined\n");

        break;
    }
}

#include "thumb_jit.h"

// Already compiled and optimized block
#include "precompiled_ops.h"


template <int PROCNUM>
bool block::emitThumbBlock()
{

    use_flags = true;
    flag_loaded = false;   // see the note in emitArmBlock()
    flag_dirty  = false;

    // reg_alloc.reset(); 

    regman.reset(true);
    emit_li(psp_fp, opcodes.front().op_pc);

    opcode last_op = opcodes.back();
    const bool islastITP = last_op._op == OP_ITP;

    load_flags();

    for (opcode &op : opcodes)
    {

        const u8 isize = 2;

        int32_t regs[]{op.rs1, op.rs2, op.rd};

        if ((op._op == OP_ITP || op._op == OP_SWI))
        {
            regman.flush_all();
            regman.reset(true);
        }
        /*else
        regman.push_sp_reg(3, regs);*/

        // reg_alloc.alloc_regs(op);

        if (last_op.op_pc != op.op_pc)
            emit_prefetch(isize, op.rs1 == 15 || op.rs2 == 15, op._op == OP_ITP);
        else
            // Last op has always to save
            emit_prefetch(isize, true, true);

        emitThumbOP<PROCNUM>(op);
    }

    if (flag_dirty)
        store_flags();
    flag_dirty = false;

    emit_prefetch_reset();
    regman.flush_all();
    regman.reset(true);

    // regman.pull_sp_regs();

    // possible idle loop, do more checks here
    return idleLoop;
}

static bool instr_is_conditional(u32 opcode)
{
    return !(CONDITION(opcode) == 0xE || (CONDITION(opcode) == 0xF && CODE(opcode) == 5));
}

template <int PROCNUM>
bool block::emitArmBlock()
{

    opcode last_op = opcodes.back();

    intr_instr = 0;

    use_flags = uses_flags;
    // flag_loaded / flag_dirty are file-scope globals that survive between block
    // compilations. load_flags() only sets flag_loaded when use_flags is true, so a
    // block that does not use flags left the PREVIOUS block's flag_loaded=true
    // standing while psp_gp held nothing -- and consumers (the RRX carry fast path,
    // check_flags) then trusted psp_gp. Reset both at block entry.
    flag_loaded = false;
    flag_dirty  = false;

    /*char * found = strstr(compiled_functions_hash, block_hash);

    if (found != NULL) {
        return arm_compiledOP[(found - compiled_functions_hash) / 51](PROCNUM);
    }*/
    //printf("JIT %s at %08X, %08X\n", PROCNUM?"ARM7":"ARM9", emit_getCurrAdr(), opcodes.front().op_pc);
    StartCodeDump();

    start_block = emit_getCurrAdr();
    emit_li(psp_fp, opcodes.front().op_pc);

    load_flags();

    regman.reset();

    const u8 isize = 4;
    for (opcode &op : opcodes)
    {
        if ((op._op == OP_ITP || op._op == OP_SWI))
        {
            // Spill only what this interpreted instruction touches, instead of
            // the whole cache. itp_reg_mask() returns all-ones for anything it
            // cannot classify, which reproduces the old flush-everything path.
            // OP_ITP keeps the raw opcode in rs1.
            ItpRegMask m = (op._op == OP_ITP && g_itp_selective_flush)
                               ? itp_reg_mask((u32)op.rs1)
                               : ItpRegMask{0xFFFFu, 0xFFFFu};

            if (m.read == 0xFFFFu && m.write == 0xFFFFu)
            {
                regman.flush_all();
                regman.reset();
            }
            else
            {
                regman.flush_selective(m.read, m.write);
            }
        }

        if (last_op.op_pc != op.op_pc)
            emit_prefetch(isize, op.rs1 == 15 || op.rs2 == 15, op._op == OP_ITP);
        else
        {
            // Last op has always to save
            emit_prefetch(isize, true, true);
        }

        emitARMOP<PROCNUM>(op, last_op.op_pc == op.op_pc);
    }

    check_flags()

    emit_prefetch_reset();
    regman.flush_all();
    regman.reset();

    /*if (!JumpOP)
    {
        ArmOpCompiled code_block = (ArmOpCompiled)JIT_COMPILED_FUNC(last_op.op_pc + 4, ARM9);

        if (code_block)
        {
            // printf("Fast jump\n");
            emit_jal(code_block);
            emit_nop();
            JumpOP = true;
        }
    }*/

    // regman.pull_sp_regs();

    /*if (opcodes.size() > 30)
        CodeDump("30_ops.bin");
   */

    // possible idle loop, do more checks here
    return idleLoop;
}

template <int PROCNUM>
void block::emitArmBranch()
{
    uint32_t conditional_label = 0;
    uint32_t jump_sz = 0;

    opcode op = opcodes.back();

    emit_li(psp_fp, op.op_pc + 4);
    emit_sw(psp_fp, psp_k0, _next_instr);

    emit_addiu(psp_at, psp_fp, 4);
    emit_sw(psp_at, psp_k0, _R15);

    // conditional(emitARMOP<PROCNUM>(op))
}

// define the template
template bool block::emitArmBlock<0>();
template bool block::emitArmBlock<1>();

template void block::emitArmBranch<0>();
template void block::emitArmBranch<1>();
