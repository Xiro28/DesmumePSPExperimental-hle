#include "types.h"

#include "arm_jit.h"
#include "instructions.h"
#include "instruction_attributes.h"
#include "armcpu.h"

#include <list>

#define _REG_NUM(i, n)		((i>>(n))&0x7)

// Hold the block-chain cycle accumulator in a callee-saved register ($s3) across
// the whole chain instead of round-tripping executed_cycles through memory each
// block. Costs one SRA guest-cache slot (s3 removed from SRA_REGS). Toggle here;
// keep AOT_VERSION in sync (the emitted dispatch code changes).
#define JIT_CYCLES_IN_REG 1

// JIT optimization feature flags — comment out to disable an optimization.
#define JIT_OPT_MULACC       // MLA→madd, UMULL→multu+mflo/mfhi, UMLAL→maddu (64-bit accurate)
#define JIT_OPT_RRX_GP       // RRX: skip lbu CPSR reload when carry is already in psp_gp
#define JIT_OPT_SATURATE     // QADD/QSUB via branchless saturation (was always INTERPRET)
#define JIT_OPT_SIGNEDMUL16  // SMULxy/SMLAxy via seh/sra+mult/madd (was always INTERPRET)
#define JIT_OPT_PREDICATE_MUL // MUL/MLA conditional path always via movz/movn (no branch)

// Profiling: count block-transition fast-hits vs compiles ([DISP]/[EMIT] prints).
// Disabled for normal builds; uncomment to profile block dispatch again.
// #define JIT_PROFILE_DISPATCH

#define RCPU   psp_k0
#if JIT_CYCLES_IN_REG
#define RCYC   psp_s3
#else
#define RCYC   psp_s0
#endif

// Which CPU the block currently being compiled belongs to. This was `static`,
// which in a header gives every translation unit its OWN copy: arm_jit.cpp set
// its copy in arm_jit_compile(), while blockdecoder.cpp read a different one
// that stayed 0 forever. Harmless while the JIT was ARM9-only (0 was always
// right); with an ARM7 JIT it silently compiled ARM7 blocks against ARM9 state.
extern uint32_t block_procnum;

#define _ARMPROC (block_procnum ? NDS_ARM7 : NDS_ARM9)

#define _cond_table(x) arm_cond_table[x]

#define _cp15(x) ((u32)(((u8*)&cp15.x) - ((u8*)&_ARMPROC)))
#define _MMU(x) ((u32)(((u8*)&MMU.x) - ((u8*)&_ARMPROC)))
#define _NDS_ARM9(x) ((u32)(((u8*)&NDS_ARM9.x) - ((u8*)&_ARMPROC)))
#define _NDS_ARM7(x) ((u32)(((u8*)&NDS_ARM7.x) - ((u8*)&_ARMPROC)))

#define _reg(x) ((u32)(((u8*)&_ARMPROC.R[x]) - ((u8*)&_ARMPROC)))
#define _reg_pos(x) ((u32)(((u8*)&_ARMPROC.R[REG_POS(i,x)]) - ((u8*)&_ARMPROC)))
#define _thumb_reg_pos(x) ((u32)(((u8*)&_ARMPROC.R[_REG_NUM(i,x)]) - ((u8*)&_ARMPROC)))

#define _R15 _reg(15)

#define _instr_adr ((u32)(((u8*)&_ARMPROC.instruct_adr) - ((u8*)&_ARMPROC)))
#define _next_instr ((u32)(((u8*)&_ARMPROC.next_instruction) - ((u8*)&_ARMPROC)))
#define _instr ((u32)(((u8*)&_ARMPROC.instruction) - ((u8*)&_ARMPROC)))

#define main_mem ((u32)(((u8*)&MMU.MAIN_MEM[0]) - ((u8*)&MMU)))
#define dtcm_mem ((u32)(((u8*)&MMU.ARM9_DTCM[0]) - ((u8*)&MMU)))
#define dtcm_addr ((u32)(((u8*)&MMU.DTCMRegion) - ((u8*)&MMU)))


#define dtcm_mem2 ((u32)(((u8*)&MMU.ARM9_DTCM[0]) - ((u8*)&MMU.MAIN_MEM[0])))
#define dtcm_addr2 ((u32)(((u8*)&MMU.DTCMRegion) - ((u8*)&MMU.MAIN_MEM[0])))


#define _flags ((u32)(((u8*)&_ARMPROC.CPSR.val) - ((u8*)&_ARMPROC)))
#define _flag_N 31
#define _flag_Z 30
#define _flag_C 29
#define _flag_V 28
#define _flag_T  5

//LBU 
#define _flag_N8 7
#define _flag_Z8 6
#define _flag_C8 5
#define _flag_V8 4

#define conditional(x) \
    if (op.condition != ((uint32_t)-1)) {\
            conditional_label = emit_Halfbranch(op.condition, op.check_condition);\
            jump_sz = emit_getCurrAdr();\
        }\
        {x;}\        
        if (conditional_label != 0 && op.condition != ((uint32_t)-1)){\
            jump_sz = emit_getCurrAdr() - jump_sz;\
            CompleteCondition(op.condition, conditional_label, emit_getCurrAdr() + jump_sz + 8);\
        }

#define conditional_branchless(dst, tmp_dst, x) \
    if (op.condition != ((uint32_t)-1) && op.check_condition) generate_condition_check(op.condition);\
    /*else if (!op.check_condition) printf("No condition 0x%x\n", emit_getCurrAdr());*/\
    {x;} \
    if (op.condition != ((uint32_t)-1)){\
        const bool isMovz = (op.condition < 8) ? (op.condition & 1) : !(op.condition & 1);\
        if (isMovz) emit_movz(dst, tmp_dst, psp_s4); \
        else emit_movn(dst, tmp_dst, psp_s4); \
    }

// Like conditional_branchless, but ALSO commits the flag byte ($28 / psp_gp)
// conditionally. Flag-setting S-ops (ADD_S/SUB_S/AND_S/EOR_S/ORR_S) must leave
// NZCV untouched when their condition is false. conditional_branchless only
// guards the destination register; the body here writes flags unconditionally,
// so we snapshot the old flags into flag_save before the body and restore them
// when the condition is false, reusing the psp_s4 predicate. flag_save must be a
// register not clobbered by the body (set_*_flags use t0..t2; pass e.g. psp_t3).
#define conditional_branchless_flags(dst, tmp_dst, flag_save, x) \
    if (op.condition != ((uint32_t)-1)) {\
        if (op.check_condition) generate_condition_check(op.condition);\
        emit_move(flag_save, psp_gp);\
    }\
    {x;} \
    if (op.condition != ((uint32_t)-1)){\
        const bool isMovz = (op.condition < 8) ? (op.condition & 1) : !(op.condition & 1);\
        if (isMovz) { emit_movz(dst, tmp_dst, psp_s4); emit_movn(psp_gp, flag_save, psp_s4); }\
        else        { emit_movn(dst, tmp_dst, psp_s4); emit_movz(psp_gp, flag_save, psp_s4); }\
    }

enum op{
    
    // Special custom opcodes
    OP_NOP = 0,
    OP_BC,
    OP_BLC,
    OP_BXC,
    OP_BXRC,
    OP_DMA,
    OP_3D_FIFO,
    OP_3D_CMD,
    OP_HALT_HACK,



    OP_ITP,
    OP_AND,
    OP_ORR,
    OP_BIC,
    OP_EOR,
    OP_ADD,
    OP_SUB,
    OP_MUL,
    OP_UMUL,
    OP_MLA,
    OP_UMLA,
    OP_CLZ,
    OP_QADD,    // SAT_32(Rm + Rn); op.rd=Rd, op.rs1=Rm, op.rs2=Rn
    OP_QSUB,    // SAT_32(Rm - Rn); op.rd=Rd, op.rs1=Rm, op.rs2=Rn
    OP_SMULXY,  // sign16(Rm.x)*sign16(Rs.y); op.imm: bit0=Rm_top bit1=Rs_top
    OP_SMLAXY,  // Rd=(Rm.x*Rs.y)+Rn; op.imm: [3:0]=Rn_reg [4]=Rm_top [5]=Rs_top
    OP_RSB,
    OP_ADC,
    OP_SBC,
    OP_RSC,
    OP_NEG,
    OP_MOV,
    OP_MVN,

    OP_LDR = 128,
    OP_STR,
    OP_LDRH,
    OP_STRH,
    OP_LDRSH,
    OP_LDRSB,
    OP_STMIA,
    OP_LDRB,
    OP_STRB,
    OP_LDM,
    OP_STM,

    OP_SWI,
    OP_LSR_0,

    OP_CMP = 512,
    OP_CMN,
    OP_TST,
    OP_AND_S,
    OP_EOR_S,
    OP_ORR_S,
    OP_ADD_S,
    OP_SUB_S,
    OP_MOV_S,
    OP_MVN_S
};

enum opType{
    PRE_OP_NONE = -1,

    PRE_OP_REG = 0,
    PRE_OP_IMM = 1,

    PRE_OP_LSL = 2,
    PRE_OP_LSR = 4,
    PRE_OP_ASR = 8,
    PRE_OP_ROR = 16,

    PRE_OP_PRE_P  = 32,
    PRE_OP_POST_P = 64,
    PRE_OP_PRE_M  = 128,
    PRE_OP_POST_M = 256,

    PRE_OP_LSL_IMM = (PRE_OP_LSL | PRE_OP_IMM),
    PRE_OP_LSR_IMM = (PRE_OP_LSR | PRE_OP_IMM),
    PRE_OP_ASR_IMM = (PRE_OP_ASR | PRE_OP_IMM),
    PRE_OP_ROR_IMM = (PRE_OP_ROR | PRE_OP_IMM),

    PRE_OP_LSL_REG = (PRE_OP_LSL | PRE_OP_REG),
    PRE_OP_LSR_REG = (PRE_OP_LSR | PRE_OP_REG),
    PRE_OP_ASR_REG = (PRE_OP_ASR | PRE_OP_REG),
    PRE_OP_ROR_REG = (PRE_OP_ROR | PRE_OP_REG),

    PRE_OP_IMM_PRE_P  = (PRE_OP_PRE_P  | PRE_OP_IMM),
    PRE_OP_IMM_POST_P = (PRE_OP_POST_P | PRE_OP_IMM),
    PRE_OP_IMM_PRE_M  = (PRE_OP_PRE_M  | PRE_OP_IMM),
    PRE_OP_IMM_POST_M = (PRE_OP_POST_M | PRE_OP_IMM),  

    PRE_OP_REG_PRE_P  = (PRE_OP_PRE_P  | PRE_OP_REG),
    PRE_OP_REG_POST_P = (PRE_OP_POST_P | PRE_OP_REG),
    PRE_OP_REG_PRE_M  = (PRE_OP_PRE_M  | PRE_OP_REG),
    PRE_OP_REG_POST_M = (PRE_OP_POST_M | PRE_OP_REG),        
};

enum extraFlags {
    EXTFL_NONE = 1,
    EXTFL_MERGECOND = 2,
    EXTFL_SAVECOND = 4,
    EXTFL_SKIPLOADFLAG = 8,
    EXTFL_SKIPSAVEFLAG = 16,
    EXTFL_DIRECTMEMACCESS = 32,
    EXTFL_3DCOM = 64,
    EXTFL_NOFLAGS = 128
};


struct opcode{
    uint32_t bytes;
    uint32_t rd;
    uint32_t rs1;
    uint32_t rs2;
    int32_t imm;
    uint32_t op_pc;
    uint32_t condition;
    uint32_t extra_flags;

    bool check_condition = true;
    bool dead_rd = false;   // set by liveness pass: rd is overwritten before any read

     // THUMB SPECIFIC
    bool saveN = true;
    bool saveZ = true;
    bool saveC = true;
    bool saveV = true;

    op _op;
    opType preOpType;

    opcode(op opcode, uint32_t bytes, uint32_t rd, uint32_t rs1, uint32_t rs2, uint32_t imm, opType preOpType, uint32_t pc, uint32_t condition, uint32_t extra_flags = EXTFL_NONE){
        this->_op = opcode;
        this->bytes = bytes;
        this->rd = rd;
        this->rs1 = rs1;
        this->rs2 = rs2;
        this->imm = imm;
        this->preOpType = preOpType;
        this->op_pc = pc;
        this->condition = condition;
        this->extra_flags = extra_flags|EXTFL_SAVECOND;
    }
};

class block{
    public:

        void init(){
            clearBlock();
            printf("block created\n");
        }

        bool isIdleLoop(bool thumb) {
            // see https://github.com/dolphin-emu/dolphin/blob/master/Source/Core/Core/PowerPC/PPCAnalyst.cpp#L678
            // it basically checks if one iteration of a loop depends on another
            // the rules are quite simple

            if (opcodes.size() > 4) return false;

            // Only safe to skip if IRQs are enabled — otherwise the loop can never exit via interrupt.
            if (_ARMPROC.CPSR.bits.I) return false;

            // NOTE: this whole function is currently unreachable -- isIdle is
            // hardcoded false in compile_basicblock (see the note there).
            // Checking IME/IE here was not enough: the flag is frozen into the
            // block at compile time, so it goes stale as soon as they change.

            // If there are any write or READ operations, the block isn’t idle.
            // A ReadOP means the loop polls memory (e.g. GXSTAT, VCOUNT) — skipping
            // it would stall the hardware register it’s waiting on.
            if (WriteOP || ReadOP || (!thumb && MCROP))
                return false;
        
            // Use a 32-bit bitfield to track registers.
            uint32_t regsWrittenTo = 0;          // Registers that have been written.
            uint32_t regsDisallowedToWrite = 0;  // Registers that have been used as a source before being written.
        
            // Helper lambda to process a source register.
            auto processSource = [&](int reg) {
                // Skip invalid register (assumed to be -1) or out-of-range registers.
                if (reg < 0 || reg >= 32)
                    return;
                uint32_t mask = 1u << reg;
                // Only mark as disallowed if this register hasn't been written yet.
                if (!(regsWrittenTo & mask))
                    regsDisallowedToWrite |= mask;
            };
        
            // Process each opcode in the basic block.
            for (const auto& op : opcodes) {
                // If an OP_ITP opcode is encountered, consider the block non-idle.
                if (op._op == OP_ITP)
                    return false;
        
                // Process both source registers (rs1 and rs2).
                processSource(op.rs1);
                processSource(op.rs2);
        
                // Process destination register (rd).
                if (op.rd >= 0 && op.rd < 32) {
                    uint32_t mask = 1u << op.rd;
                    // If the register being written to has been used as a source (and not yet overwritten),
                    // then the block isn't idle.
                    if (regsDisallowedToWrite & mask)
                        return false;
                    // Otherwise, mark this register as written.
                    regsWrittenTo |= mask;
                }
            }
            return true;
        }
        


        void addOP(op _op, uint32_t bytes, uint32_t pc, uint32 rd = -1, uint32 rs1 = -1, uint32 rs2 = -1, uint32_t imm = -1, opType preOpType = PRE_OP_NONE, uint32_t condition = -1, uint32_t extra_flags = EXTFL_SAVECOND){
            
            if (_op >= OP_CMP || condition != -1) uses_flags = true;

            opcodes.push_back(opcode(_op, bytes, rd, rs1, rs2, imm, preOpType, pc, condition, extra_flags));

            containsITP = containsITP || (_op == OP_ITP);

            if (_op == OP_ITP) return;

            onlyITP = false;
        }

        void clearBlock(){

#if defined(JIT_BUILD_HASH) || defined(save_hash)
            block_hash[0] = '\0';
#endif

            idleLoop = false;
            onlyITP = true;
            uses_flags = false;

            WriteOP = false;
            ReadOP = false;
            JumpOP = false;
            MCROP = false;
            jumped = false;

            manualPrefetch = false;
            containsITP = false;
            early_exit_compiled_target = 0;
            branch_link_target_pc = 0;
            branch_cond_link = false;
            branch_cond_fallthrough_pc = 0;

            branch_addr = 0;
            start_addr = 0;

            if (opcodes.size() > 0) opcodes.clear();
        }

        u32 getNOpcodes() { return opcodes.size(); }

        template<int PROCNUM>
        bool emitArmBlock();

        template<int PROCNUM>
        bool emitThumbBlock();

        template<int PROCNUM>
        void emitArmBranch();    

        void optimize_basicblock();
        void optimize_basicblockThumb();

        bool WriteOP = false;
        bool ReadOP = false;
        bool JumpOP = false;
        bool MCROP = false;
        bool uses_flags = false;
        bool manualPrefetch = false;
        bool onlyITP = true;
        bool containsITP = false;
        bool idleLoop = false;
        bool jumped = false;

        // Early-termination: set during block building when the current PC already
        // has a compiled block. The epilogue jumps directly to that block instead of
        // re-emitting its instructions. Zero means no early exit.
        uintptr_t early_exit_compiled_target = 0;

        // branch-edge-link: static guest target PC of an UNCONDITIONAL terminal B/BL.
        // The branch is always taken, so instruct_adr == this target at the epilogue,
        // letting the inline dispatch jump straight to the target's compiled block and
        // skip the JIT_MEM 2-level lookup (99.9% of transitions hit a compiled block).
        // Only set for unconditional B/BL: conditional branches may fall through, so
        // instruct_adr is not statically known and they must keep the full dispatcher.
        // Zero means no static branch link.
        u32 branch_link_target_pc = 0;

        // Set for a CONDITIONAL terminal B: the epilogue must pick between the taken
        // target (branch_link_target_pc) and the not-taken fall-through (pc, the next
        // block) by comparing instruct_adr at runtime — both slots are compile-time
        // constants. branch_cond_fallthrough_pc holds the not-taken PC (= block end).
        bool branch_cond_link = false;
        u32  branch_cond_fallthrough_pc = 0;

        u32 branch_addr = 0;
        u32 start_addr = 0;

#if defined(JIT_BUILD_HASH) || defined(save_hash)
        char block_hash[1024];
#endif
        std::vector<opcode> opcodes;

    private:
        uint32 startAddr;
        uint32 endAddr;
        uint32 getStartAddr();
        uint32 getEndAddr();
        void addOpcode(opcode op);
        void printBlock();
};

void emit_prefetch(const u8 isize, bool saveR15, bool is_ITP);

extern block currentBlock;

#define loadThumbReg(psp_reg, nds_reg) emit_lw(psp_reg, RCPU, _reg(nds_reg))
#define storeThumbReg(psp_reg, nds_reg) emit_sw(psp_reg, RCPU, _reg(nds_reg))

#define storeHalfReg(psp_reg, nds_reg) emit_sh(psp_reg, RCPU, _reg(nds_reg))