.global  set_op_logic_flags 		
.set noat 
.set noreorder

set_op_logic_flags:
    # N
    SRL  $8, $2, 31
    INS  $28, $8, 7, 1  

    # Z 
    SLTIU $8, $2, 1
    JR   $31           
    INS  $28, $8, 6, 1 
  

.global  set_sub_flags 		
.set noat  
.set noreorder

set_sub_flags: 

    # $28 = flags (4 = V, 5 = C, 6 = Z, 7 = N)
    # Corretto controllato con emulatore arm

    # C
    SLTU $t0, $a0, $a1    
    XORI $t0, $t0, 1    
    INS  $28, $t0, 5, 1   # 

    # Z 
    SLTIU $t0, $v0, 1
    INS  $28, $t0, 6, 1 

    # N
    SRL  $t1, $v0, 31
    INS  $28, $t1, 7, 1   

    # V           
    SLT $t0, $4, $5
    XOR  $t0, $t0, $t1
    JR   $31           
    INS  $28, $t0, 4, 1   


.global  set_add_flags
.set noat
.set noreorder

set_add_flags:
    # a0=left(rn), a1=right(op2), v0=alu_out
    # Matches interpreter OverflowFromADD: b1=(left^out)>>31, b2=!(left^right) [left==right]

    # C: NOT(left) < right  (= carry out)
    NOR  $t2, $a0, $zero
    SLTU $t0, $t2, $a1
    INS  $28, $t0, 5, 1

    # Z
    SLTIU $t0, $v0, 1
    INS  $28, $t0, 6, 1

    # N
    SRL  $t1, $v0, 31
    INS  $28, $t1, 7, 1

    # V: matches the interpreter's OverflowFromADD (armcpu.h):
    #   (sign(left) == sign(right)) && (sign(alu_out) != sign(left))
    # The previous code used b2 = !(left ^ right), i.e. left == right as whole
    # WORDS, and claimed in a comment that this was what the interpreter did. It
    # is not -- the interpreter compares SIGN BITS. That made V wrong whenever
    # the operands merely shared a sign, e.g. ADDS 7FB3FE6C + 01000000: V must be
    # 1 (positive+positive giving a negative result) but the word compare gave 0.
    # Only visible once JIT_ADD_S_IMM was turned on; before that ADDS went to the
    # interpreter and difftest was comparing the interpreter against itself.
    # V: matches the interpreter's OverflowFromADD (armcpu.h):
    #   (sign(left) == sign(right)) && (sign(alu_out) != sign(left))
    # The previous code compared left==right as whole WORDS, which is a different
    # condition; difftest caught it as ADDS imm 12/2 and CMNS failing on most
    # shift forms, all of which go to 0 with the sign-bit version below.
    # (Bisected against the Sims 3 boot failure and cleared: reverting this does
    # not bring the boot back, so it is unrelated to that.)
    XOR  $t0, $a0, $a1        # sign(left) == sign(right) ?
    SRL  $t0, $t0, 31
    XORI $t2, $t0, 1          # t2 = signs equal
    XOR  $t0, $a0, $v0        # sign(alu_out) != sign(left) ?
    SRL  $t1, $t0, 31
    AND  $t0, $t1, $t2
    JR   $31
    INS  $28, $t0, 4, 1

.global  set_and_flags
.set noat
.set noreorder

set_and_flags:
    SRL  $8, $2, 31
    INS  $28, $8, 7, 1

    SLTIU $8, $2, 1
    INS  $28, $8, 6, 1
     
    SLT $8, $4, $5
    JR $31
    INS  $28, $8, 4, 1


.global  asm_fast_wrt32 		
.set noat 
.set noreorder

asm_fast_wrt32:
    SRL  $8, $2, 31
    INS  $28, $8, 7, 1

    SLTIU $8, $2, 1
    INS  $28, $8, 6, 1
     
    SLT $8, $4, $5
    JR $31
    INS  $28, $8, 4, 1


#include "pspstub.s"
	STUB_FUNC  0x60A41671,usrDebugProfilerEnable
    STUB_FUNC  0xDA92A4C9,usrDebugProfilerDisable
    STUB_FUNC  0x42919B17,usrDebugProfilerClear
    STUB_FUNC  0xCC65DB8D,usrDebugProfilerGetRegs
	STUB_END
