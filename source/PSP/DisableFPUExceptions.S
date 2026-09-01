
	.global _DisableFPUExceptions

    .set push 
    .set noreorder 
_DisableFPUExceptions: 
    cfc1    $2, $31 
    lui     $8, 0x80 
    and     $8, $2, $8     # Mask off all bits except for 23 of FCR 
    ctc1    $8, $31 
    jr      $31 
    nop 
    .set pop 


# NOTE: set_op_logic_flags / set_sub_flags / set_and_flags used to be appended
# here. They are the JIT's flag helpers and belong in pspOpUtil.S, which now
# provides them (plus set_add_flags and asm_fast_wrt32). The copies here were an
# older revision -- set_sub_flags in particular computed C and V differently --
# and would have silently won at link time over the versions the ported JIT was
# difftested against.
