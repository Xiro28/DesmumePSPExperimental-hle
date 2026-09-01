template<int PROCNUM> 
void emitThumbOP(opcode& op){
    switch(op._op) {
        case OP_ITP:
        {
           if (flag_dirty) store_flags();

           flag_dirty = false;

            emit_jal(thumb_instructions_set[ARM9][op.rs1 >> 6]);

            emit_movi(psp_a0,op.rs1&0xFFFF); 

            load_flags();

            intr_instr++;
        }
        break;

        case OP_BC:
        {
            regman.flush_all(false);
            regman.reset();
            check_flags()
            emit_movi(psp_v0, 1);
            conditional(
                emit_li(psp_a0, op.imm);
                u32 _op = emit_SlideDelay();
                emit_jal(jump_to_linked_bc<true>);
                emit_Write32(_op);)
        }
        break;

        case OP_BXC:
        {
            if (flag_dirty) store_flags();

            flag_dirty = false;

            regman.flush_all();
            regman.reset();

            currentBlock.JumpOP = true;

            emit_li(psp_a0, op.imm);
            emit_jal(jump_to_linked_uncod_thumb);
            emit_nop();
        }
        break;

        case OP_LSR_0:
        {
            //printf("0x%x\n", emit_getCurrAdr());
            /*psp_gpr_t dst = reg_alloc.getReg(op.rd, psp_v0, false);
            psp_gpr_t rs1 = reg_alloc.getReg(op.rs1, psp_a0);

            if (dst == psp_v0) storeReg(psp_zero, op.rd); else emit_move(dst, psp_zero);

            emit_sra(psp_v0, rs1, 31);
            emit_sll(psp_v0, psp_v0, 2);
            emit_ori(psp_gp, psp_gp, 2);
            emit_ins(psp_gp, psp_v0, _flag_C8, 3);

            flag_dirty = true;*/
        }
        break;

        //TODO ADD reg_alloc.end at the end of the opcde to store the temp reg (in case the static one are full)
        case OP_AND:
        {
            int32_t regs[3] = {op.rd, op.rs1, op.rs2};

            regman.get(3, regs);

            const psp_gpr_t dst = (psp_gpr_t)regs[0]; 
            const psp_gpr_t rs1 = (psp_gpr_t)regs[1]; 
            const psp_gpr_t rs2 = (psp_gpr_t)regs[2]; 

            emit_and(dst, rs1, rs2);

            if (op.saveN){
                emit_srl(psp_t0, dst, 31);
                emit_ins(psp_gp, psp_t0, _flag_N8, _flag_N8);
            }

            if (op.saveZ){
                emit_sltiu(psp_t0, dst, 1);
                emit_ins(psp_gp, psp_t0, _flag_Z8, _flag_Z8);
            }

            regman.mark_dirty(dst);

            flag_dirty = flag_dirty || (op.saveZ || op.saveN || op.saveC || op.saveV);
        }
        break; 

        //TODO! check flag for sub/cmp - not all cases are correct!!!! for now seems to work but this must be 
        //                           one of the first thing to be analyzed if some games behaves wrongly!!
        case OP_SUB:
        { 
            int32_t regs[3] = {op.rd, op.rs1, op.rs2};

            regman.get(3, regs);
            const psp_gpr_t dst = (psp_gpr_t)regs[0]; 
            const psp_gpr_t rs1 = (psp_gpr_t)regs[1]; 
            const psp_gpr_t rs2 = (psp_gpr_t)regs[2]; 

            if (op.preOpType == PRE_OP_IMM){
                emit_subiu(dst, rs1, op.imm);
            }else{
                emit_subu(dst, rs1, rs2);
            }

            if (!(op.extra_flags & EXTFL_NOFLAGS)){
                if (op.saveN){
                    emit_srl(psp_t1, dst, 31);
                    emit_ins(psp_gp, psp_t1, _flag_N8, _flag_N8);
                }

                if (op.saveZ){
                    emit_sltiu(psp_t0, dst, 1);
                    emit_ins(psp_gp, psp_t0, _flag_Z8, _flag_Z8);
                }

                if (op.saveC){
                    emit_sltu(psp_t0, rs1, rs2);
                    emit_xori(psp_t0, psp_t0, 1);
                    emit_ins(psp_gp, psp_t0, _flag_C8, _flag_C8);
                }

                if (op.saveV){
                    if (!op.saveN) emit_srl(psp_t1, psp_v0, 31);

                    emit_slt(psp_t2, rs1, rs2);
                    emit_xor(psp_t2, psp_t2, psp_t1);
                    emit_ins(psp_gp, psp_t2, _flag_V8, _flag_V8);
                }
                flag_dirty = flag_dirty || (op.saveZ || op.saveN || op.saveC || op.saveV);
            }

            regman.mark_dirty(dst);

        }
        break;

        case OP_ADD:
        {
            int32_t regs[3] = {op.rd, op.rs1, op.rs2};

            regman.get(3, regs);
            const psp_gpr_t dst = (psp_gpr_t)regs[0]; 
            const psp_gpr_t rs1 = (psp_gpr_t)regs[1]; 
            const psp_gpr_t rs2 = (psp_gpr_t)regs[2]; 

            if (op.preOpType == PRE_OP_IMM){
                emit_addiu(dst, rs1, op.imm);
            }else{
                emit_addu(dst, rs1, rs2);
            }

            if (!(op.extra_flags & EXTFL_NOFLAGS)){
                
                if (op.saveN){
                    emit_srl(psp_t0, dst, 31);
                    emit_ins(psp_gp, psp_t0, _flag_N8, _flag_N8);
                }

                if (op.saveZ){
                    emit_sltiu(psp_t0, dst, 1);
                    emit_ins(psp_gp, psp_t0, _flag_Z8, _flag_Z8);
                }

                 if (op.saveC){
                    emit_sltu(psp_t0, dst, rs1);
                    emit_ins(psp_gp, psp_t0, _flag_C8, _flag_C8);
                }

                if (op.saveV){
                    if (op.preOpType == PRE_OP_IMM){
                        emit_li(psp_t1, op.imm);
                        emit_xor(psp_t2, rs1, psp_t1);  // rs1 ^ imm
                    } else {
                        emit_xor(psp_t2, rs1, rs2);      // rs1 ^ rs2
                    }
                    emit_xor(psp_t3, rs1, dst);          // rs1 ^ dst
                    emit_nor(psp_t2, psp_t2, psp_zero);  // ~(rs1^rs2)
                    emit_and(psp_t2, psp_t3, psp_t2);
                    emit_srl(psp_t2, psp_t2, 31);
                    emit_ins(psp_gp, psp_t2, _flag_V8, _flag_V8);
                }

                flag_dirty = flag_dirty || (op.saveZ || op.saveN || op.saveC || op.saveV);   
            }

            regman.mark_dirty(dst);
        }
        break;

        case OP_EOR:
        {
            int32_t regs[3] = {op.rd, op.rs1, op.rs2};

            regman.get(3, regs);

            const psp_gpr_t dst = (psp_gpr_t)regs[0]; 
            const psp_gpr_t rs1 = (psp_gpr_t)regs[1]; 
            const psp_gpr_t rs2 = (psp_gpr_t)regs[2]; 

            emit_xor(dst, rs1, rs2);
            
            if (op.saveN){
                emit_srl(psp_t0, dst, 31);
                emit_ins(psp_gp, psp_t0, _flag_N8, _flag_N8);
            }

            if (op.saveZ){
                emit_sltiu(psp_t0, dst, 1);
                emit_ins(psp_gp, psp_t0, _flag_Z8, _flag_Z8);
            }

            regman.mark_dirty(dst); 

            flag_dirty = flag_dirty || (op.saveZ || op.saveN);            
        }
        break;

        case OP_ORR:
        {
            int32_t regs[3] = {op.rd, op.rs1, op.rs2};

            regman.get(3, regs);
            const psp_gpr_t dst = (psp_gpr_t)regs[0]; 
            const psp_gpr_t rs1 = (psp_gpr_t)regs[1]; 
            const psp_gpr_t rs2 = (psp_gpr_t)regs[2]; 

            emit_or(dst, rs1, rs2);
            
            if (op.saveN){
                emit_srl(psp_t0, dst, 31);
                emit_ins(psp_gp, psp_t0, _flag_N8, _flag_N8);
            }

            if (op.saveZ){
                emit_sltiu(psp_t0, dst, 1);
                emit_ins(psp_gp, psp_t0, _flag_Z8, _flag_Z8);
            }

            regman.mark_dirty(dst);

            flag_dirty = flag_dirty || (op.saveZ || op.saveN);   
        }
        break;

        case OP_TST:
        {
            int32_t regs[2] = {op.rs1, op.rs2};

            regman.get(2, regs);
            const psp_gpr_t rs1 = (psp_gpr_t)regs[0]; 
            const psp_gpr_t rs2 = (psp_gpr_t)regs[1];

            emit_and(psp_t1, rs1, rs2);

            if (op.saveN){
                emit_srl(psp_t0, psp_t1, 31);
                emit_ins(psp_gp, psp_t0, _flag_N8, _flag_N8);
            }

            if (op.saveZ){
                emit_sltiu(psp_t0, psp_t1, 1);
                emit_ins(psp_gp, psp_t0, _flag_Z8, _flag_Z8);
            } 

            flag_dirty = flag_dirty || (op.saveZ || op.saveN);          
        }
        break;

        case OP_SWI:
        {
            //uint32_t optmizeDelaySlot = emit_SlideDelay();

            emit_jal(cpu->swi_tab[op.rs1]);
            emit_nop(); // safety
            //emit_Write32(optmizeDelaySlot);
        }
        break;

        case OP_CMP:
        {
            int32_t regs[2] = {op.rs1, op.rs2};

            regman.get(2, regs);
            const psp_gpr_t rs1 = (psp_gpr_t)regs[0]; 
            psp_gpr_t rs2 = (psp_gpr_t)regs[1];

            if (op.preOpType == PRE_OP_IMM){
                rs2 = psp_a1;
                emit_li(rs2, op.imm);
            }
            emit_subu(psp_v0, rs1, rs2);

           /* emit_move(psp_a0, rs1);
            emit_move(psp_a1, rs2);
            

            emit_jal(set_sub_flags); 
            emit_subu(psp_v0, psp_a0, psp_a1);  */

            if (op.saveN){
                emit_srl(psp_t1, psp_v0, 31);
                emit_ins(psp_gp, psp_t1, _flag_N8, _flag_N8);
            }
            
            if (op.saveZ){
                emit_sltiu(psp_t3, psp_v0, 1);
                emit_ins(psp_gp, psp_t3, _flag_Z8, _flag_Z8);
            }

            if (op.saveC){
                emit_sltu(psp_t0, rs1, rs2);
                emit_xori(psp_t0, psp_t0, 1);
                emit_ins(psp_gp, psp_t0, _flag_C8, _flag_C8);
            }

            // V
            if (op.saveV){
                if (!op.saveN) emit_srl(psp_t1, psp_v0, 31);

                emit_slt(psp_t2, rs1, rs2);
                emit_xor(psp_t2, psp_t2, psp_t1);
                emit_ins(psp_gp, psp_t2, _flag_V8, _flag_V8);
            }


            flag_dirty = flag_dirty || (op.saveZ || op.saveN || op.saveC || op.saveV);      
        }
        break;
 
        case OP_MUL:
        {
            int32_t regs[3] = {op.rd, op.rs1, op.rs2};

            regman.get(3, regs);
            const psp_gpr_t dst = (psp_gpr_t)regs[0]; 
            const psp_gpr_t rs1 = (psp_gpr_t)regs[1];
            const psp_gpr_t rs2 = (psp_gpr_t)regs[2];

            emit_mult(rs1, rs2);
            emit_mflo(dst);

            regman.mark_dirty(dst);
            
            if (op.saveN){
                emit_srl(psp_t0, dst, 31);
                emit_ins(psp_gp, psp_t0, _flag_N8, _flag_N8);
            }

            if (op.saveZ){
                emit_sltiu(psp_t0, dst, 1);
                emit_ins(psp_gp, psp_t0, _flag_Z8, _flag_Z8);
            }

            flag_dirty = flag_dirty || (op.saveZ || op.saveN);          
        }
        break;

        case OP_MOV:
        {
            int32_t regs[2] = {op.rd, op.rs1};
            regman.get(2, regs);
            const psp_gpr_t dst = (psp_gpr_t)regs[0];
            const psp_gpr_t src = (psp_gpr_t)regs[1];

            if (op.preOpType == PRE_OP_IMM) {
                emit_li(dst, op.imm);
            } else if (op.preOpType == PRE_OP_LSL_IMM) {
                emit_sll(dst, src, op.imm ? op.imm : 0);
                if (op.saveC && op.imm) {
                    // C = bit that was shifted out = bit (32-imm) of src
                    emit_srl(psp_t0, src, 32 - op.imm);
                    emit_andi(psp_t0, psp_t0, 1);
                    emit_ins(psp_gp, psp_t0, _flag_C8, _flag_C8);
                }
            } else if (op.preOpType == PRE_OP_LSR_IMM) {
                u32 sh = op.imm ? op.imm : 32;
                if (sh >= 32) {
                    if (op.saveC) {
                        emit_srl(psp_t0, src, 31);
                        emit_ins(psp_gp, psp_t0, _flag_C8, _flag_C8);
                    }
                    emit_move(dst, psp_zero);
                } else {
                    if (op.saveC) {
                        emit_srl(psp_t0, src, sh - 1);
                        emit_andi(psp_t0, psp_t0, 1);
                        emit_ins(psp_gp, psp_t0, _flag_C8, _flag_C8);
                    }
                    emit_srl(dst, src, sh);
                }
            } else if (op.preOpType == PRE_OP_ASR_IMM) {
                u32 sh = op.imm ? op.imm : 32;
                if (sh >= 32) {
                    emit_sra(dst, src, 31);
                    if (op.saveC) {
                        emit_srl(psp_t0, src, 31);
                        emit_ins(psp_gp, psp_t0, _flag_C8, _flag_C8);
                    }
                } else {
                    if (op.saveC) {
                        emit_srl(psp_t0, src, sh - 1);
                        emit_andi(psp_t0, psp_t0, 1);
                        emit_ins(psp_gp, psp_t0, _flag_C8, _flag_C8);
                    }
                    emit_sra(dst, src, sh);
                }
            } else {
                emit_move(dst, src);
            }

            if (op.saveN) {
                emit_srl(psp_t0, dst, 31);
                emit_ins(psp_gp, psp_t0, _flag_N8, _flag_N8);
            }
            if (op.saveZ) {
                emit_sltiu(psp_t0, dst, 1);
                emit_ins(psp_gp, psp_t0, _flag_Z8, _flag_Z8);
            }

            regman.mark_dirty(dst);

#if THUMB_JIT_DEBUG
            {
                regman.flush_all();
                regman.reset();
                emit_move(psp_a0, RCPU);
                emit_li(psp_a1, op.preOpType == PRE_OP_LSL_IMM ? TJD_LSL :
                                op.preOpType == PRE_OP_LSR_IMM ? TJD_LSR :
                                op.preOpType == PRE_OP_ASR_IMM ? TJD_ASR : 99);
                emit_li(psp_a2, op.op_pc);
                emit_li(psp_a3, op.rd);
                emit_jal(thumb_jit_trace);
                emit_nop();
            }
#endif

            flag_dirty = flag_dirty || (op.saveZ || op.saveN || op.saveC);
        }
        break;

        case OP_LDR:
        case OP_LDRH:
        {
           int32_t regs[3] = {op.rd, op.rs1, op.rs2};

           regman.get(3, regs);
           const psp_gpr_t rd = (psp_gpr_t)regs[0]; 
           const psp_gpr_t rs1 = (psp_gpr_t)regs[1];
           const psp_gpr_t rs2 = (psp_gpr_t)regs[2];

           //printf("0x%x\n", emit_getCurrAdr());
 
            if (op.rs1 == (uint32_t)-1) {
                // PC-relative or absolute address precomputed in imm
                emit_li(psp_a0, op.imm & ~3u);
            } else if (op.preOpType == PRE_OP_REG) {
                emit_addu(psp_a0, rs1, rs2);
            } else {
                emit_addiu(psp_a0, rs1, op.imm);
            }

            regman.flush_all();

            if (op._op == OP_LDR){
                emit_sll(regs[0], psp_a0, 3);           // save rotation in callee-saved reg

                emit_jal(_MMU_read32<PROCNUM>);
                emit_ins(psp_a0, psp_zero, 1, 0);       // addr &= ~3  (delay slot)

                emit_rotrv(regs[0], psp_v0, regs[0]);  // rotate by saved amount

            }else{
                emit_jal(_MMU_read16<PROCNUM>);
                emit_ins(psp_a0, psp_zero, 0, 0);

                emit_move(regs[0], psp_v0);
            }

            regman.reset(); 

            regman.map(op.rd, (psp_gpr_t)regs[0]);
            regman.mark_dirty((psp_gpr_t)regs[0]);
           break;
        }

        case OP_STR:
        case OP_STRH:
        {

           //printf("0x%x\n", emit_getCurrAdr());

           int32_t regs[3] = {op.rd, op.rs1, op.rs2};

           regman.get(3, regs);
           const psp_gpr_t rs1 = (psp_gpr_t)regs[0]; 
           const psp_gpr_t rs2 = (psp_gpr_t)regs[1];  


           if (op.preOpType == PRE_OP_REG) {
                //printf("0x%x\n", emit_getCurrAdr());
                const psp_gpr_t rs3 = (psp_gpr_t)regs[2]; 
                emit_addu(psp_a0, rs1, rs3);
            }
            else
                emit_addiu(psp_a0, rs1, op.imm);

            emit_move(psp_a1, rs2);
            regman.flush_all(true);


            if (op._op == OP_STR){
                    emit_jal(_MMU_write32<PROCNUM>); 
                    emit_ins(psp_a0, psp_zero, 1, 0);
            }else{
                    emit_jal(_MMU_write16<PROCNUM>); 
                    emit_ins(psp_a0, psp_zero, 0, 0);
            }

           break;
        }

        case OP_MVN:
        {
            int32_t regs[2] = {op.rd, op.rs1};

            regman.get(2, regs);
            const psp_gpr_t dst = (psp_gpr_t)regs[0]; 
            const psp_gpr_t rs1 = (psp_gpr_t)regs[1]; 

            emit_not(dst, rs1);
            
            if (op.saveN){
                emit_srl(psp_t0, dst, 31);
                emit_ins(psp_gp, psp_t0, _flag_N8, _flag_N8);
            }

            if (op.saveZ){
                emit_sltiu(psp_t0, dst, 1);
                emit_ins(psp_gp, psp_t0, _flag_Z8, _flag_Z8);
            }

            regman.mark_dirty(dst);

            flag_dirty = flag_dirty || (op.saveZ || op.saveN);      
        }
        break;

        case OP_BIC:
        {
            int32_t regs[3] = {op.rd, op.rs1, op.rs2};
            regman.get(3, regs);
            const psp_gpr_t dst = (psp_gpr_t)regs[0];
            const psp_gpr_t rs1 = (psp_gpr_t)regs[1];
            const psp_gpr_t rs2 = (psp_gpr_t)regs[2];

            emit_not(psp_t0, rs2);
            emit_and(dst, rs1, psp_t0);

            if (op.saveN) {
                emit_srl(psp_t0, dst, 31);
                emit_ins(psp_gp, psp_t0, _flag_N8, _flag_N8);
            }
            if (op.saveZ) {
                emit_sltiu(psp_t0, dst, 1);
                emit_ins(psp_gp, psp_t0, _flag_Z8, _flag_Z8);
            }

            regman.mark_dirty(dst);

#if THUMB_JIT_DEBUG
            { regman.flush_all(); regman.reset();
              emit_move(psp_a0, RCPU); emit_li(psp_a1, TJD_BIC);
              emit_li(psp_a2, op.op_pc); emit_li(psp_a3, op.rd);
              emit_jal(thumb_jit_trace); emit_nop(); }
#endif

            flag_dirty = flag_dirty || (op.saveZ || op.saveN);
        }
        break;

        case OP_NEG:
        {
            int32_t regs[2] = {op.rd, op.rs1};
            regman.get(2, regs);
            const psp_gpr_t dst = (psp_gpr_t)regs[0];
            const psp_gpr_t rs1 = (psp_gpr_t)regs[1];

            emit_negu(dst, rs1);

            if (op.saveN) {
                emit_srl(psp_t0, dst, 31);
                emit_ins(psp_gp, psp_t0, _flag_N8, _flag_N8);
            }
            if (op.saveZ) {
                emit_sltiu(psp_t0, dst, 1);
                emit_ins(psp_gp, psp_t0, _flag_Z8, _flag_Z8);
            }
            // C = (rs1 != 0) ? 0 : 1  (borrow for NEG = 0 - rs1)
            if (op.saveC) {
                emit_sltu(psp_t0, psp_zero, rs1);  // t0 = (rs1 != 0)
                emit_xori(psp_t0, psp_t0, 1);       // invert
                emit_ins(psp_gp, psp_t0, _flag_C8, _flag_C8);
            }
            if (op.saveV) {
                // V = (rs1 == 0x80000000)
                emit_lui(psp_t1, 0x8000);
                emit_xor(psp_t0, rs1, psp_t1);
                emit_sltiu(psp_t0, psp_t0, 1);
                emit_ins(psp_gp, psp_t0, _flag_V8, _flag_V8);
            }

            regman.mark_dirty(dst);

#if THUMB_JIT_DEBUG
            { regman.flush_all(); regman.reset();
              emit_move(psp_a0, RCPU); emit_li(psp_a1, TJD_NEG);
              emit_li(psp_a2, op.op_pc); emit_li(psp_a3, op.rd);
              emit_jal(thumb_jit_trace); emit_nop(); }
#endif

            flag_dirty = flag_dirty || (op.saveZ || op.saveN || op.saveC || op.saveV);
        }
        break;

        case OP_CMN:
        {
            int32_t regs[2] = {op.rs1, op.rs2};
            regman.get(2, regs);
            const psp_gpr_t rs1 = (psp_gpr_t)regs[0];
            const psp_gpr_t rs2 = (psp_gpr_t)regs[1];

            emit_addu(psp_v0, rs1, rs2);

            if (op.saveN) {
                emit_srl(psp_t1, psp_v0, 31);
                emit_ins(psp_gp, psp_t1, _flag_N8, _flag_N8);
            }
            if (op.saveZ) {
                emit_sltiu(psp_t3, psp_v0, 1);
                emit_ins(psp_gp, psp_t3, _flag_Z8, _flag_Z8);
            }
            if (op.saveC) {
                // carry-out of add: result < rs1
                emit_sltu(psp_t0, psp_v0, rs1);
                emit_ins(psp_gp, psp_t0, _flag_C8, _flag_C8);
            }
            if (op.saveV) {
                // overflow for add: (~(rs1 ^ rs2)) & (rs1 ^ result)
                emit_xor(psp_t3, rs1, rs2);
                emit_nor(psp_t3, psp_t3, psp_zero);  // ~(rs1^rs2)
                emit_xor(psp_t2, rs1, psp_v0);        // rs1 ^ result
                emit_and(psp_t2, psp_t2, psp_t3);
                emit_srl(psp_t2, psp_t2, 31);
                emit_ins(psp_gp, psp_t2, _flag_V8, _flag_V8);
            }

            flag_dirty = flag_dirty || (op.saveZ || op.saveN || op.saveC || op.saveV);
        }
        break;

        case OP_LDRB:
        {
            int32_t regs[3] = {op.rd, op.rs1, op.rs2};
            regman.get(3, regs);
            const psp_gpr_t rd  = (psp_gpr_t)regs[0];
            const psp_gpr_t rs1 = (psp_gpr_t)regs[1];
            const psp_gpr_t rs2 = (psp_gpr_t)regs[2];

            if (op.rs1 == (uint32_t)-1) {
                // PC-relative precomputed address
                emit_li(psp_a0, op.imm);
            } else if (op.preOpType == PRE_OP_REG) {
                emit_addu(psp_a0, rs1, rs2);
            } else {
                emit_addiu(psp_a0, rs1, op.imm);
            }

            regman.flush_all();
            emit_jal(_MMU_read08<PROCNUM>);
            emit_nop();

            regman.reset();
            regman.map(op.rd, rd);
            emit_move(rd, psp_v0);
            regman.mark_dirty(rd);

#if THUMB_JIT_DEBUG
            { regman.flush_all(); regman.reset();
              emit_move(psp_a0, RCPU); emit_li(psp_a1, TJD_LDRB);
              emit_li(psp_a2, op.op_pc); emit_li(psp_a3, op.rd);
              emit_jal(thumb_jit_trace); emit_nop(); }
#endif
        }
        break;

        case OP_STRB:
        {
            int32_t regs[3] = {op.rd, op.rs1, op.rs2};
            regman.get(3, regs);
            const psp_gpr_t base = (psp_gpr_t)regs[0];  // base address register
            const psp_gpr_t val  = (psp_gpr_t)regs[1];  // value to store
            const psp_gpr_t rs2  = (psp_gpr_t)regs[2];  // index (if PRE_OP_REG)

            if (op.preOpType == PRE_OP_REG) {
                emit_addu(psp_a0, base, rs2);
            } else {
                emit_addiu(psp_a0, base, op.imm);
            }
            emit_move(psp_a1, val);
            regman.flush_all(true);

            emit_jal(_MMU_write08<PROCNUM>);
            emit_nop();

#if THUMB_JIT_DEBUG
            { regman.flush_all(); regman.reset();
              emit_move(psp_a0, RCPU); emit_li(psp_a1, TJD_STRB);
              emit_li(psp_a2, op.op_pc); emit_li(psp_a3, op.rd);
              emit_jal(thumb_jit_trace); emit_nop(); }
#endif
        }
        break;

        default:
            printf("Unknown Thumb OP: %d\n", op._op);
            exit(1);
        break;
    }
}

template bool block::emitThumbBlock<0>();
template bool block::emitThumbBlock<1>();