	.set noreorder

#include "pspstub.s"

	STUB_START "kcalli",0x40090000,0x00020005
	STUB_FUNC  0x5AC35241,kcall
	STUB_FUNC  0xDA5C7061,kernelRegisterIntrHandler
	STUB_END
