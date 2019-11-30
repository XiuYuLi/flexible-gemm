cuda kernels of flexible-gemm is used in deepcore and i`think is same means as 'indirect-gemm of google'.

param comments:

    ldc : leading size of output data, >=cx*cy*bs('>' if padding else '==')
    lda : leading size of input  data, >=ax*ay*bs('>' if padding else '==')
    ldb : leading size of filter data, >=R*S*C   ('>' if padding else '==')
    cx  : W of output
    cy  : H of output
    ax  : W of input
    ay  : H of input
    su  : u stride
    sv  : v stride
    bnr : num rows of filter, R*S*C or R*S*T*C for 3D filter
    cnr : num rows of output, cx*cy*bs, bs is batch size
    pnc : num input  channel
    qnc : num output channels

===============================================================================

Index computing:

index size is R*S*C if 2d-conv
index size is R*S*T*C if 3d-conv
...
index size is R*S*...*C if nd-conv

/*
params of gen_indices
    anx, any : W&H of input  data
    bnx, bny : R&S of filter data
    inc      : num input channels
    lda      : >=anx*any*bs for CNHW layout, equal anx*any for NCHW
    du, dv   : u&v dilations
    izero    : the position where stored zero value to solve case of R*S*C is not multiply of 8, 
               e.g -> alloc input data with more 4bytes in last and store the zero value in that
*/

void gen_indices( uint32_t* idx, uint32_t anx, uint32_t any, uint32_t bnx, uint32_t bny, uint32_t inc, uint32_t lda, uint32_t du, uint32_t dv, uint32_t izero )
{
    for( uint32_t c=0; c<inc; ++c ){
        for( uint32_t v=0; v<bny; ++v ){
            for( uint32_t u=0; u<bnx; ++u ){
                *idx=(c*lda+v*dv*anx+u*du)<<2; ++idx;
            }
        }
    }
    uint32_t k=bnx*bny*inc;
	if((k&7)!=0){
        uint32_t n=alignment(k,8);
        for( uint32_t i=k; i<n; ++i ){ *idx=izero<<2; ++idx; }
    }
}