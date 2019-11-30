#define scorr_32x256(grad,suffix)\
__global__ void dk_scorr_32x256##suffix(\
          char*              d_c  ,\
    const char* __restrict__ d_a  ,\
    const char* __restrict__ d_b  ,\
    const char* __restrict__ d_x  ,\
    float                    alpha,\
    int                      ldc  ,\
    int                      lda  ,\
    int                      ldb  ,\
    int                      cx   ,\
    int                      cy   ,\
    int                      ax   ,\
    int                      ay   ,\
    int                      fnn  ,\
    int                      cnr  ,\
    int                      pnc  ,\
    int                      qnc ){\
    __shared__ char smem[20480];     \
    float c[32];                     \
    float4 a[2], b[4];               \
    unsigned int bx=blockIdx.x;      \
    unsigned int by=blockIdx.y;      \
    unsigned int bz=blockIdx.z;      \
    unsigned int tid=threadIdx.x;    \
    unsigned int lane=tid&31;        \
    unsigned int slot=tid>>5;        \
    unsigned int u=(bx<<5)+lane;     \
    unsigned int v=(by<<8)+tid;      \
    unsigned int su=u<cnr?u:(cnr-1); \
    unsigned int sv=v<pnc?v:(pnc-1); \
    unsigned int cxy=cx*cy;          \
    unsigned int idx=su%cxy;         \
    unsigned int x=(bx<<5)+lane;     \
    unsigned int y=(by<<8)+(slot<<5);\
    unsigned int bnr=qnc*fnn;\
    d_a+=bz*qnc*lda+((((su/cxy)*ax+(idx/cy))*ay+(idx%cy))<<2);\
    d_b+=bz*qnc*ldb+sv*fnn*4;\
    d_c+=(bz*pnc+y)*ldc+(x<<2);\
    if(grad){ d_x+=(bz*pnc+y)*ldc+(x<<2); }\
    const char* d_ao=&cmem[slot*4];\
    const char* d_bo=&cmem[bnr *4];\
    unsigned int ao=*((const unsigned int*)d_ao);\
    uint4 bo0=*((const uint4*)&d_bo[0x00]);\
    uint4 bo1=*((const uint4*)&d_bo[0x10]);\
    float p0=*((const float*)&d_a[ao]);    \
    float q0=*((const float*)&d_b[bo0.x]); \
    float q1=*((const float*)&d_b[bo0.y]); \
    float q2=*((const float*)&d_b[bo0.z]); \
	float q3=*((const float*)&d_b[bo0.w]); \
	float q4=*((const float*)&d_b[bo1.x]); \
	float q5=*((const float*)&d_b[bo1.y]); \
	float q6=*((const float*)&d_b[bo1.z]); \
	float q7=*((const float*)&d_b[bo1.w]); \
    char* __restrict__ sst_base =&smem[tid<<2];\
    char* __restrict__ asld_base=&smem[(lane&0xe)<<3];\
    char* __restrict__ bsld_base=&smem[0x400+((slot<<7)|((lane&0x10)<<1)|((lane&0x1)<<4))];\
    char* __restrict__ asld=asld_base;\
    char* __restrict__ bsld=bsld_base;\
    *((float*)&sst_base[0*1024])=p0;\
    *((float*)&sst_base[1*1024])=q0;\
    *((float*)&sst_base[2*1024])=q1;\
    *((float*)&sst_base[3*1024])=q2;\
    *((float*)&sst_base[4*1024])=q3;\
    *((float*)&sst_base[5*1024])=q4;\
    *((float*)&sst_base[6*1024])=q5;\
    *((float*)&sst_base[7*1024])=q6;\
    *((float*)&sst_base[8*1024])=q7;\
    __syncthreads();\
    SZERO32(c)\
    b[0]=*((float4*)&bsld[0x00]);\
    a[0]=*((float4*)&asld[0x00]);\
    b[1]=*((float4*)&bsld[0x40]);\
    unsigned int ofs=0x2400;     \
    for( int k=bnr-8; k>0; k-=8 )\
    {\
        d_bo+=32;\
        bo0=*((const uint4*)&d_bo[0x00]);\
        bo1=*((const uint4*)&d_bo[0x10]);\
        ao=*((const unsigned int*)(d_ao+=32));\
        q0=*((const float*)&d_b[bo0.x]);\
        q1=*((const float*)&d_b[bo0.y]);\
        q2=*((const float*)&d_b[bo0.z]);\
        q3=*((const float*)&d_b[bo0.w]);\
        q4=*((const float*)&d_b[bo1.x]);\
        q5=*((const float*)&d_b[bo1.y]);\
        q6=*((const float*)&d_b[bo1.z]);\
        q7=*((const float*)&d_b[bo1.w]);\
        p0=*((const float*)&d_a[ao]);   \
        b[2]=*((float4*)&bsld[1*0x400+0x00]);\
        a[1]=*((float4*)&asld[1*0x080+0x00]);\
        b[3]=*((float4*)&bsld[1*0x400+0x40]);\
        BOP4x8(c,&a[0],&b[0])                \
        b[0]=*((float4*)&bsld[2*0x400+0x00]);\
        a[0]=*((float4*)&asld[2*0x080+0x00]);\
        b[1]=*((float4*)&bsld[2*0x400+0x40]);\
        BOP4x8(c,&a[1],&b[2])                \
        b[2]=*((float4*)&bsld[3*0x400+0x00]);\
        a[1]=*((float4*)&asld[3*0x080+0x00]);\
        b[3]=*((float4*)&bsld[3*0x400+0x40]);\
        BOP4x8(c,&a[0],&b[0])                \
        b[0]=*((float4*)&bsld[4*0x400+0x00]);\
        a[0]=*((float4*)&asld[4*0x080+0x00]);\
        b[1]=*((float4*)&bsld[4*0x400+0x40]);\
        BOP4x8(c,&a[1],&b[2])                \
        b[2]=*((float4*)&bsld[5*0x400+0x00]);\
        a[1]=*((float4*)&asld[5*0x080+0x00]);\
        b[3]=*((float4*)&bsld[5*0x400+0x40]);\
        char* __restrict__ sst=sst_base+ofs; \
        BOP4x8(c,&a[0],&b[0])                \
        b[0]=*((float4*)&bsld[6*0x400+0x00]);\
        a[0]=*((float4*)&asld[6*0x080+0x00]);\
        b[1]=*((float4*)&bsld[6*0x400+0x40]);\
        *((float*)&sst[0*1024])=p0;\
        *((float*)&sst[1*1024])=q0;\
        *((float*)&sst[2*1024])=q1;\
        *((float*)&sst[3*1024])=q2;\
        *((float*)&sst[4*1024])=q3;\
        BOP4x8(c,&a[1],&b[2])\
        b[2]=*((float4*)&bsld[7*0x400+0x00]);\
        a[1]=*((float4*)&asld[7*0x080+0x00]);\
        b[3]=*((float4*)&bsld[7*0x400+0x40]);\
        *((float*)&sst[5*1024])=q4;\
        *((float*)&sst[6*1024])=q5;\
        *((float*)&sst[7*1024])=q6;\
        *((float*)&sst[8*1024])=q7;\
        BOP4x8(c,&a[0],&b[0])\
        asld=asld_base+ofs;\
        bsld=bsld_base+ofs;\
        __syncthreads();\
        b[0]=*((float4*)&bsld[0x00]);\
        a[0]=*((float4*)&asld[0x00]);\
        b[1]=*((float4*)&bsld[0x40]);\
        BOP4x8(c,&a[1],&b[2])\
        ofs^=0x2400;\
    }\
    b[2]=*((float4*)&bsld[1*0x400+0x00]);\
    a[1]=*((float4*)&asld[1*0x080+0x00]);\
    b[3]=*((float4*)&bsld[1*0x400+0x40]);\
    BOP4x8(c,&a[0],&b[0])                \
    b[0]=*((float4*)&bsld[2*0x400+0x00]);\
    a[0]=*((float4*)&asld[2*0x080+0x00]);\
    b[1]=*((float4*)&bsld[2*0x400+0x40]);\
    BOP4x8(c,&a[1],&b[2])                \
    b[2]=*((float4*)&bsld[3*0x400+0x00]);\
    a[1]=*((float4*)&asld[3*0x080+0x00]);\
    b[3]=*((float4*)&bsld[3*0x400+0x40]);\
    BOP4x8(c,&a[0],&b[0])                \
    b[0]=*((float4*)&bsld[4*0x400+0x00]);\
    a[0]=*((float4*)&asld[4*0x080+0x00]);\
    b[1]=*((float4*)&bsld[4*0x400+0x40]);\
    BOP4x8(c,&a[1],&b[2])                \
    b[2]=*((float4*)&bsld[5*0x400+0x00]);\
    a[1]=*((float4*)&asld[5*0x080+0x00]);\
    b[3]=*((float4*)&bsld[5*0x400+0x40]);\
    BOP4x8(c,&a[0],&b[0])                \
    b[0]=*((float4*)&bsld[6*0x400+0x00]);\
    a[0]=*((float4*)&asld[6*0x080+0x00]);\
    b[1]=*((float4*)&bsld[6*0x400+0x40]);\
    BOP4x8(c,&a[1],&b[2])                \
    b[2]=*((float4*)&bsld[7*0x400+0x00]);\
    a[1]=*((float4*)&asld[7*0x080+0x00]);\
    b[3]=*((float4*)&bsld[7*0x400+0x40]);\
    BOP4x8(c,&a[0],&b[0])\
    BOP4x8(c,&a[1],&b[2])\
    sgemm_epilog32x32##suffix( d_c, grad?d_x:0, &smem[slot<<9], c, lane, ldc, x, cnr, pnc-y, alpha );\
}

scorr_32x256(0,)
scorr_32x256(1,_drelu)
scorr_32x256(1,_xdrv)