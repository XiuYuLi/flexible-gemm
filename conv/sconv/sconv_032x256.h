#define sconv_32x256(add_bias,suffix)   \
__global__ void dk_sconv_32x256##suffix(\
          char *              d_c   ,\
    const char * __restrict__ d_a   ,\
    const char * __restrict__ d_b   ,\
    const float* __restrict__ d_bias,\
    float                     alpha ,\
    int                       ldc   ,\
    int                       lda   ,\
    int                       ldb   ,\
    int                       cx    ,\
    int                       cy    ,\
    int                       ax    ,\
    int                       ay    ,\
    int                       su    ,\
    int                       sv    ,\
    int                       bnr   ,\
    int                       cnr   ,\
    int                       pnc   ,\
    int                       qnc ){ \
    __shared__ char smem[20480];     \
    __shared__ float s_bias[256];    \
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
    unsigned int p=u<cnr?u:(cnr-1);  \
    unsigned int q=v<qnc?v:(qnc-1);  \
    unsigned int cxy=cx*cy;          \
    unsigned int idx=p%cxy;          \
    unsigned int x=(bx<<5)+lane;     \
    unsigned int y=(by<<8)+(slot<<5);\
    d_a+=bz*pnc*lda+((((p/cxy)*ax+sv*(idx/cy))*ay+su*(idx%cy))<<2);\
    d_b+=(bz*qnc+q)*ldb;          \
    d_c+=(bz*qnc+y)*ldc+(x<<2);   \
    const char* d_o=&cmem[slot*4];\
    unsigned int o=*((const unsigned int*)d_o);\
    float p0=*((const float*)&d_a[o]);   \
    float4 p1=*((const float4*)(d_b+ 0));\
    float4 p2=*((const float4*)(d_b+16));\
    char* __restrict__ sst_base =&smem[tid<<2];\
    char* __restrict__ asld_base=&smem[(lane&0xe)<<3];\
    char* __restrict__ bsld_base=&smem[0x400+((slot<<7)|((lane&0x10)<<1)|((lane&0x1)<<4))];\
    char* __restrict__ asld=asld_base;\
    char* __restrict__ bsld=bsld_base;\
    *((float*)&sst_base[0*1024])=p0;  \
    *((float*)&sst_base[1*1024])=p1.x;\
    *((float*)&sst_base[2*1024])=p1.y;\
    *((float*)&sst_base[3*1024])=p1.z;\
    *((float*)&sst_base[4*1024])=p1.w;\
    *((float*)&sst_base[5*1024])=p2.x;\
    *((float*)&sst_base[6*1024])=p2.y;\
    *((float*)&sst_base[7*1024])=p2.z;\
    *((float*)&sst_base[8*1024])=p2.w;\
    __syncthreads();\
    if(add_bias){ if(v<qnc){ s_bias[tid]=d_bias[by*qnc+v]; } }\
    SZERO32(c)\
    b[0]=*((float4*)&bsld[0x00]);\
    a[0]=*((float4*)&asld[0x00]);\
    b[1]=*((float4*)&bsld[0x40]);\
    unsigned int ofs=0x2400;     \
    for( int k=bnr-8; k>0; k-=8 )\
    {\
        d_b+=32;\
        o=*((const unsigned int*)(d_o+=32)); \
        p1=*((const float4*)(d_b+ 0));       \
        p2=*((const float4*)(d_b+16));       \
        p0=*((const float *)&d_a[o]);        \
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
        *((float*)&sst[0*1024])=p0;  \
        *((float*)&sst[1*1024])=p1.x;\
        *((float*)&sst[2*1024])=p1.y;\
        *((float*)&sst[3*1024])=p1.z;\
        *((float*)&sst[4*1024])=p1.w;\
        BOP4x8(c,&a[1],&b[2])\
        b[2]=*((float4*)&bsld[7*0x400+0x00]);\
        a[1]=*((float4*)&asld[7*0x080+0x00]);\
        b[3]=*((float4*)&bsld[7*0x400+0x40]);\
        *((float*)&sst[5*1024])=p2.x;\
        *((float*)&sst[6*1024])=p2.y;\
        *((float*)&sst[7*1024])=p2.z;\
        *((float*)&sst[8*1024])=p2.w;\
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
    float* bias;\
    if(add_bias){ bias=&s_bias[(slot<<5)|((lane>>4)<<3)|((lane&1)<<2)]; }\
    sgemm_epilog32x32##suffix( d_c, (const char*)bias, &smem[slot<<9], c, lane, ldc, x, cnr, qnc-y, alpha );\
}

sconv_32x256(0,)
sconv_32x256(0,_relu)
sconv_32x256(1,_bias)
sconv_32x256(1,_bias_relu)