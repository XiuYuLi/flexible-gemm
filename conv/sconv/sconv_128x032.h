#define sconv_128x32(add_bias,suffix)   \
__global__ void dk_sconv_128x32##suffix(\
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
    __shared__ char smem[10240];     \
    __shared__ float s_bias[32];     \
    float c[32];                     \
    float4 a[2], b[4];               \
    unsigned int bx=blockIdx.x;      \
    unsigned int by=blockIdx.y;      \
    unsigned int tid=threadIdx.x;    \
    unsigned int lane=tid&31;        \
    unsigned int slot=tid>>5;        \
    unsigned int u=(bx<<7)|tid;      \
    unsigned int r=u<cnr?u:(cnr-1);  \
    unsigned int s=lane<qnc?lane:(qnc-1);\
    unsigned int cxy=cx*cy;\
    unsigned int idx=r%cxy;\
    d_a+=by*pnc*lda+((((r/cxy)*ax+sv*(idx/cy))*ay+su*(idx%cy))<<2);\
    d_b+=(by*qnc+s)*ldb+(slot<<3);\
    d_c+=by*qnc*ldc+(u<<2);\
    char* d_o=&cmem[0];\
    uint4 o0=*((const uint4*)&d_o[0x00]);\
    uint4 o1=*((const uint4*)&d_o[0x10]);\
    float p0=*((const float*)&d_a[o0.x]);\
    float p1=*((const float*)&d_a[o0.y]);\
    float p2=*((const float*)&d_a[o0.z]);\
    float p3=*((const float*)&d_a[o0.w]);\
    float p4=*((const float*)&d_a[o1.x]);\
    float p5=*((const float*)&d_a[o1.y]);\
    float p6=*((const float*)&d_a[o1.z]);\
    float p7=*((const float*)&d_a[o1.w]);\
    float2 q=*((const float2*)d_b);\
    char* __restrict__ asst_base=&smem[tid<<2];\
    char* __restrict__ bsst_base=&smem[0x1000|(slot<<8)|(lane<<2)];\
    char* __restrict__ asld_base=&smem[(slot<<7)|((lane&0xe)<<3)]; \
    char* __restrict__ bsld_base=&smem[0x1000|((lane&0x10)<<1)|((lane&0x1)<<4)];\
    char* __restrict__ asld=asld_base;\
    char* __restrict__ bsld=bsld_base;\
    *((float*)&asst_base[0*512])=p0; \
    *((float*)&asst_base[1*512])=p1; \
    *((float*)&asst_base[2*512])=p2; \
    *((float*)&asst_base[3*512])=p3; \
    *((float*)&asst_base[4*512])=p4; \
    *((float*)&asst_base[5*512])=p5; \
    *((float*)&asst_base[6*512])=p6; \
    *((float*)&asst_base[7*512])=p7; \
    *((float*)&bsst_base[0*128])=q.x;\
    *((float*)&bsst_base[1*128])=q.y;\
    __syncthreads();\
    if(add_bias){ if(tid<qnc){ s_bias[tid]=d_bias[by*qnc+tid]; } }\
    SZERO32(c)\
    b[0]=*((float4*)&bsld[0x00]);\
    a[0]=*((float4*)&asld[0x00]);\
    b[1]=*((float4*)&bsld[0x40]);\
    unsigned int ofs=0x1400;     \
    for( int k=bnr-8; k>0; k-=8 )\
    {\
        d_o+=32;\
        o0=*((const uint4*)&d_o[0x00]);\
        o1=*((const uint4*)&d_o[0x10]);\
        p0=*((const float*)&d_a[o0.x]);\
        p1=*((const float*)&d_a[o0.y]);\
        p2=*((const float*)&d_a[o0.z]);\
        p3=*((const float*)&d_a[o0.w]);\
        p4=*((const float*)&d_a[o1.x]);\
        p5=*((const float*)&d_a[o1.y]);\
        p6=*((const float*)&d_a[o1.z]);\
        p7=*((const float*)&d_a[o1.w]);\
        q=*((const float2*)(d_b+=32)); \
        b[2]=*((float4*)&bsld[1*128+0x00]);\
        a[1]=*((float4*)&asld[1*512+0x00]);\
        b[3]=*((float4*)&bsld[1*128+0x40]);\
        BOP4x8(c,&a[0],&b[0])              \
        b[0]=*((float4*)&bsld[2*128+0x00]);\
        a[0]=*((float4*)&asld[2*512+0x00]);\
        b[1]=*((float4*)&bsld[2*128+0x40]);\
        BOP4x8(c,&a[1],&b[2])              \
        b[2]=*((float4*)&bsld[3*128+0x00]);\
        a[1]=*((float4*)&asld[3*512+0x00]);\
        b[3]=*((float4*)&bsld[3*128+0x40]);\
        BOP4x8(c,&a[0],&b[0])              \
        b[0]=*((float4*)&bsld[4*128+0x00]);\
        a[0]=*((float4*)&asld[4*512+0x00]);\
        b[1]=*((float4*)&bsld[4*128+0x40]);\
        BOP4x8(c,&a[1],&b[2])              \
        b[2]=*((float4*)&bsld[5*128+0x00]);\
        a[1]=*((float4*)&asld[5*512+0x00]);\
        b[3]=*((float4*)&bsld[5*128+0x40]);\
        BOP4x8(c,&a[0],&b[0])              \
        b[0]=*((float4*)&bsld[6*128+0x00]);\
        a[0]=*((float4*)&asld[6*512+0x00]);\
        b[1]=*((float4*)&bsld[6*128+0x40]);\
        char* __restrict__ asst=asst_base+ofs;\
        char* __restrict__ bsst=bsst_base+ofs;\
        BOP4x8(c,&a[1],&b[2])              \
        b[2]=*((float4*)&bsld[7*128+0x00]);\
        a[1]=*((float4*)&asld[7*512+0x00]);\
        b[3]=*((float4*)&bsld[7*128+0x40]);\
        BOP4x8(c,&a[0],&b[0])\
        *((float*)&asst[0*512])=p0; \
        *((float*)&asst[1*512])=p1; \
        *((float*)&asst[2*512])=p2; \
        *((float*)&asst[3*512])=p3; \
        *((float*)&asst[4*512])=p4; \
        *((float*)&asst[5*512])=p5; \
        *((float*)&asst[6*512])=p6; \
        *((float*)&asst[7*512])=p7; \
        *((float*)&bsst[0*128])=q.x;\
        *((float*)&bsst[1*128])=q.y;\
        asld=asld_base+ofs;\
        bsld=bsld_base+ofs;\
        __syncthreads();\
        b[0]=*((float4*)&bsld[0x00]);\
        a[0]=*((float4*)&asld[0x00]);\
        b[1]=*((float4*)&bsld[0x40]);\
        BOP4x8(c,&a[1],&b[2])\
        ofs^=0x1400;\
    }\
    b[2]=*((float4*)&bsld[1*128+0x00]);\
    a[1]=*((float4*)&asld[1*512+0x00]);\
    b[3]=*((float4*)&bsld[1*128+0x40]);\
    BOP4x8(c,&a[0],&b[0])              \
    b[0]=*((float4*)&bsld[2*128+0x00]);\
    a[0]=*((float4*)&asld[2*512+0x00]);\
    b[1]=*((float4*)&bsld[2*128+0x40]);\
    BOP4x8(c,&a[1],&b[2])              \
    b[2]=*((float4*)&bsld[3*128+0x00]);\
    a[1]=*((float4*)&asld[3*512+0x00]);\
    b[3]=*((float4*)&bsld[3*128+0x40]);\
    BOP4x8(c,&a[0],&b[0])              \
    b[0]=*((float4*)&bsld[4*128+0x00]);\
    a[0]=*((float4*)&asld[4*512+0x00]);\
    b[1]=*((float4*)&bsld[4*128+0x40]);\
    BOP4x8(c,&a[1],&b[2])              \
    b[2]=*((float4*)&bsld[5*128+0x00]);\
    a[1]=*((float4*)&asld[5*512+0x00]);\
    b[3]=*((float4*)&bsld[5*128+0x40]);\
    BOP4x8(c,&a[0],&b[0])              \
    b[0]=*((float4*)&bsld[6*128+0x00]);\
    a[0]=*((float4*)&asld[6*512+0x00]);\
    b[1]=*((float4*)&bsld[6*128+0x40]);\
    BOP4x8(c,&a[1],&b[2])              \
    b[2]=*((float4*)&bsld[7*128+0x00]);\
    a[1]=*((float4*)&asld[7*512+0x00]);\
    b[3]=*((float4*)&bsld[7*128+0x40]);\
    BOP4x8(c,&a[0],&b[0])\
    BOP4x8(c,&a[1],&b[2])\
    float* bias;\
    if(add_bias){ bias=&s_bias[((lane&0x10)>>1)|((lane&1)<<2)]; }\
    sgemm_epilog32x32##suffix( d_c, (const char*)bias, &smem[slot<<9], c, lane, ldc, u, cnr, qnc, alpha );\
}

sconv_128x32(0,)
sconv_128x32(0,_relu)
sconv_128x32(1,_bias)
sconv_128x32(1,_bias_relu)