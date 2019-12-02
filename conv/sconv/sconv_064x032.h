#define sconv_64x32(add_bias,suffix)\
__global__ void dk_sconv_64x32##suffix(\
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
    __shared__ char smem[6144];      \
    __shared__ float s_bias[32];     \
    float c[32];                     \
    float4 a[2], b[4];               \
    unsigned int bx=blockIdx.x;      \
    unsigned int by=blockIdx.y;      \
    unsigned int tid=threadIdx.x;    \
    unsigned int lane=tid&31;        \
    unsigned int slot=tid>>5;        \
    unsigned int u=(bx<<6)|tid;      \
    unsigned int r=u<cnr?u:(cnr-1);  \
    unsigned int s=lane<qnc?lane:(qnc-1);\
    unsigned int cxy=cx*cy;\
    unsigned int idx=r%cxy;\
    d_a+=by*pnc*lda+((((r/cxy)*ax+sv*(idx/cy))*ay+su*(idx%cy))<<2);\
    d_b+=(by*qnc+s)*ldb+(slot<<4);\
    d_c+=by*qnc*ldc+(u<<2);\
    char* d_o=&cmem[0];\
    unsigned int o0=*((const unsigned int*)(d_o+0x00));\
    unsigned int o1=*((const unsigned int*)(d_o+0x04));\
    unsigned int o2=*((const unsigned int*)(d_o+0x08));\
    unsigned int o3=*((const unsigned int*)(d_o+0x0c));\
    unsigned int o4=*((const unsigned int*)(d_o+0x10));\
    unsigned int o5=*((const unsigned int*)(d_o+0x14));\
    unsigned int o6=*((const unsigned int*)(d_o+0x18));\
    unsigned int o7=*((const unsigned int*)(d_o+0x1c));\
    float p0=*((const float*)&d_a[o0]);\
    float p1=*((const float*)&d_a[o1]);\
    float p2=*((const float*)&d_a[o2]);\
    float p3=*((const float*)&d_a[o3]);\
    float p4=*((const float*)&d_a[o4]);\
    float p5=*((const float*)&d_a[o5]);\
    float p6=*((const float*)&d_a[o6]);\
    float p7=*((const float*)&d_a[o7]);\
    float4 q=*((const float4*)d_b);    \
    char* __restrict__ asst_base=&smem[tid<<2];\
    char* __restrict__ bsst_base=&smem[0x800|(slot<<9)|(lane<<2)];\
    char* __restrict__ asld_base=&smem[(slot<<7)|((lane&0xe)<<3)];\
    char* __restrict__ bsld_base=&smem[0x800|((lane&0x10)<<1)|((lane&0x1)<<4)];\
    char* __restrict__ asld=asld_base;\
    char* __restrict__ bsld=bsld_base;\
    *((float*)&asst_base[0*256])=p0; \
    *((float*)&asst_base[1*256])=p1; \
    *((float*)&asst_base[2*256])=p2; \
    *((float*)&asst_base[3*256])=p3; \
    *((float*)&asst_base[4*256])=p4; \
    *((float*)&asst_base[5*256])=p5; \
    *((float*)&asst_base[6*256])=p6; \
    *((float*)&asst_base[7*256])=p7; \
    *((float*)&bsst_base[0*128])=q.x;\
    *((float*)&bsst_base[1*128])=q.y;\
    *((float*)&bsst_base[2*128])=q.z;\
    *((float*)&bsst_base[3*128])=q.w;\
    __syncthreads();\
    if(add_bias){ if(tid<qnc){ s_bias[tid]=d_bias[by*qnc+tid]; } }\
    SZERO32(c)\
    b[0]=*((float4*)&bsld[0x00]);\
    a[0]=*((float4*)&asld[0x00]);\
    b[1]=*((float4*)&bsld[0x40]);\
    unsigned int ofs=0xc00;      \
    for( int k=bnr-8; k>0; k-=8 )\
    {\
        d_o+=32;\
        o0=*((const unsigned int*)(d_o+0x00));\
        o1=*((const unsigned int*)(d_o+0x04));\
        o2=*((const unsigned int*)(d_o+0x08));\
        o3=*((const unsigned int*)(d_o+0x0c));\
        o4=*((const unsigned int*)(d_o+0x10));\
        o5=*((const unsigned int*)(d_o+0x14));\
        o6=*((const unsigned int*)(d_o+0x18));\
        o7=*((const unsigned int*)(d_o+0x1c));\
        p0=*((const float*)&d_a[o0]); \
        p1=*((const float*)&d_a[o1]); \
        p2=*((const float*)&d_a[o2]); \
        p3=*((const float*)&d_a[o3]); \
        p4=*((const float*)&d_a[o4]); \
        p5=*((const float*)&d_a[o5]); \
        p6=*((const float*)&d_a[o6]); \
        p7=*((const float*)&d_a[o7]); \
        q=*((const float4*)(d_b+=32));\
        b[2]=*((float4*)&bsld[1*128+0x00]);\
        a[1]=*((float4*)&asld[1*256+0x00]);\
        b[3]=*((float4*)&bsld[1*128+0x40]);\
        BOP4x8(c,&a[0],&b[0])              \
        b[0]=*((float4*)&bsld[2*128+0x00]);\
        a[0]=*((float4*)&asld[2*256+0x00]);\
        b[1]=*((float4*)&bsld[2*128+0x40]);\
        BOP4x8(c,&a[1],&b[2])              \
        b[2]=*((float4*)&bsld[3*128+0x00]);\
        a[1]=*((float4*)&asld[3*256+0x00]);\
        b[3]=*((float4*)&bsld[3*128+0x40]);\
        BOP4x8(c,&a[0],&b[0])              \
        b[0]=*((float4*)&bsld[4*128+0x00]);\
        a[0]=*((float4*)&asld[4*256+0x00]);\
        b[1]=*((float4*)&bsld[4*128+0x40]);\
        BOP4x8(c,&a[1],&b[2])              \
        b[2]=*((float4*)&bsld[5*128+0x00]);\
        a[1]=*((float4*)&asld[5*256+0x00]);\
        b[3]=*((float4*)&bsld[5*128+0x40]);\
        BOP4x8(c,&a[0],&b[0])              \
        b[0]=*((float4*)&bsld[6*128+0x00]);\
        a[0]=*((float4*)&asld[6*256+0x00]);\
        b[1]=*((float4*)&bsld[6*128+0x40]);\
        char* __restrict__ asst=asst_base+ofs;\
        char* __restrict__ bsst=bsst_base+ofs;\
        BOP4x8(c,&a[1],&b[2])              \
        b[2]=*((float4*)&bsld[7*128+0x00]);\
        a[1]=*((float4*)&asld[7*256+0x00]);\
        b[3]=*((float4*)&bsld[7*128+0x40]);\
        BOP4x8(c,&a[0],&b[0])\
        *((float*)&asst[0*256])=p0; \
        *((float*)&asst[1*256])=p1; \
        *((float*)&asst[2*256])=p2; \
        *((float*)&asst[3*256])=p3; \
        *((float*)&asst[4*256])=p4; \
        *((float*)&asst[5*256])=p5; \
        *((float*)&asst[6*256])=p6; \
        *((float*)&asst[7*256])=p7; \
        *((float*)&bsst[0*128])=q.x;\
        *((float*)&bsst[1*128])=q.y;\
        *((float*)&bsst[2*128])=q.z;\
        *((float*)&bsst[3*128])=q.w;\
        asld=asld_base+ofs;\
        bsld=bsld_base+ofs;\
        __syncthreads();\
        b[0]=*((float4*)&bsld[0x00]);\
        a[0]=*((float4*)&asld[0x00]);\
        b[1]=*((float4*)&bsld[0x40]);\
        BOP4x8(c,&a[1],&b[2])\
        ofs^=0xc00;\
    }\
    b[2]=*((float4*)&bsld[1*128+0x00]);\
    a[1]=*((float4*)&asld[1*256+0x00]);\
    b[3]=*((float4*)&bsld[1*128+0x40]);\
    BOP4x8(c,&a[0],&b[0])              \
    b[0]=*((float4*)&bsld[2*128+0x00]);\
    a[0]=*((float4*)&asld[2*256+0x00]);\
    b[1]=*((float4*)&bsld[2*128+0x40]);\
    BOP4x8(c,&a[1],&b[2])              \
    b[2]=*((float4*)&bsld[3*128+0x00]);\
    a[1]=*((float4*)&asld[3*256+0x00]);\
    b[3]=*((float4*)&bsld[3*128+0x40]);\
    BOP4x8(c,&a[0],&b[0])              \
    b[0]=*((float4*)&bsld[4*128+0x00]);\
    a[0]=*((float4*)&asld[4*256+0x00]);\
    b[1]=*((float4*)&bsld[4*128+0x40]);\
    BOP4x8(c,&a[1],&b[2])              \
    b[2]=*((float4*)&bsld[5*128+0x00]);\
    a[1]=*((float4*)&asld[5*256+0x00]);\
    b[3]=*((float4*)&bsld[5*128+0x40]);\
    BOP4x8(c,&a[0],&b[0])              \
    b[0]=*((float4*)&bsld[6*128+0x00]);\
    a[0]=*((float4*)&asld[6*256+0x00]);\
    b[1]=*((float4*)&bsld[6*128+0x40]);\
    BOP4x8(c,&a[1],&b[2])              \
    b[2]=*((float4*)&bsld[7*128+0x00]);\
    a[1]=*((float4*)&asld[7*256+0x00]);\
    b[3]=*((float4*)&bsld[7*128+0x40]);\
    BOP4x8(c,&a[0],&b[0])\
    BOP4x8(c,&a[1],&b[2])\
    float* bias;\
    if(add_bias){ bias=&s_bias[((lane&0x10)>>1)|((lane&1)<<2)]; }\
    sgemm_epilog32x32##suffix( d_c, (const char*)bias, &smem[slot<<9], c, lane, ldc, u, cnr, qnc, alpha );\
}

sconv_64x32(0,)
sconv_64x32(0,_relu)
sconv_64x32(1,_bias)
sconv_64x32(1,_bias_relu)