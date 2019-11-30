#define sconv_64x64(add_bias,suffix)   \
__global__ void dk_sconv_64x64##suffix(\
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
    __shared__ char smem[8192];      \
    __shared__ float s_bias[64];     \
    float c[64];                     \
    float4 a[4], b[4];               \
    unsigned int bx=blockIdx.x;      \
    unsigned int by=blockIdx.y;      \
    unsigned int tid=threadIdx.x;    \
    unsigned int lane=tid&31;        \
    unsigned int slot=tid>>5;        \
    unsigned int u=(bx<<6)|tid;      \
    unsigned int v=tid;              \
    unsigned int p=u<cnr?u:(cnr-1);  \
    unsigned int q=v<qnc?v:(qnc-1);  \
    unsigned int cxy=cx*cy;          \
    unsigned int idx=p%cxy;          \
    unsigned int x=(bx<<5)|lane;     \
    unsigned int y=slot<<5;          \
    d_a+=by*pnc*lda+((((p/cxy)*ax+sv*(idx/cy))*ay+su*(idx%cy))<<2);\
    d_b+=(by*qnc+q)*ldb;       \
    d_c+=(by*qnc+y)*ldc+(x<<3);\
    const char* d_o=&cmem[0];  \
    unsigned int o0=*((const unsigned int*)(d_o+0x00));\
    unsigned int o1=*((const unsigned int*)(d_o+0x04));\
    unsigned int o2=*((const unsigned int*)(d_o+0x08));\
    unsigned int o3=*((const unsigned int*)(d_o+0x0c));\
    unsigned int o4=*((const unsigned int*)(d_o+0x10));\
    unsigned int o5=*((const unsigned int*)(d_o+0x14));\
    unsigned int o6=*((const unsigned int*)(d_o+0x18));\
    unsigned int o7=*((const unsigned int*)(d_o+0x1c));\
    float p0=*((const float*)&d_a[o0]);  \
    float p1=*((const float*)&d_a[o1]);  \
    float p2=*((const float*)&d_a[o2]);  \
    float p3=*((const float*)&d_a[o3]);  \
    float p4=*((const float*)&d_a[o4]);  \
    float p5=*((const float*)&d_a[o5]);  \
    float p6=*((const float*)&d_a[o6]);  \
    float p7=*((const float*)&d_a[o7]);  \
    float4 q0=*((const float4*)(d_b+ 0));\
    float4 q1=*((const float4*)(d_b+16));\
    char* __restrict__ sst_base =&smem[tid<<2];\
    char* __restrict__ asld_base=&smem[(lane&0xe)<<3];\
    char* __restrict__ bsld_base=&smem[0x800|(slot<<7)|((lane&0x10)<<1)|((lane&0x1)<<4)];\
    char* __restrict__ asld=asld_base;\
    char* __restrict__ bsld=bsld_base;\
    *((float*)&sst_base[0*256])=p0;\
    *((float*)&sst_base[1*256])=p1;\
    *((float*)&sst_base[2*256])=p2;\
    *((float*)&sst_base[3*256])=p3;\
    *((float*)&sst_base[4*256])=p4;\
    *((float*)&sst_base[5*256])=p5;\
    *((float*)&sst_base[6*256])=p6;\
    *((float*)&sst_base[7*256])=p7;\
    *((float*)&sst_base[2048+0*256])=q0.x;\
    *((float*)&sst_base[2048+1*256])=q0.y;\
    *((float*)&sst_base[2048+2*256])=q0.z;\
    *((float*)&sst_base[2048+3*256])=q0.w;\
    *((float*)&sst_base[2048+4*256])=q1.x;\
    *((float*)&sst_base[2048+5*256])=q1.y;\
    *((float*)&sst_base[2048+6*256])=q1.z;\
    *((float*)&sst_base[2048+7*256])=q1.w;\
    __syncthreads();\
    if(add_bias){ if(tid<qnc){ s_bias[tid]=d_bias[by*qnc+tid];	} }\
    SZERO64(c)\
    b[0]=*((float4*)&bsld[0x00]);\
    a[0]=*((float4*)&asld[0x00]);\
    b[1]=*((float4*)&bsld[0x40]);\
    a[1]=*((float4*)&asld[0x80]);\
    unsigned int ofs=0x1000;     \
    for( int k=bnr-8; k>0; k-=8 )\
    {\
        d_o+=32; d_b+=32;\
        q0=*((const float4*)(d_b+ 0));\
        q1=*((const float4*)(d_b+16));\
        o0=*((const unsigned int*)(d_o+0x00));\
        o1=*((const unsigned int*)(d_o+0x04));\
        o2=*((const unsigned int*)(d_o+0x08));\
        o3=*((const unsigned int*)(d_o+0x0c));\
        o4=*((const unsigned int*)(d_o+0x10));\
        o5=*((const unsigned int*)(d_o+0x14));\
        o6=*((const unsigned int*)(d_o+0x18));\
        o7=*((const unsigned int*)(d_o+0x1c));\
        p0=*((const float*)&d_a[o0]);\
        p1=*((const float*)&d_a[o1]);\
        p2=*((const float*)&d_a[o2]);\
        p3=*((const float*)&d_a[o3]);\
        p4=*((const float*)&d_a[o4]);\
        p5=*((const float*)&d_a[o5]);\
        p6=*((const float*)&d_a[o6]);\
        p7=*((const float*)&d_a[o7]);\
        b[2]=*((float4*)&bsld[1*256+0x00]); \
        a[2]=*((float4*)&asld[1*256+0x00]); \
        b[3]=*((float4*)&bsld[1*256+0x40]); \
        a[3]=*((float4*)&asld[1*256+0x80]); \
        BOP8x8(c,&a[0],&b[0])               \
        b[0]=*((float4*)&bsld[2*256+0x00]); \
        a[0]=*((float4*)&asld[2*256+0x00]); \
        b[1]=*((float4*)&bsld[2*256+0x40]); \
        a[1]=*((float4*)&asld[2*256+0x80]); \
        BOP8x8(c,&a[2],&b[2])               \
        b[2]=*((float4*)&bsld[3*256+0x00]); \
        a[2]=*((float4*)&asld[3*256+0x00]); \
        b[3]=*((float4*)&bsld[3*256+0x40]); \
        a[3]=*((float4*)&asld[3*256+0x80]); \
        BOP8x8(c,&a[0],&b[0])               \
        b[0]=*((float4*)&bsld[4*256+0x00]); \
        a[0]=*((float4*)&asld[4*256+0x00]); \
        b[1]=*((float4*)&bsld[4*256+0x40]); \
        a[1]=*((float4*)&asld[4*256+0x80]); \
        BOP8x8(c,&a[2],&b[2])               \
        b[2]=*((float4*)&bsld[5*256+0x00]); \
        a[2]=*((float4*)&asld[5*256+0x00]); \
        b[3]=*((float4*)&bsld[5*256+0x40]); \
        a[3]=*((float4*)&asld[5*256+0x80]); \
        BOP8x8(c,&a[0],&b[0])               \
        b[0]=*((float4*)&bsld[6*256+0x00]); \
        a[0]=*((float4*)&asld[6*256+0x00]); \
        b[1]=*((float4*)&bsld[6*256+0x40]); \
        a[1]=*((float4*)&asld[6*256+0x80]); \
        char* __restrict__ sst=sst_base+ofs;\
        BOP8x8(c,&a[2],&b[2])               \
        b[2]=*((float4*)&bsld[7*256+0x00]); \
        a[2]=*((float4*)&asld[7*256+0x00]); \
        b[3]=*((float4*)&bsld[7*256+0x40]); \
        a[3]=*((float4*)&asld[7*256+0x80]); \
        *((float*)&sst[0*256])=p0;\
        *((float*)&sst[1*256])=p1;\
        *((float*)&sst[2*256])=p2;\
        *((float*)&sst[3*256])=p3;\
        *((float*)&sst[4*256])=p4;\
        *((float*)&sst[5*256])=p5;\
        *((float*)&sst[6*256])=p6;\
        *((float*)&sst[7*256])=p7;\
        BOP8x8(c,&a[0],&b[0])\
        *((float*)&sst[2048+0*256])=q0.x;\
        *((float*)&sst[2048+1*256])=q0.y;\
        *((float*)&sst[2048+2*256])=q0.z;\
        *((float*)&sst[2048+3*256])=q0.w;\
        *((float*)&sst[2048+4*256])=q1.x;\
        *((float*)&sst[2048+5*256])=q1.y;\
        *((float*)&sst[2048+6*256])=q1.z;\
        *((float*)&sst[2048+7*256])=q1.w;\
        asld=asld_base+ofs;\
        bsld=bsld_base+ofs;\
        __syncthreads();\
        b[0]=*((float4*)&bsld[0x00]);\
        a[0]=*((float4*)&asld[0x00]);\
        b[1]=*((float4*)&bsld[0x40]);\
        a[1]=*((float4*)&asld[0x80]);\
        BOP8x8(c,&a[2],&b[2])\
        ofs^=0x1000;\
    }\
    b[2]=*((float4*)&bsld[1*256+0x00]);\
    a[2]=*((float4*)&asld[1*256+0x00]);\
    b[3]=*((float4*)&bsld[1*256+0x40]);\
    a[3]=*((float4*)&asld[1*256+0x80]);\
    BOP8x8(c,&a[0],&b[0])              \
    b[0]=*((float4*)&bsld[2*256+0x00]);\
    a[0]=*((float4*)&asld[2*256+0x00]);\
    b[1]=*((float4*)&bsld[2*256+0x40]);\
    a[1]=*((float4*)&asld[2*256+0x80]);\
    BOP8x8(c,&a[2],&b[2])              \
    b[2]=*((float4*)&bsld[3*256+0x00]);\
    a[2]=*((float4*)&asld[3*256+0x00]);\
    b[3]=*((float4*)&bsld[3*256+0x40]);\
    a[3]=*((float4*)&asld[3*256+0x80]);\
    BOP8x8(c,&a[0],&b[0])              \
    b[0]=*((float4*)&bsld[4*256+0x00]);\
    a[0]=*((float4*)&asld[4*256+0x00]);\
    b[1]=*((float4*)&bsld[4*256+0x40]);\
    a[1]=*((float4*)&asld[4*256+0x80]);\
    BOP8x8(c,&a[2],&b[2])              \
    b[2]=*((float4*)&bsld[5*256+0x00]);\
    a[2]=*((float4*)&asld[5*256+0x00]);\
    b[3]=*((float4*)&bsld[5*256+0x40]);\
    a[3]=*((float4*)&asld[5*256+0x80]);\
    BOP8x8(c,&a[0],&b[0])              \
    b[0]=*((float4*)&bsld[6*256+0x00]);\
    a[0]=*((float4*)&asld[6*256+0x00]);\
    b[1]=*((float4*)&bsld[6*256+0x40]);\
    a[1]=*((float4*)&asld[6*256+0x80]);\
    BOP8x8(c,&a[2],&b[2])              \
    b[2]=*((float4*)&bsld[7*256+0x00]);\
    a[2]=*((float4*)&asld[7*256+0x00]);\
    b[3]=*((float4*)&bsld[7*256+0x40]);\
    a[3]=*((float4*)&asld[7*256+0x80]);\
    BOP8x8(c,&a[0],&b[0])\
    BOP8x8(c,&a[2],&b[2])\
    float* bias;\
    if(add_bias){ bias=&s_bias[(slot<<5)|((lane&0x10)>>1)|((lane&1)<<2)]; }\
    sgemm_epilog64x32##suffix( d_c, (const char*)bias, &smem[slot<<10], c, lane, ldc, x, cnr>>1, qnc-y, alpha );\
}

sconv_64x64(0,)
sconv_64x64(0,_relu)
sconv_64x64(1,_bias)
sconv_64x64(1,_bias_relu)