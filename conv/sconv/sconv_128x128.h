#define sconv_128x128(add_bias,suffix)   \
__global__ void dk_sconv_128x128##suffix(\
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
    __shared__ char smem[16384];     \
    __shared__ float s_bias[128];    \
    float c[64];                     \
    float4 a[4], b[4];               \
    unsigned int bx=blockIdx.x;      \
    unsigned int by=blockIdx.y;      \
    unsigned int tid=threadIdx.x;    \
    unsigned int vtid=tid>>7;        \
    unsigned int utid=tid&127;       \
    unsigned int lane=tid&31;        \
    unsigned int slot=tid>>5;        \
    unsigned int slot_x=slot&1;      \
    unsigned int slot_y=slot>>1;     \
    unsigned int u=(bx<<7)+utid;     \
    unsigned int v=(by<<7)+utid;     \
    unsigned int r=u<cnr?u:(cnr-1);  \
    unsigned int s=v<qnc?v:(qnc-1);  \
    unsigned int cxy=cx*cy;          \
    unsigned int idx=r%cxy;          \
    unsigned int x=(bx<<6)+(slot_x<<5)+lane;\
    unsigned int y=(by<<7)+(slot_y<<5);     \
    d_a+=((((r/cxy)*ax+sv*(idx/cy))*ay+su*(idx%cy))<<2);\
    d_b+=s*ldb+(vtid<<4);\
    d_c+=y*ldc+(x<<3);   \
    const char* d_o=&cmem[vtid*16];\
    uint4 o=*((const uint4*)d_o);  \
    float p0=*((const float *)&d_a[o.x]);\
    float p1=*((const float *)&d_a[o.y]);\
    float p2=*((const float *)&d_a[o.z]);\
    float p3=*((const float *)&d_a[o.w]);\
    float4 q=*((const float4*)d_b);\
    char* __restrict__ sst_base=&smem[(vtid<<11)|(utid<<2)];\
    char* __restrict__ asld_base=&smem[(slot_x<<8)|((lane&0xe)<<3)];\
    char* __restrict__ bsld_base=&smem[0x1000|(slot_y<<7)|((lane&0x10)<<1)|((lane&0x1)<<4)];\
    char* __restrict__ asld=asld_base;\
    char* __restrict__ bsld=bsld_base;\
    *((float*)&sst_base[0x0000])=p0; \
    *((float*)&sst_base[0x0200])=p1; \
    *((float*)&sst_base[0x0400])=p2; \
    *((float*)&sst_base[0x0600])=p3; \
    *((float*)&sst_base[0x1000])=q.x;\
    *((float*)&sst_base[0x1200])=q.y;\
    *((float*)&sst_base[0x1400])=q.z;\
    *((float*)&sst_base[0x1600])=q.w;\
    __syncthreads();\
    if(add_bias){ if((v<qnc)&(vtid==0)){ s_bias[utid]=d_bias[v]; } }\
    SZERO64(c)\
    b[0]=*((float4*)&bsld[0x00]);\
    a[0]=*((float4*)&asld[0x00]);\
    b[1]=*((float4*)&bsld[0x40]);\
    a[1]=*((float4*)&asld[0x80]);\
    unsigned int ofs=8192;       \
    for( int k=bnr-8; k>0; k-=8 )\
    {                            \
        o=*((const uint4*)(d_o+=32)); \
        q=*((const float4*)(d_b+=32));\
        p0=*((const float*)&d_a[o.x]);\
        p1=*((const float*)&d_a[o.y]);\
        p2=*((const float*)&d_a[o.z]);\
        p3=*((const float*)&d_a[o.w]);\
        b[2]=*((float4*)&bsld[1*512+0x00]); \
        a[2]=*((float4*)&asld[1*512+0x00]); \
        b[3]=*((float4*)&bsld[1*512+0x40]); \
        a[3]=*((float4*)&asld[1*512+0x80]); \
        BOP8x8(c,&a[0],&b[0])               \
        b[0]=*((float4*)&bsld[2*512+0x00]); \
        a[0]=*((float4*)&asld[2*512+0x00]); \
        b[1]=*((float4*)&bsld[2*512+0x40]); \
        a[1]=*((float4*)&asld[2*512+0x80]); \
        BOP8x8(c,&a[2],&b[2])               \
        b[2]=*((float4*)&bsld[3*512+0x00]); \
        a[2]=*((float4*)&asld[3*512+0x00]); \
        b[3]=*((float4*)&bsld[3*512+0x40]); \
        a[3]=*((float4*)&asld[3*512+0x80]); \
        BOP8x8(c,&a[0],&b[0])               \
        b[0]=*((float4*)&bsld[4*512+0x00]); \
        a[0]=*((float4*)&asld[4*512+0x00]); \
        b[1]=*((float4*)&bsld[4*512+0x40]); \
        a[1]=*((float4*)&asld[4*512+0x80]); \
        BOP8x8(c,&a[2],&b[2])               \
        b[2]=*((float4*)&bsld[5*512+0x00]); \
        a[2]=*((float4*)&asld[5*512+0x00]); \
        b[3]=*((float4*)&bsld[5*512+0x40]); \
        a[3]=*((float4*)&asld[5*512+0x80]); \
        BOP8x8(c,&a[0],&b[0])               \
        b[0]=*((float4*)&bsld[6*512+0x00]); \
        a[0]=*((float4*)&asld[6*512+0x00]); \
        b[1]=*((float4*)&bsld[6*512+0x40]); \
        a[1]=*((float4*)&asld[6*512+0x80]); \
        char* __restrict__ sst=sst_base+ofs;\
        BOP8x8(c,&a[2],&b[2])               \
        b[2]=*((float4*)&bsld[7*512+0x00]); \
        a[2]=*((float4*)&asld[7*512+0x00]); \
        b[3]=*((float4*)&bsld[7*512+0x40]); \
        a[3]=*((float4*)&asld[7*512+0x80]); \
        *((float*)&sst[0x0000])=p0;\
        *((float*)&sst[0x0200])=p1;\
        *((float*)&sst[0x0400])=p2;\
        *((float*)&sst[0x0600])=p3;\
        BOP8x8(c,&a[0],&b[0])\
        *((float*)&sst[0x1000])=q.x;\
        *((float*)&sst[0x1200])=q.y;\
        *((float*)&sst[0x1400])=q.z;\
        *((float*)&sst[0x1600])=q.w;\
        asld=asld_base+ofs;\
        bsld=bsld_base+ofs;\
        __syncthreads();   \
        b[0]=*((float4*)&bsld[0x00]);\
        a[0]=*((float4*)&asld[0x00]);\
        b[1]=*((float4*)&bsld[0x40]);\
        a[1]=*((float4*)&asld[0x80]);\
        BOP8x8(c,&a[2],&b[2])\
        ofs^=0x2000;\
    }\
    b[2]=*((float4*)&bsld[1*512+0x00]);\
    a[2]=*((float4*)&asld[1*512+0x00]);\
    b[3]=*((float4*)&bsld[1*512+0x40]);\
    a[3]=*((float4*)&asld[1*512+0x80]);\
    BOP8x8(c,&a[0],&b[0])              \
    b[0]=*((float4*)&bsld[2*512+0x00]);\
    a[0]=*((float4*)&asld[2*512+0x00]);\
    b[1]=*((float4*)&bsld[2*512+0x40]);\
    a[1]=*((float4*)&asld[2*512+0x80]);\
    BOP8x8(c,&a[2],&b[2])              \
    b[2]=*((float4*)&bsld[3*512+0x00]);\
    a[2]=*((float4*)&asld[3*512+0x00]);\
    b[3]=*((float4*)&bsld[3*512+0x40]);\
    a[3]=*((float4*)&asld[3*512+0x80]);\
    BOP8x8(c,&a[0],&b[0])              \
    b[0]=*((float4*)&bsld[4*512+0x00]);\
    a[0]=*((float4*)&asld[4*512+0x00]);\
    b[1]=*((float4*)&bsld[4*512+0x40]);\
    a[1]=*((float4*)&asld[4*512+0x80]);\
    BOP8x8(c,&a[2],&b[2])              \
    b[2]=*((float4*)&bsld[5*512+0x00]);\
    a[2]=*((float4*)&asld[5*512+0x00]);\
    b[3]=*((float4*)&bsld[5*512+0x40]);\
    a[3]=*((float4*)&asld[5*512+0x80]);\
    BOP8x8(c,&a[0],&b[0])              \
    b[0]=*((float4*)&bsld[6*512+0x00]);\
    a[0]=*((float4*)&asld[6*512+0x00]);\
    b[1]=*((float4*)&bsld[6*512+0x40]);\
    a[1]=*((float4*)&asld[6*512+0x80]);\
    BOP8x8(c,&a[2],&b[2])              \
    b[2]=*((float4*)&bsld[7*512+0x00]);\
    a[2]=*((float4*)&asld[7*512+0x00]);\
    b[3]=*((float4*)&bsld[7*512+0x40]);\
    a[3]=*((float4*)&asld[7*512+0x80]);\
    BOP8x8(c,&a[0],&b[0])\
    BOP8x8(c,&a[2],&b[2])\
    float* bias;         \
    if(add_bias){ bias=&s_bias[(slot_y<<5)|((lane&0x10)>>1)|((lane&1)<<2)]; }\
    sgemm_epilog64x32##suffix( d_c, (const char*)bias, &smem[slot<<10], c, lane, ldc, x, (cnr+1)>>1, qnc-y, alpha );\
}

sconv_128x128(0,)
sconv_128x128(0,_relu)
sconv_128x128(1,_bias)
sconv_128x128(1,_bias_relu)
