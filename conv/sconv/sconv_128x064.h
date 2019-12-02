#define sconv_128x64(add_bias,suffix)   \
__global__ void dk_sconv_128x64##suffix(\
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
    __shared__ char smem[6144*2];    \
    __shared__ float s_bias[64];     \
    float c[64];                     \
    float4 a[4], b[4];               \
    unsigned int bx=blockIdx.x;      \
    unsigned int by=blockIdx.y;      \
    unsigned int tid=threadIdx.x;    \
    unsigned int lane=tid&31;        \
    unsigned int slot=tid>>5;        \
    unsigned int slot_x=slot&1;      \
    unsigned int slot_y=slot>>1;     \
    unsigned int utid=tid&63;        \
    unsigned int vtid=tid>>6;        \
    unsigned int u=(bx<<7)|tid;      \
    unsigned int v=utid;             \
    unsigned int r=u<cnr?u:(cnr-1);  \
    unsigned int s=v<qnc?v:(qnc-1);  \
    unsigned int cxy=cx*cy;          \
    unsigned int idx=r%cxy;          \
    unsigned int x=(bx<<6)|(slot_x<<5)|lane;\
    unsigned int y=slot_y<<5;\
    d_a+=by*pnc*lda+((((r/cxy)*ax+sv*(idx/cy))*ay+su*(idx%cy))<<2);\
    d_b+=(by*qnc+s)*ldb+(vtid<<4);\
    d_c+=(by*qnc+y)*ldc+(x<<3);   \
    char* __restrict__ asst_base=&smem[tid<<2];\
    char* __restrict__ bsst_base=&smem[0x1000|(vtid<<10)|(utid<<2)];\
    char* __restrict__ asld_base=&smem[(slot_x<<8)|((lane&0xe)<<3)];\
    char* __restrict__ bsld_base=&smem[0x1000|(slot_y<<7)|((lane&0x10)<<1)|((lane&0x1)<<4)];\
    const char* d_o=&cmem[0];            \
    uint4 ol=*((const uint4*)(d_o+0x00));\
    uint4 oh=*((const uint4*)(d_o+0x10));\
    float p0=*((const float*)&d_a[ol.x]);\
    float p1=*((const float*)&d_a[ol.y]);\
    float p2=*((const float*)&d_a[ol.z]);\
    float p3=*((const float*)&d_a[ol.w]);\
    float p4=*((const float*)&d_a[oh.x]);\
    float p5=*((const float*)&d_a[oh.y]);\
    float p6=*((const float*)&d_a[oh.z]);\
    float p7=*((const float*)&d_a[oh.w]);\
    float4 q=*((const float4*)d_b);      \
    SZERO64(c)                       \
    *((float*)&asst_base[0*512])=p0; \
    *((float*)&asst_base[1*512])=p1; \
    *((float*)&asst_base[2*512])=p2; \
    *((float*)&asst_base[3*512])=p3; \
    *((float*)&asst_base[4*512])=p4; \
    *((float*)&asst_base[5*512])=p5; \
    *((float*)&asst_base[6*512])=p6; \
    *((float*)&asst_base[7*512])=p7; \
    *((float*)&bsst_base[0*256])=q.x;\
    *((float*)&bsst_base[1*256])=q.y;\
    *((float*)&bsst_base[2*256])=q.z;\
    *((float*)&bsst_base[3*256])=q.w;\
    __syncthreads();\
    if(add_bias){ if(tid<qnc){ s_bias[tid]=d_bias[by*qnc+tid]; } }\
    char* asld=asld_base;        \
    char* bsld=bsld_base;        \
    a[0]=*((float4*)&asld[0x00]);\
    b[0]=*((float4*)&bsld[0x00]);\
    a[1]=*((float4*)&asld[0x80]);\
    b[1]=*((float4*)&bsld[0x40]);\
    unsigned int ofs=0x1800;     \
    for( int k=bnr-8; k>0; k-=8 )\
    {                            \
        d_o+=32;                 \
        ol=*((const uint4 *)(d_o+0x00));\
        oh=*((const uint4 *)(d_o+0x10));\
        q =*((const float4*)(d_b+=32)); \
        p0=*((const float *)&d_a[ol.x]);\
        p1=*((const float *)&d_a[ol.y]);\
        p2=*((const float *)&d_a[ol.z]);\
        p3=*((const float *)&d_a[ol.w]);\
        p4=*((const float *)&d_a[oh.x]);\
        p5=*((const float *)&d_a[oh.y]);\
        p6=*((const float *)&d_a[oh.z]);\
        p7=*((const float *)&d_a[oh.w]);\
        a[2]=*((float4*)&asld[1*512+0x00]);\
        b[2]=*((float4*)&bsld[1*256+0x00]);\
        a[3]=*((float4*)&asld[1*512+0x80]);\
        b[3]=*((float4*)&bsld[1*256+0x40]);\
        BOP8x8(c,&a[0],&b[0])              \
        a[0]=*((float4*)&asld[2*512+0x00]);\
        b[0]=*((float4*)&bsld[2*256+0x00]);\
        a[1]=*((float4*)&asld[2*512+0x80]);\
        b[1]=*((float4*)&bsld[2*256+0x40]);\
        BOP8x8(c,&a[2],&b[2])              \
        a[2]=*((float4*)&asld[3*512+0x00]);\
        b[2]=*((float4*)&bsld[3*256+0x00]);\
        a[3]=*((float4*)&asld[3*512+0x80]);\
        b[3]=*((float4*)&bsld[3*256+0x40]);\
        BOP8x8(c,&a[0],&b[0])              \
        a[0]=*((float4*)&asld[4*512+0x00]);\
        b[0]=*((float4*)&bsld[4*256+0x00]);\
        a[1]=*((float4*)&asld[4*512+0x80]);\
        b[1]=*((float4*)&bsld[4*256+0x40]);\
        BOP8x8(c,&a[2],&b[2])              \
        a[2]=*((float4*)&asld[5*512+0x00]);\
        b[2]=*((float4*)&bsld[5*256+0x00]);\
        a[3]=*((float4*)&asld[5*512+0x80]);\
        b[3]=*((float4*)&bsld[5*256+0x40]);\
        BOP8x8(c,&a[0],&b[0])              \
        char* asst=asst_base+ofs;          \
        char* bsst=bsst_base+ofs;          \
        a[0]=*((float4*)&asld[6*512+0x00]);\
        b[0]=*((float4*)&bsld[6*256+0x00]);\
        a[1]=*((float4*)&asld[6*512+0x80]);\
        b[1]=*((float4*)&bsld[6*256+0x40]);\
        BOP8x8(c,&a[2],&b[2])              \
        *((float*)&bsst[0*256])=q.x;       \
        *((float*)&bsst[1*256])=q.y;       \
        *((float*)&bsst[2*256])=q.z;       \
        *((float*)&bsst[3*256])=q.w;       \
        a[2]=*((float4*)&asld[7*512+0x00]);\
        b[2]=*((float4*)&bsld[7*256+0x00]);\
        a[3]=*((float4*)&asld[7*512+0x80]);\
        b[3]=*((float4*)&bsld[7*256+0x40]);\
        BOP8x8(c,&a[0],&b[0])        \
        *((float*)&asst[0*512])=p0;  \
        *((float*)&asst[1*512])=p1;  \
        *((float*)&asst[2*512])=p2;  \
        *((float*)&asst[3*512])=p3;  \
        *((float*)&asst[4*512])=p4;  \
        *((float*)&asst[5*512])=p5;  \
        *((float*)&asst[6*512])=p6;  \
        *((float*)&asst[7*512])=p7;  \
        __syncthreads();             \
        asld=asld_base+ofs;          \
        bsld=bsld_base+ofs;          \
        BOP8x8(c,&a[2],&b[2])        \
        a[0]=*((float4*)&asld[0x00]);\
        b[0]=*((float4*)&bsld[0x00]);\
        b[1]=*((float4*)&bsld[0x40]);\
        a[1]=*((float4*)&asld[0x80]);\
        ofs^=0x1800;                 \
    }                                \
    a[2]=*((float4*)&asld[1*512+0x00]);\
    b[2]=*((float4*)&bsld[1*256+0x00]);\
    a[3]=*((float4*)&asld[1*512+0x80]);\
    b[3]=*((float4*)&bsld[1*256+0x40]);\
    BOP8x8(c,&a[0],&b[0])              \
    a[0]=*((float4*)&asld[2*512+0x00]);\
    b[0]=*((float4*)&bsld[2*256+0x00]);\
    a[1]=*((float4*)&asld[2*512+0x80]);\
    b[1]=*((float4*)&bsld[2*256+0x40]);\
    BOP8x8(c,&a[2],&b[2])              \
    a[2]=*((float4*)&asld[3*512+0x00]);\
    b[2]=*((float4*)&bsld[3*256+0x00]);\
    a[3]=*((float4*)&asld[3*512+0x80]);\
    b[3]=*((float4*)&bsld[3*256+0x40]);\
    BOP8x8(c,&a[0],&b[0])              \
    a[0]=*((float4*)&asld[4*512+0x00]);\
    b[0]=*((float4*)&bsld[4*256+0x00]);\
    a[1]=*((float4*)&asld[4*512+0x80]);\
    b[1]=*((float4*)&bsld[4*256+0x40]);\
    BOP8x8(c,&a[2],&b[2])              \
    a[2]=*((float4*)&asld[5*512+0x00]);\
    b[2]=*((float4*)&bsld[5*256+0x00]);\
    a[3]=*((float4*)&asld[5*512+0x80]);\
    b[3]=*((float4*)&bsld[5*256+0x40]);\
    BOP8x8(c,&a[0],&b[0])              \
    a[0]=*((float4*)&asld[6*512+0x00]);\
    b[0]=*((float4*)&bsld[6*256+0x00]);\
    a[1]=*((float4*)&asld[6*512+0x80]);\
    b[1]=*((float4*)&bsld[6*256+0x40]);\
    BOP8x8(c,&a[2],&b[2])              \
    a[2]=*((float4*)&asld[7*512+0x00]);\
    b[2]=*((float4*)&bsld[7*256+0x00]);\
    a[3]=*((float4*)&asld[7*512+0x80]);\
    b[3]=*((float4*)&bsld[7*256+0x40]);\
    BOP8x8(c,&a[0],&b[0])\
    BOP8x8(c,&a[2],&b[2])\
    float* bias;         \
    if(add_bias){ bias=&s_bias[(slot_y<<5)|((lane&0x10)>>1)|((lane&1)<<2)]; }\
    sgemm_epilog64x32##suffix( d_c, (const char*)bias, &smem[slot<<10], c, lane, ldc, x, cnr>>1, qnc-y, alpha );\
}

sconv_128x64(0,)
sconv_128x64(0,_relu)
sconv_128x64(1,_bias)
sconv_128x64(1,_bias_relu)
