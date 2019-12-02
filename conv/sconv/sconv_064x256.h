#define sconv_64x256(add_bias,suffix)   \
__global__ void dk_sconv_64x256##suffix(\
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
    float c[64];                     \
    float4 a[4], b[4];               \
    unsigned int bx=blockIdx.x;      \
    unsigned int by=blockIdx.y;      \
    unsigned int bz=blockIdx.z;      \
    unsigned int tid=threadIdx.x;    \
    unsigned int utid=tid&63;        \
    unsigned int vtid=tid>>6;        \
    unsigned int lane=tid&31;        \
    unsigned int slot=tid>>5;        \
    unsigned int u=(bx<<6)+utid;     \
    unsigned int v=(by<<8)+ tid;     \
    unsigned int p=u<cnr?u:(cnr-1);  \
    unsigned int q=v<qnc?v:(qnc-1);  \
    unsigned int cxy=cx*cy;          \
    unsigned int idx=p%cxy;          \
    unsigned int x=(bx<<5)+lane;     \
    unsigned int y=(by<<8)+(slot<<5);\
    d_a+=bz*pnc*lda+((((p/cxy)*ax+sv*(idx/cy))*ay+su*(idx%cy))<<2);\
    d_b+=(bz*qnc+q)*ldb;         \
    d_c+=(bz*qnc+y)*ldc+(x<<3);  \
    const char* d_o=&cmem[vtid*8];\
    uint2 o=*((const uint2*)d_o); \
    float  p0=*((const float *)&d_a[o.x]);\
    float  p1=*((const float *)&d_a[o.y]);\
    float4 q0=*((const float4*)(d_b+ 0) );\
    float4 q1=*((const float4*)(d_b+16) );\
    char* __restrict__ asst_base=&smem[(vtid<<9)|(utid<<2)];\
    char* __restrict__ bsst_base=&smem[0x800+(tid<<2)];\
    char* __restrict__ asld_base=&smem[(lane&0xe)<<3]; \
    char* __restrict__ bsld_base=&smem[0x800+((slot<<7)|((lane&0x10)<<1)|((lane&0x1)<<4))];\
    char* __restrict__ asld=asld_base; \
    char* __restrict__ bsld=bsld_base; \
    *((float*)&asst_base[0* 256])=p0;  \
    *((float*)&asst_base[1* 256])=p1;  \
    *((float*)&bsst_base[0*1024])=q0.x;\
    *((float*)&bsst_base[1*1024])=q0.y;\
    *((float*)&bsst_base[2*1024])=q0.z;\
    *((float*)&bsst_base[3*1024])=q0.w;\
    *((float*)&bsst_base[4*1024])=q1.x;\
    *((float*)&bsst_base[5*1024])=q1.y;\
    *((float*)&bsst_base[6*1024])=q1.z;\
    *((float*)&bsst_base[7*1024])=q1.w;\
    __syncthreads();\
    if(add_bias){ if(v<qnc){ s_bias[tid]=d_bias[bz*qnc+v];  } }\
    SZERO64(c)\
    b[0]=*((float4*)&bsld[0x00]);\
    a[0]=*((float4*)&asld[0x00]);\
    b[1]=*((float4*)&bsld[0x40]);\
    a[1]=*((float4*)&asld[0x80]);\
    unsigned int ofs=0x2800;     \
    for( int k=bnr-8; k>0; k-=8 )\
    {\
        d_b+=32;\
        o=*((const uint2*)(d_o+=32));  \
        q0=*((const float4*)(d_b+ 0)); \
        q1=*((const float4*)(d_b+16)); \
        p0=*((const float *)&d_a[o.x]);\
        p1=*((const float *)&d_a[o.y]);\
        b[2]=*((float4*)&bsld[1*0x400+0x00]); \
        a[2]=*((float4*)&asld[1*0x100+0x00]); \
        b[3]=*((float4*)&bsld[1*0x400+0x40]); \
        a[3]=*((float4*)&asld[1*0x100+0x80]); \
        BOP8x8(c,&a[0],&b[0])                 \
        b[0]=*((float4*)&bsld[2*0x400+0x00]); \
        a[0]=*((float4*)&asld[2*0x100+0x00]); \
        b[1]=*((float4*)&bsld[2*0x400+0x40]); \
        a[1]=*((float4*)&asld[2*0x100+0x80]); \
        BOP8x8(c,&a[2],&b[2])                 \
        b[2]=*((float4*)&bsld[3*0x400+0x00]); \
        a[2]=*((float4*)&asld[3*0x100+0x00]); \
        b[3]=*((float4*)&bsld[3*0x400+0x40]); \
        a[3]=*((float4*)&asld[3*0x100+0x80]); \
        BOP8x8(c,&a[0],&b[0])                 \
        b[0]=*((float4*)&bsld[4*0x400+0x00]); \
        a[0]=*((float4*)&asld[4*0x100+0x00]); \
        b[1]=*((float4*)&bsld[4*0x400+0x40]); \
        a[1]=*((float4*)&asld[4*0x100+0x80]); \
        BOP8x8(c,&a[2],&b[2])                 \
        b[2]=*((float4*)&bsld[5*0x400+0x00]); \
        a[2]=*((float4*)&asld[5*0x100+0x00]); \
        b[3]=*((float4*)&bsld[5*0x400+0x40]); \
        a[3]=*((float4*)&asld[5*0x100+0x80]); \
        char* __restrict__ asst=asst_base+ofs;\
        char* __restrict__ bsst=bsst_base+ofs;\
        BOP8x8(c,&a[0],&b[0])                 \
        b[0]=*((float4*)&bsld[6*0x400+0x00]); \
        a[0]=*((float4*)&asld[6*0x100+0x00]); \
        b[1]=*((float4*)&bsld[6*0x400+0x40]); \
        a[1]=*((float4*)&asld[6*0x100+0x80]); \
        *((float*)&asst[0*256])=p0;   \
        *((float*)&asst[1*256])=p1;   \
        *((float*)&bsst[0*1024])=q0.x;\
        *((float*)&bsst[1*1024])=q0.y;\
        *((float*)&bsst[2*1024])=q0.z;\
        *((float*)&bsst[3*1024])=q0.w;\
        BOP8x8(c,&a[2],&b[2])         \
        b[2]=*((float4*)&bsld[7*0x400+0x00]);\
        a[2]=*((float4*)&asld[7*0x100+0x00]);\
        b[3]=*((float4*)&bsld[7*0x400+0x40]);\
        a[3]=*((float4*)&asld[7*0x100+0x80]);\
        *((float*)&bsst[4*1024])=q1.x;\
        *((float*)&bsst[5*1024])=q1.y;\
        *((float*)&bsst[6*1024])=q1.z;\
        *((float*)&bsst[7*1024])=q1.w;\
        BOP8x8(c,&a[0],&b[0])\
        asld=asld_base+ofs;  \
        bsld=bsld_base+ofs;  \
        __syncthreads();\
        b[0]=*((float4*)&bsld[0x00]);\
        a[0]=*((float4*)&asld[0x00]);\
        b[1]=*((float4*)&bsld[0x40]);\
        a[1]=*((float4*)&asld[0x80]);\
        BOP8x8(c,&a[2],&b[2])\
        ofs^=0x2800;\
    }\
    b[2]=*((float4*)&bsld[1*0x400+0x00]);\
    a[2]=*((float4*)&asld[1*0x100+0x00]);\
    b[3]=*((float4*)&bsld[1*0x400+0x40]);\
    a[3]=*((float4*)&asld[1*0x100+0x80]);\
    BOP8x8(c,&a[0],&b[0])                \
    b[0]=*((float4*)&bsld[2*0x400+0x00]);\
    a[0]=*((float4*)&asld[2*0x100+0x00]);\
    b[1]=*((float4*)&bsld[2*0x400+0x40]);\
    a[1]=*((float4*)&asld[2*0x100+0x80]);\
    BOP8x8(c,&a[2],&b[2])                \
    b[2]=*((float4*)&bsld[3*0x400+0x00]);\
    a[2]=*((float4*)&asld[3*0x100+0x00]);\
    b[3]=*((float4*)&bsld[3*0x400+0x40]);\
    a[3]=*((float4*)&asld[3*0x100+0x80]);\
    BOP8x8(c,&a[0],&b[0])                \
    b[0]=*((float4*)&bsld[4*0x400+0x00]);\
    a[0]=*((float4*)&asld[4*0x100+0x00]);\
    b[1]=*((float4*)&bsld[4*0x400+0x40]);\
    a[1]=*((float4*)&asld[4*0x100+0x80]);\
    BOP8x8(c,&a[2],&b[2])                \
    b[2]=*((float4*)&bsld[5*0x400+0x00]);\
    a[2]=*((float4*)&asld[5*0x100+0x00]);\
    b[3]=*((float4*)&bsld[5*0x400+0x40]);\
    a[3]=*((float4*)&asld[5*0x100+0x80]);\
    BOP8x8(c,&a[0],&b[0])                \
    b[0]=*((float4*)&bsld[6*0x400+0x00]);\
    a[0]=*((float4*)&asld[6*0x100+0x00]);\
    b[1]=*((float4*)&bsld[6*0x400+0x40]);\
    a[1]=*((float4*)&asld[6*0x100+0x80]);\
    BOP8x8(c,&a[2],&b[2])                \
    b[2]=*((float4*)&bsld[7*0x400+0x00]);\
    a[2]=*((float4*)&asld[7*0x100+0x00]);\
    b[3]=*((float4*)&bsld[7*0x400+0x40]);\
    a[3]=*((float4*)&asld[7*0x100+0x80]);\
    BOP8x8(c,&a[0],&b[0])\
    BOP8x8(c,&a[2],&b[2])\
    float* bias;\
    if(add_bias){ bias=&s_bias[(slot<<5)|((lane&0x10)>>1)|((lane&1)<<2)]; }\
    sgemm_epilog64x32##suffix( d_c, (const char*)bias, &smem[slot<<10], c, lane, ldc, x, cnr>>1, qnc-y, alpha );\
}

sconv_64x256(0,)
sconv_64x256(0,_relu)
sconv_64x256(1,_bias)
sconv_64x256(1,_bias_relu)