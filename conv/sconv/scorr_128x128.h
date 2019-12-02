#define scorr_128x128(grad,suffix)\
__global__ void dk_scorr_128x128##suffix(\
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
    __shared__ char smem[16384]; \
    float c[64];                 \
    float4 a[4], b[4];           \
    unsigned int bx=blockIdx.x;  \
    unsigned int by=blockIdx.y;  \
    unsigned int tid=threadIdx.x;\
    unsigned int vtid=tid>>7;    \
    unsigned int utid=tid&127;   \
    unsigned int lane=tid&31;    \
    unsigned int slot=tid>>5;    \
    unsigned int slot_x=slot&1;  \
    unsigned int slot_y=slot>>1; \
    unsigned int u=(bx<<7)+utid; \
    unsigned int v=(by<<7)+utid; \
    unsigned int su=u<cnr?u:(cnr-1);\
    unsigned int sv=v<pnc?v:(pnc-1);\
    unsigned int cxy=cx*cy; \
    unsigned int idx=su%cxy;\
    unsigned int x=(bx<<6)+(slot_x<<5)+lane;\
    unsigned int y=(by<<7)+(slot_y<<5);\
    unsigned int bnr=qnc*fnn;\
    d_a+=((((su/cxy)*ax+(idx/cy))*ay+(idx%cy))<<2);\
    d_b+=sv*fnn*4;\
    d_c+=y*ldc+(x<<3);\
    if(grad){ d_x+=y*ldc+(x<<3); }\
    const char* d_ao=&cmem[vtid*16];\
    const char* d_bo=&d_ao[bnr * 4];\
    uint4 ao=*((const uint4*)d_ao); \
    uint4 bo=*((const uint4*)d_bo); \
    float p0=*((const float*)&d_a[ao.x]);\
    float p1=*((const float*)&d_a[ao.y]);\
    float p2=*((const float*)&d_a[ao.z]);\
    float p3=*((const float*)&d_a[ao.w]);\
    float q0=*((const float*)&d_b[bo.x]);\
    float q1=*((const float*)&d_b[bo.y]);\
    float q2=*((const float*)&d_b[bo.z]);\
    float q3=*((const float*)&d_b[bo.w]);\
    char* __restrict__ sst_base=&smem[(vtid<<11)|(utid<<2)];\
    char* __restrict__ asld_base=&smem[(slot_x<<8)|((lane&0xe)<<3)];\
    char* __restrict__ bsld_base=&smem[0x1000|(slot_y<<7)|((lane&0x10)<<1)|((lane&0x1)<<4)];\
    char* __restrict__ asld=asld_base;\
    char* __restrict__ bsld=bsld_base;\
    *((float*)&sst_base[0x0000])=p0;\
    *((float*)&sst_base[0x0200])=p1;\
    *((float*)&sst_base[0x0400])=p2;\
    *((float*)&sst_base[0x0600])=p3;\
    *((float*)&sst_base[0x1000])=q0;\
    *((float*)&sst_base[0x1200])=q1;\
    *((float*)&sst_base[0x1400])=q2;\
    *((float*)&sst_base[0x1600])=q3;\
    __syncthreads();\
    SZERO64(c)\
    b[0]=*((float4*)&bsld[0x00]);\
    a[0]=*((float4*)&asld[0x00]);\
    b[1]=*((float4*)&bsld[0x40]);\
    a[1]=*((float4*)&asld[0x80]);\
    unsigned int ofs=8192;       \
    for( int k=bnr-8; k>0; k-=8 )\
    {                            \
        ao=*((const uint4*)(d_ao+=32));\
        bo=*((const uint4*)(d_bo+=32));\
        p0=*((const float*)&d_a[ao.x]);\
        p1=*((const float*)&d_a[ao.y]);\
        p2=*((const float*)&d_a[ao.z]);\
        p3=*((const float*)&d_a[ao.w]);\
        q0=*((const float*)&d_b[bo.x]);\
        q1=*((const float*)&d_b[bo.y]);\
        q2=*((const float*)&d_b[bo.z]);\
        q3=*((const float*)&d_b[bo.w]);\
        b[2]=*((float4*)&bsld[1*512+0x00]); \
        a[2]=*((float4*)&asld[1*512+0x00]); \
        b[3]=*((float4*)&bsld[1*512+0x40]); \
        a[3]=*((float4*)&asld[1*512+0x80]); \
        BOP8x8(c,&a[0],&b[0])\
        b[0]=*((float4*)&bsld[2*512+0x00]); \
        a[0]=*((float4*)&asld[2*512+0x00]); \
        b[1]=*((float4*)&bsld[2*512+0x40]); \
        a[1]=*((float4*)&asld[2*512+0x80]); \
        BOP8x8(c,&a[2],&b[2])\
        b[2]=*((float4*)&bsld[3*512+0x00]); \
        a[2]=*((float4*)&asld[3*512+0x00]); \
        b[3]=*((float4*)&bsld[3*512+0x40]); \
        a[3]=*((float4*)&asld[3*512+0x80]); \
        BOP8x8(c,&a[0],&b[0])\
        b[0]=*((float4*)&bsld[4*512+0x00]); \
        a[0]=*((float4*)&asld[4*512+0x00]); \
        b[1]=*((float4*)&bsld[4*512+0x40]); \
        a[1]=*((float4*)&asld[4*512+0x80]); \
        BOP8x8(c,&a[2],&b[2])\
        b[2]=*((float4*)&bsld[5*512+0x00]); \
        a[2]=*((float4*)&asld[5*512+0x00]); \
        b[3]=*((float4*)&bsld[5*512+0x40]); \
        a[3]=*((float4*)&asld[5*512+0x80]); \
        BOP8x8(c,&a[0],&b[0])\
        b[0]=*((float4*)&bsld[6*512+0x00]); \
        a[0]=*((float4*)&asld[6*512+0x00]); \
        b[1]=*((float4*)&bsld[6*512+0x40]); \
        a[1]=*((float4*)&asld[6*512+0x80]); \
        char* __restrict__ sst=sst_base+ofs;\
        BOP8x8(c,&a[2],&b[2])\
        b[2]=*((float4*)&bsld[7*512+0x00]); \
        a[2]=*((float4*)&asld[7*512+0x00]); \
        b[3]=*((float4*)&bsld[7*512+0x40]); \
        a[3]=*((float4*)&asld[7*512+0x80]); \
        *((float*)&sst[0x0000])=p0;\
        *((float*)&sst[0x0200])=p1;\
        *((float*)&sst[0x0400])=p2;\
        *((float*)&sst[0x0600])=p3;\
        BOP8x8(c,&a[0],&b[0])\
        *((float*)&sst[0x1000])=q0;\
        *((float*)&sst[0x1200])=q1;\
        *((float*)&sst[0x1400])=q2;\
        *((float*)&sst[0x1600])=q3;\
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
    BOP8x8(c,&a[0],&b[0])\
    b[0]=*((float4*)&bsld[2*512+0x00]);\
    a[0]=*((float4*)&asld[2*512+0x00]);\
    b[1]=*((float4*)&bsld[2*512+0x40]);\
    a[1]=*((float4*)&asld[2*512+0x80]);\
    BOP8x8(c,&a[2],&b[2])\
    b[2]=*((float4*)&bsld[3*512+0x00]);\
    a[2]=*((float4*)&asld[3*512+0x00]);\
    b[3]=*((float4*)&bsld[3*512+0x40]);\
    a[3]=*((float4*)&asld[3*512+0x80]);\
    BOP8x8(c,&a[0],&b[0])\
    b[0]=*((float4*)&bsld[4*512+0x00]);\
    a[0]=*((float4*)&asld[4*512+0x00]);\
    b[1]=*((float4*)&bsld[4*512+0x40]);\
    a[1]=*((float4*)&asld[4*512+0x80]);\
    BOP8x8(c,&a[2],&b[2])\
    b[2]=*((float4*)&bsld[5*512+0x00]);\
    a[2]=*((float4*)&asld[5*512+0x00]);\
    b[3]=*((float4*)&bsld[5*512+0x40]);\
    a[3]=*((float4*)&asld[5*512+0x80]);\
    BOP8x8(c,&a[0],&b[0])\
    b[0]=*((float4*)&bsld[6*512+0x00]);\
    a[0]=*((float4*)&asld[6*512+0x00]);\
    b[1]=*((float4*)&bsld[6*512+0x40]);\
    a[1]=*((float4*)&asld[6*512+0x80]);\
    BOP8x8(c,&a[2],&b[2])\
    b[2]=*((float4*)&bsld[7*512+0x00]);\
    a[2]=*((float4*)&asld[7*512+0x00]);\
    b[3]=*((float4*)&bsld[7*512+0x40]);\
    a[3]=*((float4*)&asld[7*512+0x80]);\
    BOP8x8(c,&a[0],&b[0])\
    BOP8x8(c,&a[2],&b[2])\
	sgemm_epilog64x32##suffix( d_c, grad?d_x:0, &smem[slot<<10], c, lane, ldc, x, (cnr+1)>>1, pnc-y, alpha );\
}

scorr_128x128(0,)
scorr_128x128(1,_drelu)
scorr_128x128(1,_xdrv)
