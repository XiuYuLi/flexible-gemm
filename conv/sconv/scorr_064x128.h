#define scorr_64x128(grad,suffix)\
__global__ void dk_scorr_64x128##suffix(\
          char *              d_c  , \
    const char * __restrict__ d_a  , \
    const char * __restrict__ d_b  , \
    const char * __restrict__ d_x  , \
    float                     alpha, \
    int                       ldc  , \
    int                       lda  , \
    int                       ldb  , \
    int                       cx   , \
    int                       cy   , \
    int                       ax   , \
    int                       ay   , \
    int                       fnn  , \
    int                       cnr  , \
    int                       pnc  , \
    int                       qnc ){ \
    __shared__ char smem[12288];     \
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
    unsigned int v=(by<<7)+ tid;     \
    unsigned int su=u<cnr?u:(cnr-1); \
    unsigned int sv=v<pnc?v:(pnc-1); \
    unsigned int cxy=cx*cy;          \
    unsigned int idx=su%cxy;         \
    unsigned int x=(bx<<5)+lane;     \
    unsigned int y=(by<<7)+(slot<<5);\
    unsigned int bnr=qnc*fnn;\
    d_a+=bz*qnc*lda+((((su/cxy)*ax+(idx/cy))*ay+(idx%cy))<<2);\
    d_b+=bz*qnc*ldb+sv*fnn*4;\
    d_c+=(bz*pnc+y)*ldc+(x<<3);\
    if(grad){ d_x+=(bz*pnc+y)*ldc+(x<<3); }\
    const char* d_ao=&cmem[vtid*16];\
    const char* d_bo=&cmem[bnr * 4];\
    uint4 o0=*((const uint4*) d_ao    ); \
    uint4 o1=*((const uint4*)&d_bo[ 0]); \
    uint4 o2=*((const uint4*)&d_bo[16]); \
    float p0=*((const float*)&d_a[o0.x]);\
    float p1=*((const float*)&d_a[o0.y]);\
    float p2=*((const float*)&d_a[o0.z]);\
    float p3=*((const float*)&d_a[o0.w]);\
    float q0=*((const float*)&d_b[o1.x]);\
    float q1=*((const float*)&d_b[o1.y]);\
    float q2=*((const float*)&d_b[o1.z]);\
    float q3=*((const float*)&d_b[o1.w]);\
    float q4=*((const float*)&d_b[o2.x]);\
    float q5=*((const float*)&d_b[o2.y]);\
    float q6=*((const float*)&d_b[o2.z]);\
    float q7=*((const float*)&d_b[o2.w]);\
    char* __restrict__ asst_base=&smem[(vtid<<10)|(utid<<2)];\
    char* __restrict__ bsst_base=&smem[0x800+(tid<<2)];\
    char* __restrict__ asld_base=&smem[(lane&0xe)<<3]; \
    char* __restrict__ bsld_base=&smem[0x800+((slot<<7)|((lane&0x10)<<1)|((lane&0x1)<<4))];\
    char* __restrict__ asld=asld_base;\
    char* __restrict__ bsld=bsld_base;\
    *((float*)&asst_base[0*256])=p0;\
    *((float*)&asst_base[1*256])=p1;\
    *((float*)&asst_base[2*256])=p2;\
    *((float*)&asst_base[3*256])=p3;\
    *((float*)&bsst_base[0*512])=q0;\
    *((float*)&bsst_base[1*512])=q1;\
    *((float*)&bsst_base[2*512])=q2;\
    *((float*)&bsst_base[3*512])=q3;\
    *((float*)&bsst_base[4*512])=q4;\
    *((float*)&bsst_base[5*512])=q5;\
    *((float*)&bsst_base[6*512])=q6;\
    *((float*)&bsst_base[7*512])=q7;\
    __syncthreads();\
    SZERO64(c)\
    b[0]=*((float4*)&bsld[0x00]);\
    a[0]=*((float4*)&asld[0x00]);\
    b[1]=*((float4*)&bsld[0x40]);\
    a[1]=*((float4*)&asld[0x80]);\
    unsigned int ofs=0x1800;     \
    for( int k=bnr-8; k>0; k-=8 )\
    {\
        d_bo+=32;\
        o1=*((const uint4*)&d_bo[ 0] );\
        o2=*((const uint4*)&d_bo[16] );\
        o0=*((const uint4*)(d_ao+=32));\
        q0=*((const float*)&d_b[o1.x]);\
        q1=*((const float*)&d_b[o1.y]);\
        q2=*((const float*)&d_b[o1.z]);\
        q3=*((const float*)&d_b[o1.w]);\
        p0=*((const float*)&d_a[o0.x]);\
        p1=*((const float*)&d_a[o0.y]);\
        p2=*((const float*)&d_a[o0.z]);\
        p3=*((const float*)&d_a[o0.w]);\
        q4=*((const float*)&d_b[o2.x]);\
        q5=*((const float*)&d_b[o2.y]);\
        q6=*((const float*)&d_b[o2.z]);\
        q7=*((const float*)&d_b[o2.w]);\
        b[2]=*((float4*)&bsld[1*512+0x00]);   \
        a[2]=*((float4*)&asld[1*256+0x00]);   \
        b[3]=*((float4*)&bsld[1*512+0x40]);   \
        a[3]=*((float4*)&asld[1*256+0x80]);   \
        BOP8x8(c,&a[0],&b[0])                 \
        b[0]=*((float4*)&bsld[2*512+0x00]);   \
        a[0]=*((float4*)&asld[2*256+0x00]);   \
        b[1]=*((float4*)&bsld[2*512+0x40]);   \
        a[1]=*((float4*)&asld[2*256+0x80]);   \
        BOP8x8(c,&a[2],&b[2])                 \
        b[2]=*((float4*)&bsld[3*512+0x00]);   \
        a[2]=*((float4*)&asld[3*256+0x00]);   \
        b[3]=*((float4*)&bsld[3*512+0x40]);   \
        a[3]=*((float4*)&asld[3*256+0x80]);   \
        BOP8x8(c,&a[0],&b[0])                 \
        b[0]=*((float4*)&bsld[4*512+0x00]);   \
        a[0]=*((float4*)&asld[4*256+0x00]);   \
        b[1]=*((float4*)&bsld[4*512+0x40]);   \
        a[1]=*((float4*)&asld[4*256+0x80]);   \
        BOP8x8(c,&a[2],&b[2])                 \
        b[2]=*((float4*)&bsld[5*512+0x00]);   \
        a[2]=*((float4*)&asld[5*256+0x00]);   \
        b[3]=*((float4*)&bsld[5*512+0x40]);   \
        a[3]=*((float4*)&asld[5*256+0x80]);   \
        char* __restrict__ asst=asst_base+ofs;\
        char* __restrict__ bsst=bsst_base+ofs;\
        BOP8x8(c,&a[0],&b[0])                 \
        b[0]=*((float4*)&bsld[6*512+0x00]);   \
        a[0]=*((float4*)&asld[6*256+0x00]);   \
        b[1]=*((float4*)&bsld[6*512+0x40]);   \
        a[1]=*((float4*)&asld[6*256+0x80]);   \
        *((float*)&asst[0*256])=p0;           \
        *((float*)&asst[1*256])=p1;           \
        *((float*)&asst[2*256])=p2;           \
        *((float*)&asst[3*256])=p3;           \
        BOP8x8(c,&a[2],&b[2])                 \
        b[2]=*((float4*)&bsld[7*512+0x00]);   \
        a[2]=*((float4*)&asld[7*256+0x00]);   \
        b[3]=*((float4*)&bsld[7*512+0x40]);   \
        a[3]=*((float4*)&asld[7*256+0x80]);   \
        *((float*)&bsst[0*512])=q0;\
        *((float*)&bsst[1*512])=q1;\
        *((float*)&bsst[2*512])=q2;\
        *((float*)&bsst[3*512])=q3;\
        *((float*)&bsst[4*512])=q4;\
        *((float*)&bsst[5*512])=q5;\
        *((float*)&bsst[6*512])=q6;\
        *((float*)&bsst[7*512])=q7;\
        BOP8x8(c,&a[0],&b[0])\
        asld=asld_base+ofs;\
        bsld=bsld_base+ofs;\
        __syncthreads();\
        b[0]=*((float4*)&bsld[0x00]);\
        a[0]=*((float4*)&asld[0x00]);\
        b[1]=*((float4*)&bsld[0x40]);\
        a[1]=*((float4*)&asld[0x80]);\
        BOP8x8(c,&a[2],&b[2])\
        ofs^=0x1800;\
    }\
    b[2]=*((float4*)&bsld[1*512+0x00]);\
    a[2]=*((float4*)&asld[1*256+0x00]);\
    b[3]=*((float4*)&bsld[1*512+0x40]);\
    a[3]=*((float4*)&asld[1*256+0x80]);\
    BOP8x8(c,&a[0],&b[0])              \
    b[0]=*((float4*)&bsld[2*512+0x00]);\
    a[0]=*((float4*)&asld[2*256+0x00]);\
    b[1]=*((float4*)&bsld[2*512+0x40]);\
    a[1]=*((float4*)&asld[2*256+0x80]);\
    BOP8x8(c,&a[2],&b[2])              \
    b[2]=*((float4*)&bsld[3*512+0x00]);\
    a[2]=*((float4*)&asld[3*256+0x00]);\
    b[3]=*((float4*)&bsld[3*512+0x40]);\
    a[3]=*((float4*)&asld[3*256+0x80]);\
    BOP8x8(c,&a[0],&b[0])              \
    b[0]=*((float4*)&bsld[4*512+0x00]);\
    a[0]=*((float4*)&asld[4*256+0x00]);\
    b[1]=*((float4*)&bsld[4*512+0x40]);\
    a[1]=*((float4*)&asld[4*256+0x80]);\
    BOP8x8(c,&a[2],&b[2])              \
    b[2]=*((float4*)&bsld[5*512+0x00]);\
    a[2]=*((float4*)&asld[5*256+0x00]);\
    b[3]=*((float4*)&bsld[5*512+0x40]);\
    a[3]=*((float4*)&asld[5*256+0x80]);\
    BOP8x8(c,&a[0],&b[0])              \
    b[0]=*((float4*)&bsld[6*512+0x00]);\
    a[0]=*((float4*)&asld[6*256+0x00]);\
    b[1]=*((float4*)&bsld[6*512+0x40]);\
    a[1]=*((float4*)&asld[6*256+0x80]);\
    BOP8x8(c,&a[2],&b[2])              \
    b[2]=*((float4*)&bsld[7*512+0x00]);\
    a[2]=*((float4*)&asld[7*256+0x00]);\
    b[3]=*((float4*)&bsld[7*512+0x40]);\
    a[3]=*((float4*)&asld[7*256+0x80]);\
    BOP8x8(c,&a[0],&b[0])\
    BOP8x8(c,&a[2],&b[2])\
	sgemm_epilog64x32##suffix( d_c, grad?d_x:0, &smem[slot<<10], c, lane, ldc, x, cnr>>1, pnc-y, alpha );\
}

scorr_64x128(0,)
scorr_64x128(1,_drelu)
scorr_64x128(1,_xdrv)