#define scorr_64x64(grad,suffix)\
__global__ void dk_scorr_64x64##suffix(\
          char *              d_c  ,\
    const char * __restrict__ d_a  ,\
    const char * __restrict__ d_b  ,\
    const char * __restrict__ d_x  ,\
    float                     alpha,\
    int                       ldc  ,\
    int                       lda  ,\
    int                       ldb  ,\
    int                       cx   ,\
    int                       cy   ,\
    int                       ax   ,\
    int                       ay   ,\
    int                       fnn  ,\
    int                       cnr  ,\
    int                       pnc  ,\
    int                       qnc ){\
    __shared__ char smem[8192];     \
    float c[64];                    \
    float4 a[4], b[4];              \
    unsigned int bx=blockIdx.x;     \
    unsigned int by=blockIdx.y;     \
    unsigned int tid=threadIdx.x;   \
    unsigned int lane=tid&31;       \
    unsigned int slot=tid>>5;       \
    unsigned int u=(bx<<6)|tid;     \
    unsigned int v=tid;             \
    unsigned int su=u<cnr?u:(cnr-1);\
    unsigned int sv=v<pnc?v:(pnc-1);\
    unsigned int cxy=cx*cy;     \
    unsigned int idx=su%cxy;    \
    unsigned int x=(bx<<5)|lane;\
    unsigned int y=slot<<5;     \
    unsigned int bnr=qnc*fnn;   \
    d_a+=by*qnc*lda+((((su/cxy)*ax+(idx/cy))*ay+(idx%cy))<<2);\
    d_b+=by*qnc*ldb+sv*fnn*4;\
    d_c+=(by*pnc+y)*ldc+(x<<3);\
    if(grad){ d_x+=(by*pnc+y)*ldc+(x<<3); }\
    const char* d_ao=&cmem[0];\
    const char* d_bo=&cmem[bnr*4];\
    uint4 ao0=*((const uint4*)&d_ao[ 0]); \
    uint4 ao1=*((const uint4*)&d_ao[16]); \
    uint4 bo0=*((const uint4*)&d_bo[ 0]); \
    uint4 bo1=*((const uint4*)&d_bo[16]); \
    float p0=*((const float*)&d_a[ao0.x]);\
    float p1=*((const float*)&d_a[ao0.y]);\
    float p2=*((const float*)&d_a[ao0.z]);\
    float p3=*((const float*)&d_a[ao0.w]);\
    float p4=*((const float*)&d_a[ao1.x]);\
    float p5=*((const float*)&d_a[ao1.y]);\
    float p6=*((const float*)&d_a[ao1.z]);\
    float p7=*((const float*)&d_a[ao1.w]);\
    float q0=*((const float*)&d_b[bo0.x]);\
    float q1=*((const float*)&d_b[bo0.y]);\
    float q2=*((const float*)&d_b[bo0.z]);\
    float q3=*((const float*)&d_b[bo0.w]);\
    float q4=*((const float*)&d_b[bo1.x]);\
    float q5=*((const float*)&d_b[bo1.y]);\
    float q6=*((const float*)&d_b[bo1.z]);\
    float q7=*((const float*)&d_b[bo1.w]);\
    char* __restrict__ sst_base =&smem[tid<<2];\
    char* __restrict__ asld_base=&smem[(lane&0xe)<<3];\
    char* __restrict__ bsld_base=&smem[0x800|(slot<<7)|((lane&0x10)<<1)|((lane&0x1)<<4)];\
    char* __restrict__ asld=asld_base;\
    char* __restrict__ bsld=bsld_base;\
    *((float*)&sst_base[     0*256])=p0;\
    *((float*)&sst_base[     1*256])=p1;\
    *((float*)&sst_base[     2*256])=p2;\
    *((float*)&sst_base[     3*256])=p3;\
    *((float*)&sst_base[     4*256])=p4;\
    *((float*)&sst_base[     5*256])=p5;\
    *((float*)&sst_base[     6*256])=p6;\
    *((float*)&sst_base[     7*256])=p7;\
    *((float*)&sst_base[2048+0*256])=q0;\
    *((float*)&sst_base[2048+1*256])=q1;\
    *((float*)&sst_base[2048+2*256])=q2;\
    *((float*)&sst_base[2048+3*256])=q3;\
    *((float*)&sst_base[2048+4*256])=q4;\
    *((float*)&sst_base[2048+5*256])=q5;\
    *((float*)&sst_base[2048+6*256])=q6;\
    *((float*)&sst_base[2048+7*256])=q7;\
    __syncthreads();\
    SZERO64(c)\
    b[0]=*((float4*)&bsld[0x00]);\
    a[0]=*((float4*)&asld[0x00]);\
    b[1]=*((float4*)&bsld[0x40]);\
    a[1]=*((float4*)&asld[0x80]);\
    unsigned int ofs=0x1000;     \
    for( int k=bnr-8; k>0; k-=8 )\
    {\
        d_bo+=32; d_ao+=32;\
        bo0=*((const uint4*)&d_bo[ 0]); \
        ao0=*((const uint4*)&d_ao[ 0]); \
        bo1=*((const uint4*)&d_bo[16]); \
        ao1=*((const uint4*)&d_ao[16]); \
        q0=*((const float*)&d_b[bo0.x]);\
        q1=*((const float*)&d_b[bo0.y]);\
        q2=*((const float*)&d_b[bo0.z]);\
        q3=*((const float*)&d_b[bo0.w]);\
        p0=*((const float*)&d_a[ao0.x]);\
        p1=*((const float*)&d_a[ao0.y]);\
        p2=*((const float*)&d_a[ao0.z]);\
        p3=*((const float*)&d_a[ao0.w]);\
        q4=*((const float*)&d_b[bo1.x]);\
        q5=*((const float*)&d_b[bo1.y]);\
        q6=*((const float*)&d_b[bo1.z]);\
        q7=*((const float*)&d_b[bo1.w]);\
        p4=*((const float*)&d_a[ao1.x]);\
        p5=*((const float*)&d_a[ao1.y]);\
        p6=*((const float*)&d_a[ao1.z]);\
        p7=*((const float*)&d_a[ao1.w]);\
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
        *((float*)&sst[2048+0*256])=q0;\
        *((float*)&sst[2048+1*256])=q1;\
        *((float*)&sst[2048+2*256])=q2;\
        *((float*)&sst[2048+3*256])=q3;\
        *((float*)&sst[2048+4*256])=q4;\
        *((float*)&sst[2048+5*256])=q5;\
        *((float*)&sst[2048+6*256])=q6;\
        *((float*)&sst[2048+7*256])=q7;\
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
    sgemm_epilog64x32##suffix( d_c, grad?d_x:0, &smem[slot<<10], c, lane, ldc, x, cnr>>1, pnc-y, alpha );\
}

scorr_64x64(0,)
scorr_64x64(1,_drelu)
scorr_64x64(1,_xdrv)