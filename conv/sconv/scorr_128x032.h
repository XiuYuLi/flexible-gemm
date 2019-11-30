#define scorr_128x32(grad,suffix)\
__global__ void dk_scorr_128x32##suffix(\
          char* 			 d_c  ,\
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
    __shared__ char smem[10240]; \
    float c[32];                 \
    float4 a[2], b[4];           \
    unsigned int bx=blockIdx.x;  \
    unsigned int by=blockIdx.y;  \
    unsigned int tid=threadIdx.x;\
    unsigned int lane=tid&31;    \
    unsigned int slot=tid>>5;    \
    unsigned int u=(bx<<7)|tid;  \
    unsigned int su=u<cnr?u:(cnr-1);\
    unsigned int sv=lane<pnc?lane:(pnc-1);\
    unsigned int cxy=cx*cy; \
    unsigned int idx=su%cxy;\
    unsigned int bnr=qnc*fnn;\
    d_a+=by*qnc*lda+((((su/cxy)*ax+(idx/cy))*ay+(idx%cy))<<2);\
    d_b+=by*qnc*ldb+sv*fnn*4;\
    d_c+=by*pnc*ldc+(u<<2);  \
    if(grad){ d_x+=by*pnc*ldc+(u<<2); }\
    char* d_ao=&cmem[0];\
    char* d_bo=&cmem[bnr*4+slot*8];\
    uint4 o0=*((const uint4*)(d_ao+ 0));\
    uint4 o1=*((const uint4*)(d_ao+16));\
	uint2 o2=*((const uint2*) d_bo    );\
    float p0=*((const float*)&d_a[o0.x]);\
    float p1=*((const float*)&d_a[o0.y]);\
    float p2=*((const float*)&d_a[o0.z]);\
    float p3=*((const float*)&d_a[o0.w]);\
    float p4=*((const float*)&d_a[o1.x]);\
    float p5=*((const float*)&d_a[o1.y]);\
    float p6=*((const float*)&d_a[o1.z]);\
    float p7=*((const float*)&d_a[o1.w]);\
    float q0=*((const float*)&d_b[o2.x]);\
    float q1=*((const float*)&d_b[o2.y]);\
    char* __restrict__ asst_base=&smem[tid<<2];\
    char* __restrict__ bsst_base=&smem[0x1000|(slot<<8)|(lane<<2)];\
    char* __restrict__ asld_base=&smem[(slot<<7)|((lane&0xe)<<3)]; \
    char* __restrict__ bsld_base=&smem[0x1000|((lane&0x10)<<1)|((lane&0x1)<<4)];\
    char* __restrict__ asld=asld_base;\
    char* __restrict__ bsld=bsld_base;\
    *((float*)&asst_base[0*512])=p0;\
    *((float*)&asst_base[1*512])=p1;\
    *((float*)&asst_base[2*512])=p2;\
    *((float*)&asst_base[3*512])=p3;\
    *((float*)&asst_base[4*512])=p4;\
    *((float*)&asst_base[5*512])=p5;\
    *((float*)&asst_base[6*512])=p6;\
    *((float*)&asst_base[7*512])=p7;\
    *((float*)&bsst_base[0*128])=q0;\
    *((float*)&bsst_base[1*128])=q1;\
    __syncthreads();\
    SZERO32(c)\
    b[0]=*((float4*)&bsld[0x00]);\
    a[0]=*((float4*)&asld[0x00]);\
    b[1]=*((float4*)&bsld[0x40]);\
    unsigned int ofs=0x1400;     \
    for( int k=bnr-8; k>0; k-=8 )\
    {\
        d_ao+=32;\
        o2=*((const uint2*)(d_bo+=32));\
        o0=*((const uint4*)(d_ao+  0));\
        o1=*((const uint4*)(d_ao+ 16));\
        q0=*((const float*)&d_b[o2.x]);\
        q1=*((const float*)&d_b[o2.y]);\
        p0=*((const float*)&d_a[o0.x]);\
        p1=*((const float*)&d_a[o0.y]);\
        p2=*((const float*)&d_a[o0.z]);\
        p3=*((const float*)&d_a[o0.w]);\
        p4=*((const float*)&d_a[o1.x]);\
        p5=*((const float*)&d_a[o1.y]);\
        p6=*((const float*)&d_a[o1.z]);\
        p7=*((const float*)&d_a[o1.w]);\
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
        char* __restrict__ asst=asst_base+ofs;\
        char* __restrict__ bsst=bsst_base+ofs;\
        BOP4x8(c,&a[0],&b[0])\
        *((float*)&asst[0*512])=p0;\
        *((float*)&asst[1*512])=p1;\
        *((float*)&asst[2*512])=p2;\
        *((float*)&asst[3*512])=p3;\
        *((float*)&asst[4*512])=p4;\
        *((float*)&asst[5*512])=p5;\
        *((float*)&asst[6*512])=p6;\
        *((float*)&asst[7*512])=p7;\
        *((float*)&bsst[0*128])=q0;\
        *((float*)&bsst[1*128])=q1;\
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
    sgemm_epilog32x32##suffix( d_c, grad?d_x:0, &smem[slot<<9], c, lane, ldc, u, cnr, pnc, alpha );\
}

scorr_128x32(0,)
scorr_128x32(1,_drelu)
scorr_128x32(1,_xdrv)