#define sconv_32x32(add_bias,suffix)   \
__global__ void dk_sconv_32x32##suffix(\
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
    __shared__ char smem[4096];  \
    __shared__ float s_bias[32]; \
    float c[32];                 \
    float4 a[2], b[4];           \
    unsigned int bx=blockIdx.x;  \
    unsigned int by=blockIdx.y;  \
    unsigned int tid=threadIdx.x;\
    unsigned int x=(bx<<5)+tid;  \
    unsigned int p=x  <cnr?x  :(cnr-1);\
    unsigned int q=tid<qnc?tid:(qnc-1);\
    unsigned int cxy=cx*cy;\
    unsigned int idx=p%cxy;\
    d_a+=by*pnc*lda+((((p/cxy)*ax+sv*(idx/cy))*ay+su*(idx%cy))<<2);\
    d_b+=(by*qnc+q)*ldb;     \
    d_c+=by*qnc*ldc+(x<<2);  \
    const char* d_o=&cmem[0];\
    uint4 o0=*((const uint4*)&d_o[ 0]);  \
    uint4 o1=*((const uint4*)&d_o[16]);  \
    float p0=*((const float*)&d_a[o0.x]);\
    float p1=*((const float*)&d_a[o0.y]);\
    float p2=*((const float*)&d_a[o0.z]);\
    float p3=*((const float*)&d_a[o0.w]);\
    float p4=*((const float*)&d_a[o1.x]);\
    float p5=*((const float*)&d_a[o1.y]);\
    float p6=*((const float*)&d_a[o1.z]);\
    float p7=*((const float*)&d_a[o1.w]);\
    float4 q0=*((const float4*)(d_b+ 0));\
    float4 q1=*((const float4*)(d_b+16));\
    char* __restrict__ sst_base =&smem[tid<<2];\
    char* __restrict__ asld_base=&smem[(tid&0xe)<<3];\
    char* __restrict__ bsld_base=&smem[0x400+(((tid&0x10)<<1)|((tid&0x1)<<4))];\
    char* __restrict__ asld=asld_base;\
    char* __restrict__ bsld=bsld_base;\
    *((float*)&sst_base[ 0*128])=p0;  \
    *((float*)&sst_base[ 1*128])=p1;  \
    *((float*)&sst_base[ 2*128])=p2;  \
    *((float*)&sst_base[ 3*128])=p3;  \
    *((float*)&sst_base[ 4*128])=p4;  \
    *((float*)&sst_base[ 5*128])=p5;  \
    *((float*)&sst_base[ 6*128])=p6;  \
    *((float*)&sst_base[ 7*128])=p7;  \
    *((float*)&sst_base[ 8*128])=q0.x;\
    *((float*)&sst_base[ 9*128])=q0.y;\
    *((float*)&sst_base[10*128])=q0.z;\
    *((float*)&sst_base[11*128])=q0.w;\
    *((float*)&sst_base[12*128])=q1.x;\
    *((float*)&sst_base[13*128])=q1.y;\
    *((float*)&sst_base[14*128])=q1.z;\
    *((float*)&sst_base[15*128])=q1.w;\
    __syncthreads();\
    if(add_bias){ if(tid<qnc){ s_bias[tid]=d_bias[tid]; } }\
    SZERO32(c)\
    b[0]=*((float4*)&bsld[0x00]);\
    a[0]=*((float4*)&asld[0x00]);\
    b[1]=*((float4*)&bsld[0x40]);\
    unsigned int ofs=0x800;      \
    for( int k=bnr-8; k>0; k-=8 )\
    {\
        d_b+=32; d_o+=32;\
        o0=*((const uint4*)&d_o[ 0]);  \
        o1=*((const uint4*)&d_o[16]);  \
        q0=*((const float4*)(d_b+ 0)); \
        q1=*((const float4*)(d_b+16)); \
        p0=*((const float*)&d_a[o0.x]);\
        p1=*((const float*)&d_a[o0.y]);\
        p2=*((const float*)&d_a[o0.z]);\
        p3=*((const float*)&d_a[o0.w]);\
        p4=*((const float*)&d_a[o1.x]);\
        p5=*((const float*)&d_a[o1.y]);\
        p6=*((const float*)&d_a[o1.z]);\
        p7=*((const float*)&d_a[o1.w]);\
        b[2]=*((float4*)&bsld[1*0x80+0x00]);\
        a[1]=*((float4*)&asld[1*0x80+0x00]);\
        b[3]=*((float4*)&bsld[1*0x80+0x40]);\
        BOP4x8(c,&a[0],&b[0])               \
        b[0]=*((float4*)&bsld[2*0x80+0x00]);\
        a[0]=*((float4*)&asld[2*0x80+0x00]);\
        b[1]=*((float4*)&bsld[2*0x80+0x40]);\
        BOP4x8(c,&a[1],&b[2])               \
        b[2]=*((float4*)&bsld[3*0x80+0x00]);\
        a[1]=*((float4*)&asld[3*0x80+0x00]);\
        b[3]=*((float4*)&bsld[3*0x80+0x40]);\
        BOP4x8(c,&a[0],&b[0])               \
        b[0]=*((float4*)&bsld[4*0x80+0x00]);\
        a[0]=*((float4*)&asld[4*0x80+0x00]);\
        b[1]=*((float4*)&bsld[4*0x80+0x40]);\
        BOP4x8(c,&a[1],&b[2])               \
        b[2]=*((float4*)&bsld[5*0x80+0x00]);\
        a[1]=*((float4*)&asld[5*0x80+0x00]);\
        b[3]=*((float4*)&bsld[5*0x80+0x40]);\
        char* __restrict__ sst=sst_base+ofs;\
        BOP4x8(c,&a[0],&b[0])               \
        b[0]=*((float4*)&bsld[6*0x80+0x00]);\
        a[0]=*((float4*)&asld[6*0x80+0x00]);\
        b[1]=*((float4*)&bsld[6*0x80+0x40]);\
        *((float*)&sst[0*128])=p0;\
        *((float*)&sst[1*128])=p1;\
        *((float*)&sst[2*128])=p2;\
        *((float*)&sst[3*128])=p3;\
        *((float*)&sst[4*128])=p4;\
        *((float*)&sst[5*128])=p5;\
        *((float*)&sst[6*128])=p6;\
        *((float*)&sst[7*128])=p7;\
        BOP4x8(c,&a[1],&b[2])\
        b[2]=*((float4*)&bsld[7*0x80+0x00]);\
        a[1]=*((float4*)&asld[7*0x80+0x00]);\
        b[3]=*((float4*)&bsld[7*0x80+0x40]);\
        BOP4x8(c,&a[0],&b[0])\
        *((float*)&sst[ 8*128])=q0.x;\
        *((float*)&sst[ 9*128])=q0.y;\
        *((float*)&sst[10*128])=q0.z;\
        *((float*)&sst[11*128])=q0.w;\
        *((float*)&sst[12*128])=q1.x;\
        *((float*)&sst[13*128])=q1.y;\
        *((float*)&sst[14*128])=q1.z;\
        *((float*)&sst[15*128])=q1.w;\
        asld=asld_base+ofs;\
        bsld=bsld_base+ofs;\
        __syncthreads();\
        b[0]=*((float4*)&bsld[0x00]);\
        a[0]=*((float4*)&asld[0x00]);\
        b[1]=*((float4*)&bsld[0x40]);\
        BOP4x8(c,&a[1],&b[2])\
        ofs^=0x800;\
    }\
    b[2]=*((float4*)&bsld[1*0x80+0x00]);\
    a[1]=*((float4*)&asld[1*0x80+0x00]);\
    b[3]=*((float4*)&bsld[1*0x80+0x40]);\
    BOP4x8(c,&a[0],&b[0])               \
    b[0]=*((float4*)&bsld[2*0x80+0x00]);\
    a[0]=*((float4*)&asld[2*0x80+0x00]);\
    b[1]=*((float4*)&bsld[2*0x80+0x40]);\
    BOP4x8(c,&a[1],&b[2])               \
    b[2]=*((float4*)&bsld[3*0x80+0x00]);\
    a[1]=*((float4*)&asld[3*0x80+0x00]);\
    b[3]=*((float4*)&bsld[3*0x80+0x40]);\
    BOP4x8(c,&a[0],&b[0])               \
    b[0]=*((float4*)&bsld[4*0x80+0x00]);\
    a[0]=*((float4*)&asld[4*0x80+0x00]);\
    b[1]=*((float4*)&bsld[4*0x80+0x40]);\
    BOP4x8(c,&a[1],&b[2])               \
    b[2]=*((float4*)&bsld[5*0x80+0x00]);\
    a[1]=*((float4*)&asld[5*0x80+0x00]);\
    b[3]=*((float4*)&bsld[5*0x80+0x40]);\
    BOP4x8(c,&a[0],&b[0])               \
    b[0]=*((float4*)&bsld[6*0x80+0x00]);\
    a[0]=*((float4*)&asld[6*0x80+0x00]);\
    b[1]=*((float4*)&bsld[6*0x80+0x40]);\
    BOP4x8(c,&a[1],&b[2])               \
    b[2]=*((float4*)&bsld[7*0x80+0x00]);\
    a[1]=*((float4*)&asld[7*0x80+0x00]);\
    b[3]=*((float4*)&bsld[7*0x80+0x40]);\
    BOP4x8(c,&a[0],&b[0])\
    BOP4x8(c,&a[1],&b[2])\
    float* bias;\
    if(add_bias){ bias=&s_bias[((tid>>4)<<3)|((tid&1)<<2)]; }\
    sgemm_epilog32x32##suffix( d_c, (const char*)bias, smem, c, tid, ldc, x, cnr, qnc, alpha );\
}

sconv_32x32(0,)
sconv_32x32(0,_relu)
sconv_32x32(1,_bias)
sconv_32x32(1,_bias_relu)