#include <cuda_runtime.h>
#include <stdint.h>

struct fe { uint32_t v[8]; };

__device__ inline void fe_cpy(fe& r, const fe& a){ for(int i=0;i<8;i++) r.v[i]=a.v[i]; }
__device__ inline void fe_zero(fe& r){ for(int i=0;i<8;i++) r.v[i]=0; }

__device__ inline int fe_ge(const fe& a, const fe& b){ for(int i=7;i>=0;i--){ if(a.v[i]>b.v[i]) return 1; if(a.v[i]<b.v[i]) return 0; } return 1; }

__device__ inline void fe_add_raw(fe& r, const fe& a, const fe& b){ uint64_t c=0; for(int i=0;i<8;i++){ uint64_t t=(uint64_t)a.v[i]+b.v[i]+c; r.v[i]=(uint32_t)t; c=t>>32; } }
__device__ inline void fe_sub_raw(fe& r, const fe& a, const fe& b){ uint64_t c=0; for(int i=0;i<8;i++){ uint64_t t=(uint64_t)a.v[i]-b.v[i]-c; r.v[i]=(uint32_t)t; c=(t>>63)&1; } }

__device__ inline void fe_mod_p(fe& a){
  const uint32_t Pw[8]={0xFFFFFC2F,0xFFFFFFFE,0xFFFFFFFF,0xFFFFFFFF,0xFFFFFFFF,0xFFFFFFFF,0xFFFFFFFF,0xFFFFFFFF};
  fe P; for(int i=0;i<8;i++) P.v[i]=Pw[i];
  if(fe_ge(a, P)){
    uint64_t c=0; for(int i=0;i<8;i++){ uint64_t t=(uint64_t)a.v[i]-Pw[i]-c; a.v[i]=(uint32_t)t; c=(t>>63)&1; }
  }
}

__device__ inline void fe_add(fe& r, const fe& a, const fe& b){ fe_add_raw(r,a,b); fe_mod_p(r); }
__device__ inline void fe_sub(fe& r, const fe& a, const fe& b){ fe_sub_raw(r,a,b); const uint32_t P[8]={0xFFFFFC2F,0xFFFFFFFE,0xFFFFFFFF,0xFFFFFFFF,0xFFFFFFFF,0xFFFFFFFF,0xFFFFFFFF,0xFFFFFFFF}; if((int64_t)r.v[7]<0){ uint64_t c=0; for(int i=0;i<8;i++){ uint64_t t=(uint64_t)r.v[i]+P[i]+c; r.v[i]=(uint32_t)t; c=t>>32; } } }

__device__ inline void fe_reduce512(fe& r, uint64_t t[16]){
  uint64_t lo[8]; for(int i=0;i<8;i++) lo[i]=t[i];
  uint64_t hi[8]; for(int i=0;i<8;i++) hi[i]=t[i+8];
  uint64_t c=0; for(int i=0;i<8;i++){ uint64_t u=hi[i]; uint64_t add = lo[i] + u*977 + c; lo[i]=add & 0xFFFFFFFFULL; c=add>>32; }
  uint64_t c2=0; for(int i=0;i<8;i++){ uint64_t u=(i==0)?0:hi[i-1]; uint64_t add = lo[i] + u + c2; lo[i]=add & 0xFFFFFFFFULL; c2=add>>32; }
  uint64_t carry=c+c2; for(int i=0;i<8;i++){ uint64_t add = lo[i] + (i==0?carry:0); lo[i]=add & 0xFFFFFFFFULL; carry=add>>32; }
  for(int i=0;i<8;i++) r.v[i]=(uint32_t)lo[i]; fe_mod_p(r);
}

__device__ inline void fe_mul(fe& r, const fe& a, const fe& b){ uint64_t t[16]; for(int i=0;i<16;i++) t[i]=0; for(int i=0;i<8;i++){ uint64_t c=0; for(int j=0;j<8;j++){ uint64_t z=t[i+j] + (uint64_t)a.v[i]*(uint64_t)b.v[j] + c; t[i+j]=z & 0xFFFFFFFFULL; c=z>>32; } t[i+8]+=c; } fe_reduce512(r,t); }
__device__ inline void fe_sqr(fe& r, const fe& a){ fe_mul(r,a,a); }

__device__ inline void fe_mul_u32(fe& r, const fe& a, uint32_t k){ uint64_t t[16]; for(int i=0;i<16;i++) t[i]=0; uint64_t c=0; for(int i=0;i<8;i++){ uint64_t z=(uint64_t)a.v[i]*k + c; t[i]=z & 0xFFFFFFFFULL; c=z>>32; } t[8]=c; fe_reduce512(r,t); }

__device__ inline void fe_inv(fe& r, const fe& a){ fe x; fe_cpy(x,a); fe y; fe_sqr(y,x); fe z; fe_mul(z,y,x); fe t1; fe_sqr(t1,z); fe t2; for(int i=0;i<2;i++){ fe_sqr(t1,t1); } fe_mul(t2,t1,z); fe t3; for(int i=0;i<5;i++){ fe_sqr(t2,t2);} fe_mul(t3,t2,t1); fe t4; for(int i=0;i<10;i++){ fe_sqr(t3,t3);} fe_mul(t4,t3,t2); fe t5; for(int i=0;i<20;i++){ fe_sqr(t4,t4);} fe_mul(t5,t4,t3); fe t6; for(int i=0;i<10;i++){ fe_sqr(t5,t5);} fe_mul(t6,t5,t2); fe t7; for(int i=0;i<50;i++){ fe_sqr(t6,t6);} fe_mul(t7,t6,t5); fe t8; for(int i=0;i<100;i++){ fe_sqr(t7,t7);} fe_mul(t8,t7,t6); fe t9; for(int i=0;i<5;i++){ fe_sqr(t8,t8);} fe_mul(t9,t8,t1); for(int i=0;i<2;i++){ fe_sqr(t9,t9);} fe_mul(r,t9,x); }

struct jp { fe X; fe Y; fe Z; };

__device__ inline void to_infinity(jp& P){ fe_zero(P.Z); }
__device__ inline int is_infinity(const jp& P){ for(int i=0;i<8;i++){ if(P.Z.v[i]!=0) return 0; } return 1; }

__constant__ uint32_t GX[8]={
  0x16F81798, 0x59F2815B, 0x2DCE28D9, 0x029BFCDB,
  0xCE870B07, 0x55A06295, 0xF9DCBBAC, 0x79BE667E
};
__constant__ uint32_t GY[8]={
  0xFB10D4B8, 0x9C47D08F, 0xA6855419, 0xFD17B448,
  0x0E1108A8, 0x5DA4FBFC, 0x26A3C465, 0x483ADA77
};

__device__ inline void set_affine_G(fe& x, fe& y){ for(int i=0;i<8;i++){ x.v[i]=GX[i]; y.v[i]=GY[i]; } }

__device__ inline void jp_add_affine(jp& R, const jp& P, const fe& x2, const fe& y2){ if(is_infinity(P)){ fe_cpy(R.X,(fe&)x2); fe_cpy(R.Y,(fe&)y2); fe_zero(R.Z); R.Z.v[0]=1; return; } fe Z1Z1; fe_sqr(Z1Z1,P.Z); fe U2; fe_mul(U2,(fe&)x2,Z1Z1); fe S2; fe_mul(S2,(fe&)y2,P.Z); fe_mul(S2,S2,Z1Z1); fe H; fe_sub(H,U2,P.X); fe HH; fe_sqr(HH,H); fe I; fe_mul_u32(I,HH,4); fe J; fe_mul(J,H,I); fe r; fe t; fe_sub(t,S2,P.Y); fe_mul_u32(r,t,2); fe V; fe_mul(V,P.X,I); fe X3; fe_sqr(X3,r); fe_sub(X3,X3,J); fe_sub(X3,X3,V); fe_sub(X3,X3,V); fe Y3; fe_sub(Y3,V,X3); fe_mul(Y3,Y3,r); fe T2; fe_mul_u32(T2,P.Y,2); fe_mul(T2,T2,J); fe_sub(Y3,Y3,T2); fe Z3; fe_add(Z3,P.Z,H); fe_sqr(Z3,Z3); fe_sub(Z3,Z3,Z1Z1); fe_sub(Z3,Z3,HH); fe_cpy(R.X,X3); fe_cpy(R.Y,Y3); fe_cpy(R.Z,Z3); }

__device__ inline void jp_double(jp& R, const jp& P){ fe XX; fe_sqr(XX,P.X); fe YY; fe_sqr(YY,P.Y); fe YYYY; fe_sqr(YYYY,YY); fe S; fe t; fe_mul_u32(S,P.X,4); fe_mul(S,S,YY); fe M; fe_mul_u32(M,XX,3); fe X3; fe_sqr(X3,M); fe t2; fe_mul_u32(t2,S,2); fe_sub(X3,X3,t2); fe Y3; fe_sub(Y3,S,X3); fe_mul(Y3,Y3,M); fe T; fe_mul_u32(T,YYYY,8); fe_sub(Y3,Y3,T); fe Z3; fe_mul_u32(Z3,P.Y,2); fe_mul(Z3,Z3,P.Z); fe_cpy(R.X,X3); fe_cpy(R.Y,Y3); fe_cpy(R.Z,Z3); }

__device__ inline void scalar_mul_G(jp& R, const uint8_t k[32]){ jp Q; to_infinity(Q); fe gx,gy; set_affine_G(gx,gy); for(int i=0;i<32;i++){ uint8_t byte=k[31-i]; for(int b=7;b>=0;b--){ if(!is_infinity(Q)) { jp D; jp_double(D,Q); Q=D; } else { } if((byte>>b)&1){ jp A; jp_add_affine(A,Q,gx,gy); Q=A; } } }
 R=Q; }

__device__ inline void jacobian_to_comp33(const jp& P, uint8_t out[33]){ fe zinv; fe_inv(zinv,P.Z); fe zinv2; fe_sqr(zinv2,zinv); fe zinv3; fe_mul(zinv3,zinv2,zinv); fe x; fe_mul(x,P.X,zinv2); fe y; fe_mul(y,P.Y,zinv3); uint8_t prefix = (y.v[0] & 1) ? 0x03 : 0x02; out[0]=prefix; for(int i=0;i<8;i++){ uint32_t w=x.v[7-i]; out[1+i*4+0]=(uint8_t)(w>>24); out[1+i*4+1]=(uint8_t)(w>>16); out[1+i*4+2]=(uint8_t)(w>>8); out[1+i*4+3]=(uint8_t)(w); }
}

__device__ inline void priv_add_uint(uint8_t k[32], uint32_t inc){ uint32_t c=inc; for(int i=31;i>=0;i--){ uint32_t t=(uint32_t)k[i] + (c & 0xFF); k[i]=(uint8_t)t; c=(c>>8) + (t>>8); if(c==0) break; } }

static __device__ inline uint32_t rotr32(uint32_t x, uint32_t n){return (x>>n)|(x<<(32-n));}
static __device__ inline uint32_t ch(uint32_t x,uint32_t y,uint32_t z){return (x & y) ^ (~x & z);} 
static __device__ inline uint32_t maj(uint32_t x,uint32_t y,uint32_t z){return (x & y) ^ (x & z) ^ (y & z);} 
static __device__ inline uint32_t bigsig0(uint32_t x){return rotr32(x,2)^rotr32(x,13)^rotr32(x,22);} 
static __device__ inline uint32_t bigsig1(uint32_t x){return rotr32(x,6)^rotr32(x,11)^rotr32(x,25);} 
static __device__ inline uint32_t small0(uint32_t x){return rotr32(x,7)^rotr32(x,18) ^ (x>>3);} 
static __device__ inline uint32_t small1(uint32_t x){return rotr32(x,17)^rotr32(x,19) ^ (x>>10);} 
static __device__ const uint32_t K256[64]={
0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,0x3956c25b,0x59f111f1,0x923f82a4,0xab1c5ed5,
0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,
0xe49b69c1,0xefbe4786,0x0fc19dc6,0x240ca1cc,0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,
0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,0xc6e00bf3,0xd5a79147,0x06ca6351,0x14292967,
0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,
0xa2bfe8a1,0xa81a664b,0xc24b8b70,0xc76c51a3,0xd192e819,0xd6990624,0xf40e3585,0x106aa070,
0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,0x391c0cb3,0x4ed8aa4a,0x5b9cca4f,0x682e6ff3,
0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2};

__device__ void sha256_33(const uint8_t* in, uint8_t out[32]){
  uint8_t B[64];
  for(int i=0;i<33;i++) B[i]=in[i];
  B[33]=0x80; for(int i=34;i<56;i++) B[i]=0; B[56]=0;B[57]=0;B[58]=0;B[59]=0;B[60]=0;B[61]=0;B[62]=0x01;B[63]=0x08;
  uint32_t W[64];
  for(int i=0;i<16;i++){ int j=i*4; W[i]=((uint32_t)B[j]<<24)|((uint32_t)B[j+1]<<16)|((uint32_t)B[j+2]<<8)|((uint32_t)B[j+3]); }
  for(int i=16;i<64;i++) W[i]=small1(W[i-2])+W[i-7]+small0(W[i-15])+W[i-16];
  uint32_t a=0x6a09e667,b=0xbb67ae85,c=0x3c6ef372,d=0xa54ff53a,e=0x510e527f,f=0x9b05688c,g=0x1f83d9ab,h=0x5be0cd19;
  for(int i=0;i<64;i++){ uint32_t t1=h + bigsig1(e) + ch(e,f,g) + K256[i] + W[i]; uint32_t t2=bigsig0(a) + maj(a,b,c); h=g;g=f;f=e;e=d+t1;d=c;c=b;b=a;a=t1+t2; }
  uint32_t H0=0x6a09e667+a,H1=0xbb67ae85+b,H2=0x3c6ef372+c,H3=0xa54ff53a+d,H4=0x510e527f+e,H5=0x9b05688c+f,H6=0x1f83d9ab+g,H7=0x5be0cd19+h;
  uint32_t S[8]={H0,H1,H2,H3,H4,H5,H6,H7};
  for(int i=0;i<8;i++){ out[i*4+0]=(uint8_t)(S[i]>>24); out[i*4+1]=(uint8_t)(S[i]>>16); out[i*4+2]=(uint8_t)(S[i]>>8); out[i*4+3]=(uint8_t)S[i]; }
}

__device__ inline uint32_t rol32(uint32_t x, uint32_t n){return (x<<n)|(x>>(32-n));}

__device__ void ripemd160_32(const uint8_t* in, uint8_t out[20]){
  uint32_t X[16]; for(int i=0;i<16;i++){ int j=i*4; if(i<8){ uint32_t w=((uint32_t)in[j]) | ((uint32_t)in[j+1]<<8) | ((uint32_t)in[j+2]<<16) | ((uint32_t)in[j+3]<<24); X[i]=w; } else if(i==8){ X[i]=0x00000080; } else if(i<14){ X[i]=0; } else if(i==14){ X[i]=256; } else { X[i]=0; } }
  uint32_t h0=0x67452301,h1=0xefcdab89,h2=0x98badcfe,h3=0x10325476,h4=0xc3d2e1f0; uint32_t a=h0,b=h1,c=h2,d=h3,e=h4;
  auto F=[&](uint32_t x,uint32_t y,uint32_t z){return x^y^z;}; auto G=[&](uint32_t x,uint32_t y,uint32_t z){return (x & y) | (~x & z);}; auto H=[&](uint32_t x,uint32_t y,uint32_t z){return (x | ~y) ^ z;}; auto I=[&](uint32_t x,uint32_t y,uint32_t z){return (x & z) | (y & ~z);}; auto J=[&](uint32_t x,uint32_t y,uint32_t z){return x ^ (y | ~z);}; auto R=[&](uint32_t &aa,uint32_t bb,uint32_t cc,uint32_t dd,uint32_t &ee,uint32_t ff,uint32_t kk,uint32_t ss){ aa=rol32(aa + ff + kk, ss) + ee; cc=rol32(cc,10); ee=dd; dd=cc; cc=bb; bb=aa; };
  int rl[5][16]={{11,14,15,12,5,8,7,9,11,13,14,15,6,7,9,8},{7,6,8,13,11,9,7,15,7,12,15,9,11,7,13,12},{11,13,6,7,14,9,13,15,14,8,13,6,5,12,7,5},{11,12,14,15,14,15,9,8,9,14,5,6,8,6,5,12},{9,15,5,11,6,8,13,12,5,12,13,14,11,8,5,6}};
  int rr[5][16]={{8,9,9,11,13,15,15,5,7,7,8,11,14,14,12,6},{9,13,15,7,12,8,9,11,7,7,12,7,6,15,13,11},{9,7,15,11,8,6,6,14,12,13,5,14,13,13,7,5},{15,5,8,11,14,14,6,14,6,9,12,9,12,5,15,8},{8,5,12,9,12,5,14,6,8,13,6,5,15,13,11,11}};
  int zl[5][16]={{0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15},{7,4,13,1,10,6,15,3,12,0,9,5,2,14,11,8},{3,10,14,4,9,15,8,1,2,7,0,6,13,11,5,12},{1,9,11,10,0,8,12,4,13,3,7,15,14,5,6,2},{4,0,5,9,7,12,2,10,14,1,3,8,11,6,15,13}};
  int zr[5][16]={{5,14,7,0,9,2,11,4,13,6,15,8,1,10,3,12},{6,11,3,7,0,13,5,10,14,15,8,12,4,9,1,2},{15,5,1,3,7,14,6,9,11,8,12,2,10,0,4,13},{8,6,4,1,3,11,15,0,5,12,2,13,9,7,10,14},{12,15,10,4,1,5,8,7,6,2,13,14,0,3,9,11}};
  uint32_t al=a,bl=b,cl=c,dl=d,el=e; for(int j=0;j<16;j++) R(al,bl,cl,dl,el,F(bl,cl,dl)+X[zl[0][j]],0x00000000,rl[0][j]); for(int j=0;j<16;j++) R(al,bl,cl,dl,el,G(bl,cl,dl)+X[zl[1][j]],0x5a827999,rl[1][j]); for(int j=0;j<16;j++) R(al,bl,cl,dl,el,H(bl,cl,dl)+X[zl[2][j]],0x6ed9eba1,rl[2][j]); for(int j=0;j<16;j++) R(al,bl,cl,dl,el,I(bl,cl,dl)+X[zl[3][j]],0x8f1bbcdc,rl[3][j]); for(int j=0;j<16;j++) R(al,bl,cl,dl,el,J(bl,cl,dl)+X[zl[4][j]],0xa953fd4e,rl[4][j]);
  uint32_t ar=a,br=b,cr=c,dr=d,er=e; for(int j=0;j<16;j++) R(ar,br,cr,dr,er,J(br,cr,dr)+X[zr[0][j]],0x50a28be6,rr[0][j]); for(int j=0;j<16;j++) R(ar,br,cr,dr,er,I(br,cr,dr)+X[zr[1][j]],0x5c4dd124,rr[1][j]); for(int j=0;j<16;j++) R(ar,br,cr,dr,er,H(br,cr,dr)+X[zr[2][j]],0x6d703ef3,rr[2][j]); for(int j=0;j<16;j++) R(ar,br,cr,dr,er,G(br,cr,dr)+X[zr[3][j]],0x7a6d76e9,rr[3][j]); for(int j=0;j<16;j++) R(ar,br,cr,dr,er,F(br,cr,dr)+X[zr[4][j]],0x00000000,rr[4][j]);
  uint32_t t=h1 + cl + dr; h1 = h2 + dl + er; h2 = h3 + el + ar; h3 = h4 + al + br; h4 = h0 + bl + cr; h0 = t; uint32_t HH[5]={h0,h1,h2,h3,h4}; for(int i=0;i<5;i++){ out[i*4+0]=(uint8_t)(HH[i]&0xff); out[i*4+1]=(uint8_t)((HH[i]>>8)&0xff); out[i*4+2]=(uint8_t)((HH[i]>>16)&0xff); out[i*4+3]=(uint8_t)((HH[i]>>24)&0xff); }
}

__global__ void pubkey_hash_kernel(const uint8_t* base_priv, uint8_t* out_pub33, uint8_t* out_hash20, size_t n){
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if(i>=n) return;
  uint8_t k[32]; for(int j=0;j<32;j++) k[j]=base_priv[j]; priv_add_uint(k,(uint32_t)i);
  jp P; scalar_mul_G(P,k);
  uint8_t cpk[33]; jacobian_to_comp33(P,cpk);
  for(int j=0;j<33;j++) out_pub33[i*33+j]=cpk[j];
  uint8_t s[32]; sha256_33(cpk,s); uint8_t r[20]; ripemd160_32(s,r);
  for(int j=0;j<20;j++) out_hash20[i*20+j]=r[j];
}

static uint8_t* D_BASE = 0;
static uint8_t* D_PUB = 0;
static uint8_t* D_HASH = 0;
static size_t CAP_N = 0;

extern "C" void gpu_pubkey_hash_from_priv_batch(const uint8_t* base_priv, uint8_t* out_pub33, uint8_t* out_hash20, size_t n){
  if (CAP_N < n) {
    if (D_PUB) cudaFree(D_PUB);
    if (D_HASH) cudaFree(D_HASH);
    cudaMalloc((void**)&D_PUB, n*33);
    cudaMalloc((void**)&D_HASH, n*20);
    CAP_N = n;
  }
  if (!D_BASE) cudaMalloc((void**)&D_BASE, 32);
  cudaMemcpy(D_BASE, base_priv, 32, cudaMemcpyHostToDevice);
  int dev=0; cudaDeviceProp prop; cudaGetDeviceProperties(&prop,dev);
  int mp = prop.multiProcessorCount;
  int blockSize = 512;
  int gridSize = (int)((n + blockSize - 1)/blockSize);
  int minGrid = mp * 4; if (gridSize < minGrid) gridSize = minGrid;
  dim3 block(blockSize);
  dim3 grid(gridSize);
  pubkey_hash_kernel<<<grid,block>>>(D_BASE, D_PUB, D_HASH, n);
  cudaDeviceSynchronize();
  cudaMemcpy(out_pub33, D_PUB, n*33, cudaMemcpyDeviceToHost);
  cudaMemcpy(out_hash20, D_HASH, n*20, cudaMemcpyDeviceToHost);
}

extern "C" void gpu_pubkey_hash_from_priv_batch_stream(const uint8_t* base_priv, uint8_t* out_pub33, uint8_t* out_hash20, size_t n, cudaStream_t stream){
  if (CAP_N < n) {
    if (D_PUB) cudaFree(D_PUB);
    if (D_HASH) cudaFree(D_HASH);
    cudaMalloc((void**)&D_PUB, n*33);
    cudaMalloc((void**)&D_HASH, n*20);
    CAP_N = n;
  }
  if (!D_BASE) cudaMalloc((void**)&D_BASE, 32);
  cudaMemcpyAsync(D_BASE, base_priv, 32, cudaMemcpyHostToDevice, stream);
  int dev=0; cudaDeviceProp prop; cudaGetDeviceProperties(&prop,dev);
  int mp = prop.multiProcessorCount;
  int blockSize = 512;
  int gridSize = (int)((n + blockSize - 1)/blockSize);
  int minGrid = mp * 4; if (gridSize < minGrid) gridSize = minGrid;
  pubkey_hash_kernel<<<gridSize,blockSize,0,stream>>>(D_BASE, D_PUB, D_HASH, n);
  cudaMemcpyAsync(out_pub33, D_PUB, n*33, cudaMemcpyDeviceToHost, stream);
  cudaMemcpyAsync(out_hash20, D_HASH, n*20, cudaMemcpyDeviceToHost, stream);
}
