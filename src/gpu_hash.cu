#include <cuda_runtime.h>
#include <stdint.h>

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

static __device__ void sha256_33(const uint8_t* in, uint8_t out[32]){
  uint8_t B[64];
  for(int i=0;i<33;i++) B[i]=in[i];
  B[33]=0x80;
  for(int i=34;i<56;i++) B[i]=0;
  B[56]=0;B[57]=0;B[58]=0;B[59]=0;B[60]=0;B[61]=0;B[62]=0x01;B[63]=0x08;
  uint32_t W[64];
  for(int i=0;i<16;i++){
    int j=i*4;W[i]=((uint32_t)B[j]<<24)|((uint32_t)B[j+1]<<16)|((uint32_t)B[j+2]<<8)|((uint32_t)B[j+3]);
  }
  for(int i=16;i<64;i++) W[i]=small1(W[i-2])+W[i-7]+small0(W[i-15])+W[i-16];
  uint32_t a=0x6a09e667,b=0xbb67ae85,c=0x3c6ef372,d=0xa54ff53a,e=0x510e527f,f=0x9b05688c,g=0x1f83d9ab,h=0x5be0cd19;
  for(int i=0;i<64;i++){
    uint32_t t1=h + bigsig1(e) + ch(e,f,g) + K256[i] + W[i];
    uint32_t t2=bigsig0(a) + maj(a,b,c);
    h=g;g=f;f=e;e=d+t1;d=c;c=b;b=a;a=t1+t2;
  }
  uint32_t H0=0x6a09e667+a,H1=0xbb67ae85+b,H2=0x3c6ef372+c,H3=0xa54ff53a+d,H4=0x510e527f+e,H5=0x9b05688c+f,H6=0x1f83d9ab+g,H7=0x5be0cd19+h;
  uint32_t S[8]={H0,H1,H2,H3,H4,H5,H6,H7};
  for(int i=0;i<8;i++){ out[i*4+0]=(uint8_t)(S[i]>>24); out[i*4+1]=(uint8_t)(S[i]>>16); out[i*4+2]=(uint8_t)(S[i]>>8); out[i*4+3]=(uint8_t)S[i]; }
}

static __device__ inline uint32_t rol32(uint32_t x, uint32_t n){return (x<<n)|(x>>(32-n));}

static __device__ void ripemd160_32(const uint8_t* in, uint8_t out[20]){
  uint32_t X[16];
  for(int i=0;i<16;i++){
    int j=i*4; if(i<8){ uint32_t w=((uint32_t)in[j]) | ((uint32_t)in[j+1]<<8) | ((uint32_t)in[j+2]<<16) | ((uint32_t)in[j+3]<<24); X[i]=w; }
    else if(i==8){ X[i]=0x00000080; }
    else if(i<14){ X[i]=0; }
    else if(i==14){ X[i]=256; }
    else { X[i]=0; }
  }
  uint32_t h0=0x67452301,h1=0xefcdab89,h2=0x98badcfe,h3=0x10325476,h4=0xc3d2e1f0;
  uint32_t a=h0,b=h1,c=h2,d=h3,e=h4;
  auto F=[&](uint32_t x,uint32_t y,uint32_t z){return x^y^z;};
  auto G=[&](uint32_t x,uint32_t y,uint32_t z){return (x & y) | (~x & z);};
  auto H=[&](uint32_t x,uint32_t y,uint32_t z){return (x | ~y) ^ z;};
  auto I=[&](uint32_t x,uint32_t y,uint32_t z){return (x & z) | (y & ~z);};
  auto J=[&](uint32_t x,uint32_t y,uint32_t z){return x ^ (y | ~z);};
  auto R=[&](uint32_t &a,uint32_t b,uint32_t c,uint32_t d,uint32_t &e,uint32_t f,uint32_t k,uint32_t s){ a=rol32(a + f + k, s) + e; c=rol32(c,10); e=d; d=c; c=b; b=a; };
  int rl[5][16]={{11,14,15,12,5,8,7,9,11,13,14,15,6,7,9,8},{7,6,8,13,11,9,7,15,7,12,15,9,11,7,13,12},{11,13,6,7,14,9,13,15,14,8,13,6,5,12,7,5},{11,12,14,15,14,15,9,8,9,14,5,6,8,6,5,12},{9,15,5,11,6,8,13,12,5,12,13,14,11,8,5,6}};
  int rr[5][16]={{8,9,9,11,13,15,15,5,7,7,8,11,14,14,12,6},{9,13,15,7,12,8,9,11,7,7,12,7,6,15,13,11},{9,7,15,11,8,6,6,14,12,13,5,14,13,13,7,5},{15,5,8,11,14,14,6,14,6,9,12,9,12,5,15,8},{8,5,12,9,12,5,14,6,8,13,6,5,15,13,11,11}};
  int zl[5][16]={{0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15},{7,4,13,1,10,6,15,3,12,0,9,5,2,14,11,8},{3,10,14,4,9,15,8,1,2,7,0,6,13,11,5,12},{1,9,11,10,0,8,12,4,13,3,7,15,14,5,6,2},{4,0,5,9,7,12,2,10,14,1,3,8,11,6,15,13}};
  int zr[5][16]={{5,14,7,0,9,2,11,4,13,6,15,8,1,10,3,12},{6,11,3,7,0,13,5,10,14,15,8,12,4,9,1,2},{15,5,1,3,7,14,6,9,11,8,12,2,10,0,4,13},{8,6,4,1,3,11,15,0,5,12,2,13,9,7,10,14},{12,15,10,4,1,5,8,7,6,2,13,14,0,3,9,11}};
  uint32_t al=a,bl=b,cl=c,dl=d,el=e;
  for(int j=0;j<16;j++) R(al,bl,cl,dl,el,F(bl,cl,dl)+X[zl[0][j]],0x00000000,rl[0][j]);
  for(int j=0;j<16;j++) R(al,bl,cl,dl,el,G(bl,cl,dl)+X[zl[1][j]],0x5a827999,rl[1][j]);
  for(int j=0;j<16;j++) R(al,bl,cl,dl,el,H(bl,cl,dl)+X[zl[2][j]],0x6ed9eba1,rl[2][j]);
  for(int j=0;j<16;j++) R(al,bl,cl,dl,el,I(bl,cl,dl)+X[zl[3][j]],0x8f1bbcdc,rl[3][j]);
  for(int j=0;j<16;j++) R(al,bl,cl,dl,el,J(bl,cl,dl)+X[zl[4][j]],0xa953fd4e,rl[4][j]);
  uint32_t ar=a,br=b,cr=c,dr=d,er=e;
  for(int j=0;j<16;j++) R(ar,br,cr,dr,er,J(br,cr,dr)+X[zr[0][j]],0x50a28be6,rr[0][j]);
  for(int j=0;j<16;j++) R(ar,br,cr,dr,er,I(br,cr,dr)+X[zr[1][j]],0x5c4dd124,rr[1][j]);
  for(int j=0;j<16;j++) R(ar,br,cr,dr,er,H(br,cr,dr)+X[zr[2][j]],0x6d703ef3,rr[2][j]);
  for(int j=0;j<16;j++) R(ar,br,cr,dr,er,G(br,cr,dr)+X[zr[3][j]],0x7a6d76e9,rr[3][j]);
  for(int j=0;j<16;j++) R(ar,br,cr,dr,er,F(br,cr,dr)+X[zr[4][j]],0x00000000,rr[4][j]);
  uint32_t t=h1 + cl + dr; h1 = h2 + dl + er; h2 = h3 + el + ar; h3 = h4 + al + br; h4 = h0 + bl + cr; h0 = t;
  uint32_t HH[5]={h0,h1,h2,h3,h4};
  for(int i=0;i<5;i++){ out[i*4+0]=(uint8_t)(HH[i]&0xff); out[i*4+1]=(uint8_t)((HH[i]>>8)&0xff); out[i*4+2]=(uint8_t)((HH[i]>>16)&0xff); out[i*4+3]=(uint8_t)((HH[i]>>24)&0xff); }
}

static __global__ void hash160_kernel(const uint8_t* pubkeys33, uint8_t* out_hashes20, size_t n){
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if(i>=n) return;
  const uint8_t* pk = pubkeys33 + i*33;
  uint8_t s[32];
  sha256_33(pk,s);
  uint8_t r[20];
  ripemd160_32(s,r);
  uint8_t* o = out_hashes20 + i*20;
  for(int k=0;k<20;k++) o[k]=r[k];
}

extern "C" void gpu_hash160_batch(const uint8_t* pubkeys33, uint8_t* out_hashes20, size_t n){
  uint8_t* d_in=0; uint8_t* d_out=0;
  cudaMalloc((void**)&d_in, n*33);
  cudaMalloc((void**)&d_out, n*20);
  cudaMemcpy(d_in, pubkeys33, n*33, cudaMemcpyHostToDevice);
  dim3 block(256);
  dim3 grid((unsigned)((n + block.x - 1)/block.x));
  hash160_kernel<<<grid,block>>>(d_in,d_out,n);
  cudaDeviceSynchronize();
  cudaMemcpy(out_hashes20, d_out, n*20, cudaMemcpyDeviceToHost);
  cudaFree(d_in);
  cudaFree(d_out);
}
