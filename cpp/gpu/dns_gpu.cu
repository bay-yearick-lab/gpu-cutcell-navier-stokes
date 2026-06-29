// dns_gpu.cu
//
// GPU port of the 3d_to_1d B-spline mimetic apply (periodic cube), one CUDA thread
// per 1D line, exactly replicating the CPU FastDeriv reduce-solve-expand:
//   solveB  = banded-LU + Sherman-Morrison-Woodbury cyclic solve  (B^{-1})
//   matvec  = circulant stencil  (cBp for d1, cBpp for d2)
// plus the discrete pressure projection via cuFFT (consistent div.grad, diagonal in
// Fourier space with symbol = sum_axes (BMC d1 symbol)^2 from the same stencils).
// All factors built ONCE on host (verified C++ FastDeriv), copied read-only to the
// device.  Every operator diff-gated against the CPU Deriv3D to roundoff; the
// projection gated by div(u^{n+1}) -> roundoff.
//
// Build (sm_120):
//   make -C cpp gpu        (see cpp/gpu/Makefile; ARCH=sm_XX overrides sm_120)

#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cmath>
#include <complex>
#include <chrono>
#include <cuda_runtime.h>
#include <cufft.h>

#include "bm_apply.h"
#include "bm3d.h"
#include "linalg.h"

#define CK(x) do{ cudaError_t e_=(x); if(e_!=cudaSuccess){ \
    printf("CUDA error %s:%d : %s\n",__FILE__,__LINE__,cudaGetErrorString(e_)); std::exit(1);} }while(0)
#define FK(x) do{ cufftResult r_=(x); if(r_!=CUFFT_SUCCESS){ \
    printf("cuFFT error %s:%d : %d\n",__FILE__,__LINE__,(int)r_); std::exit(1);} }while(0)

static constexpr int MAXN = 513, MAXM = 40;

struct DevAxis {
    int n=0,w=0,kl=0,m=0,sbw=0;
    double *band=nullptr,*Dss=nullptr,*Minv=nullptr,*Z=nullptr,*sten1=nullptr,*sten2=nullptr;
    int *S=nullptr;
};

__device__ __forceinline__ double band_at(const double* b,int w,int kl,int i,int j){ return b[(size_t)i*w+(j-i+kl)]; }

__device__ void bmc_line(double* xl,const DevAxis A,int order,double* coeff){
    const int n=A.n,kl=A.kl,ku=A.kl,w=A.w,m=A.m,sbw=A.sbw;
    for(int i=0;i<n;++i){int jlo=(i-kl>0)?i-kl:0;double s=xl[i];for(int j=jlo;j<i;++j)s-=band_at(A.band,w,kl,i,j)*xl[j];xl[i]=s;}
    for(int i=n-1;i>=0;--i){int jhi=(i+ku<n-1)?i+ku:n-1;double s=xl[i];for(int j=i+1;j<=jhi;++j)s-=band_at(A.band,w,kl,i,j)*xl[j];xl[i]=s/band_at(A.band,w,kl,i,i);}
    double ys[MAXM],rhs[MAXM],wv[MAXM];
    for(int r=0;r<m;++r) ys[r]=xl[A.S[r]];
    for(int r=0;r<m;++r){double s=0;for(int k=0;k<m;++k)s+=A.Dss[(size_t)r*m+k]*ys[k];rhs[r]=s;}
    for(int r=0;r<m;++r){double s=0;for(int k=0;k<m;++k)s+=A.Minv[(size_t)r*m+k]*rhs[k];wv[r]=s;}
    for(int c=0;c<m;++c){const double wc=wv[c];const double* zc=&A.Z[(size_t)c*n];for(int i=0;i<n;++i)xl[i]-=zc[i]*wc;}
    for(int i=0;i<n;++i) coeff[i]=xl[i];
    const double* st=(order==1)?A.sten1:A.sten2;
    for(int i=0;i<n;++i){double acc=0;for(int o=-sbw;o<=sbw;++o){int j=i+o;if(j<0)j+=n;else if(j>=n)j-=n;acc+=st[o+sbw]*coeff[j];}xl[i]=acc;}
}

__global__ void deriv_axis_kernel(const double* in,double* out,int nx,int ny,int nz,int axis,DevAxis A,int order,int acc){
    int n,stride;long long nlines;
    if(axis==0){n=nx;stride=1;nlines=(long long)ny*nz;}
    else if(axis==1){n=ny;stride=nx;nlines=(long long)nx*nz;}
    else{n=nz;stride=(long long)nx*ny;nlines=(long long)nx*ny;}
    long long L=(long long)blockIdx.x*blockDim.x+threadIdx.x; if(L>=nlines)return;
    size_t base; if(axis==0)base=(size_t)L*nx; else if(axis==1)base=(size_t)(L/nx)*nx*ny+(L%nx); else base=(size_t)(L/nx)*nx+(L%nx);
    double xl[MAXN],coeff[MAXN];
    for(int i=0;i<n;++i) xl[i]=in[base+(size_t)i*stride];
    bmc_line(xl,A,order,coeff);
    if(acc) for(int i=0;i<n;++i) out[base+(size_t)i*stride]+=xl[i];
    else    for(int i=0;i<n;++i) out[base+(size_t)i*stride] =xl[i];
}

// cuFFT helper kernels
__global__ void r2c(const double* r,cufftDoubleComplex* c,size_t N){size_t i=(size_t)blockIdx.x*blockDim.x+threadIdx.x;if(i<N){c[i].x=r[i];c[i].y=0;}}
__global__ void cmul_real(cufftDoubleComplex* c,const double* s,size_t N){size_t i=(size_t)blockIdx.x*blockDim.x+threadIdx.x;if(i<N){c[i].x*=s[i];c[i].y*=s[i];}}
__global__ void c2r(const cufftDoubleComplex* c,double* r,double sc,size_t N){size_t i=(size_t)blockIdx.x*blockDim.x+threadIdx.x;if(i<N)r[i]=c[i].x*sc;}
__global__ void fsub(double* u,const double* g,size_t N){size_t i=(size_t)blockIdx.x*blockDim.x+threadIdx.x;if(i<N)u[i]-=g[i];}

static DevAxis upload_axis(const FastDeriv& f){
    const CyclicSolver& cyc=f.Bcyc; const BandSolver& core=cyc.core;
    DevAxis h; h.n=core.n;h.kl=core.kl;h.w=core.w;h.m=cyc.m;h.sbw=f.cBpp.bw;
    if(h.m>MAXM||h.n>MAXN){printf("cap exceeded\n");std::exit(1);}
    DenseMatrix Minv(h.m); {std::vector<double> e(h.m,0.0),x;for(int c=0;c<h.m;++c){for(int r=0;r<h.m;++r)e[r]=(r==c)?1.0:0.0;dense_lu_solve(cyc.M,e,x);for(int r=0;r<h.m;++r)Minv(r,c)=x[r];}}
    std::vector<double> mr((size_t)h.m*h.m); for(int r=0;r<h.m;++r)for(int c=0;c<h.m;++c)mr[(size_t)r*h.m+c]=Minv(r,c);
    DevAxis d=h;
    CK(cudaMalloc(&d.band,(size_t)h.n*h.w*sizeof(double)));CK(cudaMemcpy(d.band,core.band.data(),(size_t)h.n*h.w*sizeof(double),cudaMemcpyHostToDevice));
    CK(cudaMalloc(&d.S,(size_t)h.m*sizeof(int)));CK(cudaMemcpy(d.S,cyc.S.data(),(size_t)h.m*sizeof(int),cudaMemcpyHostToDevice));
    CK(cudaMalloc(&d.Dss,(size_t)h.m*h.m*sizeof(double)));CK(cudaMemcpy(d.Dss,cyc.Dss.data(),(size_t)h.m*h.m*sizeof(double),cudaMemcpyHostToDevice));
    CK(cudaMalloc(&d.Minv,(size_t)h.m*h.m*sizeof(double)));CK(cudaMemcpy(d.Minv,mr.data(),(size_t)h.m*h.m*sizeof(double),cudaMemcpyHostToDevice));
    CK(cudaMalloc(&d.Z,(size_t)h.m*h.n*sizeof(double)));CK(cudaMemcpy(d.Z,cyc.Z.data(),(size_t)h.m*h.n*sizeof(double),cudaMemcpyHostToDevice));
    CK(cudaMalloc(&d.sten1,(size_t)(2*h.sbw+1)*sizeof(double)));CK(cudaMemcpy(d.sten1,f.cBp.st.data(),(size_t)(2*h.sbw+1)*sizeof(double),cudaMemcpyHostToDevice));
    CK(cudaMalloc(&d.sten2,(size_t)(2*h.sbw+1)*sizeof(double)));CK(cudaMemcpy(d.sten2,f.cBpp.st.data(),(size_t)(2*h.sbw+1)*sizeof(double),cudaMemcpyHostToDevice));
    return d;
}

static double relerr(const Field3D& a,const Field3D& b){double mx=0,me=0;for(size_t i=0;i<a.size();++i){mx=fmax(mx,fabs(b.a[i]));me=fmax(me,fabs(a.a[i]-b.a[i]));}return me/(mx+1e-300);}
static double maxabs(const Field3D& a){double m=0;for(size_t i=0;i<a.size();++i)m=fmax(m,fabs(a.a[i]));return m;}

int main(int argc,char**argv){
    int N=(argc>1)?atoi(argv[1]):128; int p=(argc>2)?atoi(argv[2]):4; const double L=2.0*M_PI;
    printf("GPU BMC apply + projection: N=%d p=%d (periodic cube)\n",N,p);
    FastDeriv f=FastDeriv::build(true,N,p,L); DevAxis dA=upload_axis(f);
    Deriv3D D(N,N,N,p,L,L,L,true);
    int T=64; long long nl=(long long)N*N; auto B=[&](){return (int)((nl+T-1)/T);};
    size_t NN=(size_t)N*N*N, bytes=NN*sizeof(double);

    Field3D sx(N,N,N),sy(N,N,N),sz(N,N,N);
    for(int k=0;k<N;++k)for(int j=0;j<N;++j)for(int i=0;i<N;++i){double x=L*i/N,y=L*j/N,z=L*k/N;
        sx.a[sx.idx(i,j,k)]=sin(x)*cos(2*y)*sin(3*z);
        sy.a[sy.idx(i,j,k)]=cos(2*x)*sin(y)*cos(z);
        sz.a[sz.idx(i,j,k)]=sin(x)*sin(y)*sin(2*z);}
    double *d_sx,*d_sy,*d_sz,*d_o,*d_t; CK(cudaMalloc(&d_sx,bytes));CK(cudaMalloc(&d_sy,bytes));CK(cudaMalloc(&d_sz,bytes));CK(cudaMalloc(&d_o,bytes));CK(cudaMalloc(&d_t,bytes));
    CK(cudaMemcpy(d_sx,sx.a.data(),bytes,cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_sy,sy.a.data(),bytes,cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_sz,sz.a.data(),bytes,cudaMemcpyHostToDevice));
    Field3D g(N,N,N);

    deriv_axis_kernel<<<B(),T>>>(d_sx,d_o,N,N,N,0,dA,2,0);
    deriv_axis_kernel<<<B(),T>>>(d_sx,d_o,N,N,N,1,dA,2,1);
    deriv_axis_kernel<<<B(),T>>>(d_sx,d_o,N,N,N,2,dA,2,1); CK(cudaDeviceSynchronize());
    CK(cudaMemcpy(g.a.data(),d_o,bytes,cudaMemcpyDeviceToHost));
    {Field3D cpu(N,N,N);D.laplacian_generic(sx,cpu,BMC_D2);double e=relerr(g,cpu);printf("  laplacian   rel %.3e  %s\n",e,(e<1e-10)?"PASS":"FAIL");}
    {Field3D gx(N,N,N),gy(N,N,N),gz(N,N,N),cx(N,N,N),cy(N,N,N),cz(N,N,N);
     deriv_axis_kernel<<<B(),T>>>(d_sx,d_o,N,N,N,0,dA,1,0);CK(cudaDeviceSynchronize());CK(cudaMemcpy(gx.a.data(),d_o,bytes,cudaMemcpyDeviceToHost));
     deriv_axis_kernel<<<B(),T>>>(d_sx,d_o,N,N,N,1,dA,1,0);CK(cudaDeviceSynchronize());CK(cudaMemcpy(gy.a.data(),d_o,bytes,cudaMemcpyDeviceToHost));
     deriv_axis_kernel<<<B(),T>>>(d_sx,d_o,N,N,N,2,dA,1,0);CK(cudaDeviceSynchronize());CK(cudaMemcpy(gz.a.data(),d_o,bytes,cudaMemcpyDeviceToHost));
     D.grad(sx,cx,cy,cz);double e=fmax(fmax(relerr(gx,cx),relerr(gy,cy)),relerr(gz,cz));printf("  gradient    rel %.3e  %s\n",e,(e<1e-10)?"PASS":"FAIL");}

    // ---------- cuFFT discrete projection ----------
    // EXACT consistent div.grad symbol: the eigenvalues of the PHYSICAL operator
    // L = sum_a D1_a^2, obtained as FFT(L applied to a unit impulse).  This equals
    // the apply-convention circulant symbol exactly, including the Nyquist mode
    // where the even-degree value operator is near-singular and an analytic
    // stencil-ratio symbol (sym_cBp/sym_cB) mishandles it.
    cufftDoubleComplex* d_c; CK(cudaMalloc(&d_c,NN*sizeof(cufftDoubleComplex)));
    cufftHandle plan; FK(cufftPlan3d(&plan,N,N,N,CUFFT_Z2Z));
    size_t TT=256; auto Bn=[&](size_t n){return (unsigned)((n+TT-1)/TT);};
    double* d_invL; CK(cudaMalloc(&d_invL,bytes));
    { double* d_imp; CK(cudaMalloc(&d_imp,bytes)); CK(cudaMemset(d_imp,0,bytes));
      double one=1.0; CK(cudaMemcpy(d_imp,&one,sizeof(double),cudaMemcpyHostToDevice));
      double* d_lap; CK(cudaMalloc(&d_lap,bytes)); CK(cudaMemset(d_lap,0,bytes));
      for(int a=0;a<3;++a){ deriv_axis_kernel<<<B(),T>>>(d_imp,d_t,N,N,N,a,dA,1,0); CK(cudaDeviceSynchronize());
        deriv_axis_kernel<<<B(),T>>>(d_t,d_lap,N,N,N,a,dA,1,1); CK(cudaDeviceSynchronize()); }
      r2c<<<Bn(NN),TT>>>(d_lap,d_c,NN); CK(cudaDeviceSynchronize());
      FK(cufftExecZ2Z(plan,d_c,d_c,CUFFT_FORWARD));
      std::vector<cufftDoubleComplex> hc(NN); CK(cudaMemcpy(hc.data(),d_c,NN*sizeof(cufftDoubleComplex),cudaMemcpyDeviceToHost));
      std::vector<double> invL(NN); for(size_t i=0;i<NN;++i) invL[i]=(i==0)?0.0:1.0/hc[i].x;
      CK(cudaMemcpy(d_invL,invL.data(),bytes,cudaMemcpyHostToDevice));
      cudaFree(d_imp); cudaFree(d_lap); }

    // u* = (sx,sy,sz) is not div-free.  Project: u^{n+1} = u* - grad phi, L phi = div u*.
    // div u* -> d_o
    deriv_axis_kernel<<<B(),T>>>(d_sx,d_o,N,N,N,0,dA,1,0);
    deriv_axis_kernel<<<B(),T>>>(d_sy,d_o,N,N,N,1,dA,1,1);
    deriv_axis_kernel<<<B(),T>>>(d_sz,d_o,N,N,N,2,dA,1,1); CK(cudaDeviceSynchronize());
    {Field3D dv(N,N,N);CK(cudaMemcpy(dv.a.data(),d_o,bytes,cudaMemcpyDeviceToHost));printf("  div(u*)    max %.3e (pre-projection)\n",maxabs(dv));}
    // phi = IFFT( FFT(div)/Lsym )/N^3   (d_t holds phi)
    r2c<<<Bn(NN),TT>>>(d_o,d_c,NN); CK(cudaDeviceSynchronize());
    FK(cufftExecZ2Z(plan,d_c,d_c,CUFFT_FORWARD));
    cmul_real<<<Bn(NN),TT>>>(d_c,d_invL,NN);
    FK(cufftExecZ2Z(plan,d_c,d_c,CUFFT_INVERSE));
    c2r<<<Bn(NN),TT>>>(d_c,d_t,1.0/(double)NN,NN); CK(cudaDeviceSynchronize());
    // u^{n+1} = u* - grad phi  (grad phi per axis into d_o, subtract)
    deriv_axis_kernel<<<B(),T>>>(d_t,d_o,N,N,N,0,dA,1,0); CK(cudaDeviceSynchronize()); fsub<<<Bn(NN),TT>>>(d_sx,d_o,NN);
    deriv_axis_kernel<<<B(),T>>>(d_t,d_o,N,N,N,1,dA,1,0); CK(cudaDeviceSynchronize()); fsub<<<Bn(NN),TT>>>(d_sy,d_o,NN);
    deriv_axis_kernel<<<B(),T>>>(d_t,d_o,N,N,N,2,dA,1,0); CK(cudaDeviceSynchronize()); fsub<<<Bn(NN),TT>>>(d_sz,d_o,NN);
    CK(cudaDeviceSynchronize());
    // div(u^{n+1}) -> should be roundoff
    deriv_axis_kernel<<<B(),T>>>(d_sx,d_o,N,N,N,0,dA,1,0);
    deriv_axis_kernel<<<B(),T>>>(d_sy,d_o,N,N,N,1,dA,1,1);
    deriv_axis_kernel<<<B(),T>>>(d_sz,d_o,N,N,N,2,dA,1,1); CK(cudaDeviceSynchronize());
    {Field3D dv(N,N,N);CK(cudaMemcpy(dv.a.data(),d_o,bytes,cudaMemcpyDeviceToHost));double m=maxabs(dv);
     printf("  div(u^n+1) max %.3e  %s (projection)\n",m,(m<1e-9)?"PASS":"FAIL");}

    // projection throughput
    int reps=20; auto t0=std::chrono::high_resolution_clock::now();
    for(int r=0;r<reps;++r){
        deriv_axis_kernel<<<B(),T>>>(d_sx,d_o,N,N,N,0,dA,1,0);deriv_axis_kernel<<<B(),T>>>(d_sy,d_o,N,N,N,1,dA,1,1);deriv_axis_kernel<<<B(),T>>>(d_sz,d_o,N,N,N,2,dA,1,1);
        r2c<<<Bn(NN),TT>>>(d_o,d_c,NN);FK(cufftExecZ2Z(plan,d_c,d_c,CUFFT_FORWARD));cmul_real<<<Bn(NN),TT>>>(d_c,d_invL,NN);FK(cufftExecZ2Z(plan,d_c,d_c,CUFFT_INVERSE));c2r<<<Bn(NN),TT>>>(d_c,d_t,1.0/(double)NN,NN);
        deriv_axis_kernel<<<B(),T>>>(d_t,d_o,N,N,N,0,dA,1,0);fsub<<<Bn(NN),TT>>>(d_sx,d_o,NN);
        deriv_axis_kernel<<<B(),T>>>(d_t,d_o,N,N,N,1,dA,1,0);fsub<<<Bn(NN),TT>>>(d_sy,d_o,NN);
        deriv_axis_kernel<<<B(),T>>>(d_t,d_o,N,N,N,2,dA,1,0);fsub<<<Bn(NN),TT>>>(d_sz,d_o,NN);
    } CK(cudaDeviceSynchronize());
    auto t1=std::chrono::high_resolution_clock::now();
    printf("  GPU projection: %.2f ms/projection\n",std::chrono::duration<double,std::milli>(t1-t0).count()/reps);
    cufftDestroy(plan);
    return 0;
}
