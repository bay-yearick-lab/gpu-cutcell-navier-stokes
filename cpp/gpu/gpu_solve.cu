// gpu_solve.cu
//
// GPU brick 8: the fast-diag IMPLICIT SOLVE (FastDiagHill::solve) assembled on
// device from the proven primitives.  Per-mode col-major layout buf[k*nr+m]
// (shared by bricks 4 & 7), so:
//   - shared real matrices (M2,Qx2,Qy2,Qx2T,Qy2T) apply via ONE cuBLAS GEMM with
//     transa=OP_T (row-major Mat read col-major = transpose), ncol = nmodes;
//   - per-mode Ainv[k]/Sinv[k] apply via cublasDgemmStridedBatched (brick 4);
//   - DFT_z/IDFT_z via the brick-7 direct kernels;
//   - complex assembly (div, Gp, u=ut-corr) via pointwise kernels.
// Verified vs CPU FastDiagHill::solve to roundoff on a real Hill3D.

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include "hill3d.h"
#include "hill_fastdiag.h"

__global__ void k_dft(int nr,int nz,const double* f,double* Xr,double* Xi){
    int t=blockIdx.x*blockDim.x+threadIdx.x; if(t>=nr*nz) return; int k=t/nr,m=t%nr; double re=0,im=0;
    for(int iz=0;iz<nz;++iz){ double an=-2.0*M_PI*iz*k/nz; double v=f[(size_t)m*nz+iz]; re+=v*cos(an); im+=v*sin(an);} Xr[(size_t)k*nr+m]=re; Xi[(size_t)k*nr+m]=im; }
__global__ void k_idft(int nr,int nz,const double* Ur,const double* Ui,double* f){
    int t=blockIdx.x*blockDim.x+threadIdx.x; if(t>=nr*nz) return; int m=t/nz,iz=t%nz; double s=0;
    for(int k=0;k<nz;++k){ double an=2.0*M_PI*iz*k/nz; s+=Ur[(size_t)k*nr+m]*cos(an)-Ui[(size_t)k*nr+m]*sin(an);} f[(size_t)m*nz+iz]=s/nz; }
__global__ void k_scale_mu(int nr,int nm,double* x,const double* mu){
    int t=blockIdx.x*blockDim.x+threadIdx.x; if(t>=nr*nm) return; x[t]*=mu[t/nr]; }
__global__ void k_div(int nr,int nm,double* dr,double* di,const double* axr,const double* axi,
                      const double* ayr,const double* ayi,const double* azr,const double* azi,const double* mu,const double* kap){
    int t=blockIdx.x*blockDim.x+threadIdx.x; if(t>=nr*nm) return; int k=t/nr; double m=mu[k],kp=kap[k];
    dr[t]=m*axr[t]+m*ayr[t]-kp*azi[t]; di[t]=m*axi[t]+m*ayi[t]+kp*azr[t]; }
__global__ void k_gp(int nr,int nm,double* gxr,double* gxi,double* gyr,double* gyi,double* gzr,double* gzi,
                     const double* qxr,const double* qxi,const double* qyr,const double* qyi,const double* m2pr,const double* m2pi,const double* mu,const double* kap){
    int t=blockIdx.x*blockDim.x+threadIdx.x; if(t>=nr*nm) return; int k=t/nr; double m=mu[k],kp=kap[k];
    gxr[t]=m*qxr[t]; gxi[t]=m*qxi[t]; gyr[t]=m*qyr[t]; gyi[t]=m*qyi[t]; gzr[t]=kp*m2pi[t]; gzi[t]=-kp*m2pr[t]; }
__global__ void k_sub(int n,double* U,const double* ut,const double* corr){ int t=blockIdx.x*blockDim.x+threadIdx.x; if(t>=n) return; U[t]=ut[t]-corr[t]; }

int main(){
    const int ne=6,p=3,nz=12; const double Lz=4.5;
    Hill3D H; H.build(ne,p,nz,Lz);
    FastDiagHill fd; fd.build(H,0.01);
    const int nr=fd.nr, nm=fd.nz;
    std::printf("gpu_solve: nr=%d nmodes=%d\n", nr, nm);

    // random input velocity (ux,uy,uz) as FieldH
    FieldH ux(nr,nz),uy(nr,nz),uz(nr,nz); srand(8675309);
    for(auto&v:ux.v)v=(double)rand()/RAND_MAX-0.5; for(auto&v:uy.v)v=(double)rand()/RAND_MAX-0.5; for(auto&v:uz.v)v=(double)rand()/RAND_MAX-0.5;
    // CPU reference
    auto ref = fd.solve(ux,uy,uz);

    cublasHandle_t hb; cublasCreate(&hb); const double one=1.0,zero=0.0;
    auto up=[&](const std::vector<double>&h){ double*d; cudaMalloc(&d,h.size()*sizeof(double)); cudaMemcpy(d,h.data(),h.size()*sizeof(double),cudaMemcpyHostToDevice); return d; };
    auto alloc=[&](int n){ double*d; cudaMalloc(&d,(size_t)n*sizeof(double)); return d; };
    // shared matrices (row-major)
    double *dM2=up(fd.M2.a),*dQx=up(fd.Qx2.a),*dQy=up(fd.Qy2.a),*dQxT=up(fd.Qx2T.a),*dQyT=up(fd.Qy2T.a);
    double *dmu=up(fd.mu),*dkap=up(fd.kap);
    // packed per-mode inverses
    std::vector<double> Apk((size_t)nr*nr*nm),Spk((size_t)nr*nr*nm);
    for(int k=0;k<nm;++k){ for(int i=0;i<nr*nr;++i){ Apk[(size_t)k*nr*nr+i]=fd.Ainv[k].a[i]; Spk[(size_t)k*nr*nr+i]=fd.Sinv[k].a[i]; } }
    double *dAi=up(Apk),*dSi=up(Spk);
    double *dux=up(ux.v),*duy=up(uy.v),*duz=up(uz.v);

    const int NM=nr*nm;
    // mode buffers
    double *Xr=alloc(NM),*Xi=alloc(NM),*Yr=alloc(NM),*Yi=alloc(NM),*Zr=alloc(NM),*Zi=alloc(NM);
    double *t1=alloc(NM),*t2=alloc(NM);
    double *uxr=alloc(NM),*uxi=alloc(NM),*uyr=alloc(NM),*uyi=alloc(NM),*uzr=alloc(NM),*uzi=alloc(NM);
    double *axr=alloc(NM),*axi=alloc(NM),*ayr=alloc(NM),*ayi=alloc(NM),*azr=alloc(NM),*azi=alloc(NM);
    double *dr=alloc(NM),*di=alloc(NM),*pr=alloc(NM),*pi=alloc(NM);
    double *gxr=alloc(NM),*gxi=alloc(NM),*gyr=alloc(NM),*gyi=alloc(NM),*gzr=alloc(NM),*gzi=alloc(NM);
    double *cxr=alloc(NM),*cxi=alloc(NM),*cyr=alloc(NM),*cyi=alloc(NM),*czr=alloc(NM),*czi=alloc(NM);
    double *Uxr=alloc(NM),*Uxi=alloc(NM),*Uyr=alloc(NM),*Uyi=alloc(NM),*Uzr=alloc(NM),*Uzi=alloc(NM);
    double *ox=alloc(nr*nz),*oy=alloc(nr*nz),*oz=alloc(nr*nz);

    int B=128; auto G=[&](int n){return (n+B-1)/B;};
    // shared apply: OUT(nr x nm) = A(nr x nr) . IN(nr x nm), A row-major => OP_T
    auto SH=[&](double* A,const double* IN,double* OUT){ cublasDgemm(hb,CUBLAS_OP_T,CUBLAS_OP_N,nr,nm,nr,&one,A,nr,IN,nr,&zero,OUT,nr); };
    // batched per-mode: OUT_k = Minv_k . IN_k
    auto BA=[&](double* Mp,const double* IN,double* OUT){ cublasDgemmStridedBatched(hb,CUBLAS_OP_T,CUBLAS_OP_N,nr,1,nr,&one,Mp,nr,(long long)nr*nr,IN,nr,(long long)nr,&zero,OUT,nr,(long long)nr,nm); };

    // DFT
    k_dft<<<G(nr*nz),B>>>(nr,nz,dux,Xr,Xi); k_dft<<<G(nr*nz),B>>>(nr,nz,duy,Yr,Yi); k_dft<<<G(nr*nz),B>>>(nr,nz,duz,Zr,Zi);
    // tent ut = Ainv*(mu*M2*U)
    SH(dM2,Xr,t1); k_scale_mu<<<G(NM),B>>>(nr,nm,t1,dmu); BA(dAi,t1,uxr);
    SH(dM2,Xi,t2); k_scale_mu<<<G(NM),B>>>(nr,nm,t2,dmu); BA(dAi,t2,uxi);
    SH(dM2,Yr,t1); k_scale_mu<<<G(NM),B>>>(nr,nm,t1,dmu); BA(dAi,t1,uyr);
    SH(dM2,Yi,t2); k_scale_mu<<<G(NM),B>>>(nr,nm,t2,dmu); BA(dAi,t2,uyi);
    SH(dM2,Zr,t1); k_scale_mu<<<G(NM),B>>>(nr,nm,t1,dmu); BA(dAi,t1,uzr);
    SH(dM2,Zi,t2); k_scale_mu<<<G(NM),B>>>(nr,nm,t2,dmu); BA(dAi,t2,uzi);
    // div assembly
    SH(dQx,uxr,axr); SH(dQx,uxi,axi); SH(dQy,uyr,ayr); SH(dQy,uyi,ayi); SH(dM2,uzr,azr); SH(dM2,uzi,azi);
    k_div<<<G(NM),B>>>(nr,nm,dr,di,axr,axi,ayr,ayi,azr,azi,dmu,dkap);
    // p = Sinv*div
    BA(dSi,dr,pr); BA(dSi,di,pi);
    // Gp = mu Qx2^T p, mu Qy2^T p, gz=(-i kap)M2 p
    SH(dQxT,pr,axr); SH(dQxT,pi,axi); SH(dQyT,pr,ayr); SH(dQyT,pi,ayi); SH(dM2,pr,azr); SH(dM2,pi,azi);
    k_gp<<<G(NM),B>>>(nr,nm,gxr,gxi,gyr,gyi,gzr,gzi,axr,axi,ayr,ayi,azr,azi,dmu,dkap);
    // corr = Ainv*Gp ; u = ut - corr
    BA(dAi,gxr,cxr); BA(dAi,gxi,cxi); BA(dAi,gyr,cyr); BA(dAi,gyi,cyi); BA(dAi,gzr,czr); BA(dAi,gzi,czi);
    k_sub<<<G(NM),B>>>(NM,Uxr,uxr,cxr); k_sub<<<G(NM),B>>>(NM,Uxi,uxi,cxi);
    k_sub<<<G(NM),B>>>(NM,Uyr,uyr,cyr); k_sub<<<G(NM),B>>>(NM,Uyi,uyi,cyi);
    k_sub<<<G(NM),B>>>(NM,Uzr,uzr,czr); k_sub<<<G(NM),B>>>(NM,Uzi,uzi,czi);
    // IDFT
    k_idft<<<G(nr*nz),B>>>(nr,nz,Uxr,Uxi,ox); k_idft<<<G(nr*nz),B>>>(nr,nz,Uyr,Uyi,oy); k_idft<<<G(nr*nz),B>>>(nr,nz,Uzr,Uzi,oz);
    cudaError_t err=cudaDeviceSynchronize(); if(err!=cudaSuccess){ std::printf("CUDA error: %s\n",cudaGetErrorString(err)); return 1; }

    std::vector<double> hx(nr*nz),hy(nr*nz),hz(nr*nz);
    cudaMemcpy(hx.data(),ox,hx.size()*sizeof(double),cudaMemcpyDeviceToHost);
    cudaMemcpy(hy.data(),oy,hy.size()*sizeof(double),cudaMemcpyDeviceToHost);
    cudaMemcpy(hz.data(),oz,hz.size()*sizeof(double),cudaMemcpyDeviceToHost);
    auto cmp=[&](const std::vector<double>&g,const FieldH&r,const char* nmn){ double d=0,s=0;
        for(size_t i=0;i<g.size();++i){ d=std::max(d,std::fabs(g[i]-r.v[i])); s=std::max(s,std::fabs(r.v[i])); }
        std::printf("  %s: max|gpu-cpu|=%.3e rel=%.3e\n",nmn,d,d/std::max(s,1e-300)); return d/std::max(s,1e-300); };
    std::printf("device FastDiagHill::solve vs CPU:\n");
    double e=0; e=std::max(e,cmp(hx,ref[0],"ux")); e=std::max(e,cmp(hy,ref[1],"uy")); e=std::max(e,cmp(hz,ref[2],"uz"));
    std::printf("=> max rel = %.3e %s\n", e, e<1e-10?"ROUNDOFF/OK":"MISMATCH");
    cublasDestroy(hb);
    return 0;
}
