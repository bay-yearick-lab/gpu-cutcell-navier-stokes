// gpu_skewfull.cu
//
// GPU brick 9: the COMPLETE hill_skew_conv on device (brick 6 + the mass_solve it
// deferred).  mass_solve = applyz(apply2d(out, M2inv), Mzinv) -- two dense applies
// as direct kernels (k_apply2d, k_applyz; unambiguous, cuBLAS-GEMM is the perf path
// = brick 1).  Verified vs the FULL CPU hill_skew_conv to roundoff.  Closes the one
// piece brick 6 deferred, so the RK3 step (brick 10) is trivial elementwise glue.

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <cuda_runtime.h>
#include "hill3d.h"

__global__ void k_xy(int n2q,int nz,const int* rp,const int* col,const double* val,const double* c,double* tmp){
    int t=blockIdx.x*blockDim.x+threadIdx.x; if(t>=n2q*nz) return; int q=t/nz,iz=t%nz; double s=0;
    for(int a=rp[q];a<rp[q+1];++a) s+=val[a]*c[(size_t)col[a]*nz+iz]; tmp[(size_t)q*nz+iz]=s; }
__global__ void k_z(int n2q,int nz,int nzq,const double* B,const double* tmp,double* out){
    int t=blockIdx.x*blockDim.x+threadIdx.x; if(t>=n2q*nzq) return; int q=t/nzq,qz=t%nzq; double s=0;
    for(int iz=0;iz<nz;++iz) s+=B[(size_t)qz*nz+iz]*tmp[(size_t)q*nz+iz]; out[(size_t)q*nzq+qz]=s; }
__global__ void k_zint(int n2q,int nz,int nzq,const double* B,const double* wz,const double* fq,double* tmp){
    int t=blockIdx.x*blockDim.x+threadIdx.x; if(t>=n2q*nz) return; int q=t/nz,iz=t%nz; double s=0;
    for(int qz=0;qz<nzq;++qz) s+=B[(size_t)qz*nz+iz]*wz[qz]*fq[(size_t)q*nzq+qz]; tmp[(size_t)q*nz+iz]=s; }
__global__ void k_scatter(int n2q,int nz,const int* rp,const int* col,const double* val,const double* w2,const double* tmp,double* red){
    int t=blockIdx.x*blockDim.x+threadIdx.x; if(t>=n2q*nz) return; int q=t/nz,iz=t%nz; double tv=w2[q]*tmp[(size_t)q*nz+iz];
    for(int a=rp[q];a<rp[q+1];++a) atomicAdd(&red[(size_t)col[a]*nz+iz], val[a]*tv); }
__global__ void k_pw1(int N,const double* wx,const double* wy,const double* wz,const double* uiq,const double* gx,const double* gy,const double* gz,
                      double* conv,double* Fx,double* Fy,double* Fz){
    int k=blockIdx.x*blockDim.x+threadIdx.x; if(k>=N) return;
    conv[k]=wx[k]*gx[k]+wy[k]*gy[k]+wz[k]*gz[k]; Fx[k]=wx[k]*uiq[k]; Fy[k]=wy[k]*uiq[k]; Fz[k]=wz[k]*uiq[k]; }
__global__ void k_pw2(int N,const double* la,const double* dgx,const double* dgy,const double* dgz,double* out){
    int k=blockIdx.x*blockDim.x+threadIdx.x; if(k>=N) return; out[k]=0.5*(la[k]-(dgx[k]+dgy[k]+dgz[k])); }
// mass_solve: apply2d (M2inv per z-slice) then applyz (Mzinv per 2D mode)
__global__ void k_apply2d(int nr2,int nz,const double* M2inv,const double* in,double* out){
    int t=blockIdx.x*blockDim.x+threadIdx.x; if(t>=nr2*nz) return; int i=t/nz,iz=t%nz; double s=0;
    for(int j=0;j<nr2;++j) s+=M2inv[(size_t)i*nr2+j]*in[(size_t)j*nz+iz]; out[(size_t)i*nz+iz]=s; }
__global__ void k_applyz(int nr2,int nz,const double* Mzinv,const double* in,double* out){
    int t=blockIdx.x*blockDim.x+threadIdx.x; if(t>=nr2*nz) return; int m=t/nz,iz=t%nz; double s=0;
    for(int jz=0;jz<nz;++jz) s+=Mzinv[(size_t)iz*nz+jz]*in[(size_t)m*nz+jz]; out[(size_t)m*nz+iz]=s; }

static void csr(const Hill3D& H,const std::vector<std::vector<double>>& SV,std::vector<int>& rp,std::vector<int>& col,std::vector<double>& val){
    int n2q=H.n2q; rp.assign(n2q+1,0); col.clear(); val.clear();
    for(int q=0;q<n2q;++q){ rp[q+1]=rp[q]+(int)H.sp_idx[q].size();
        for(size_t a=0;a<H.sp_idx[q].size();++a){ col.push_back(H.sp_idx[q][a]); val.push_back(SV[q][a]); } } }

int main(){
    const int ne=8,p=3,nz=16; const double Lz=4.5;
    Hill3D H; H.build(ne,p,nz,Lz);
    const int n2q=H.n2q,nzq=H.nzq,nr2=H.nr2,NQ=n2q*nzq,NC=nr2*nz;

    FieldH ui(nr2,nz); QField wxq(n2q,nzq),wyq(n2q,nzq),wzq(n2q,nzq); srand(31337);
    for(auto&v:ui.v)v=(double)rand()/RAND_MAX-0.5; for(auto&v:wxq.v)v=(double)rand()/RAND_MAX-0.5;
    for(auto&v:wyq.v)v=(double)rand()/RAND_MAX-0.5; for(auto&v:wzq.v)v=(double)rand()/RAND_MAX-0.5;

    FieldH ref = hill_skew_conv(H, wxq, wyq, wzq, ui);   // FULL CPU reference

    std::vector<int> rpP,colP,rpX,colX,rpY,colY; std::vector<double> vP,vX,vY;
    csr(H,H.sp_phi,rpP,colP,vP); csr(H,H.sp_dpx,rpX,colX,vX); csr(H,H.sp_dpy,rpY,colY,vY);
    const int nnz=(int)colP.size();
    auto up=[&](const std::vector<double>&h){ double*d; cudaMalloc(&d,h.size()*sizeof(double)); cudaMemcpy(d,h.data(),h.size()*sizeof(double),cudaMemcpyHostToDevice); return d; };
    auto upi=[&](const std::vector<int>&h){ int*d; cudaMalloc(&d,h.size()*sizeof(int)); cudaMemcpy(d,h.data(),h.size()*sizeof(int),cudaMemcpyHostToDevice); return d; };
    auto al=[&](int n){ double*d; cudaMalloc(&d,(size_t)n*sizeof(double)); return d; };
    double *d_ui=up(ui.v),*d_wx=up(wxq.v),*d_wy=up(wyq.v),*d_wz=up(wzq.v),*d_Vz=up(H.Vz.a),*d_dVz=up(H.dVz.a),*d_wzq=up(H.wz),*d_w2=up(H.w2);
    double *d_vP=up(vP),*d_vX=up(vX),*d_vY=up(vY),*d_M2i=up(H.M2inv.a),*d_Mzi=up(H.Mzinv.a);
    int *d_rpP=upi(rpP),*d_colP=upi(colP),*d_rpX=upi(rpX),*d_colX=upi(colX),*d_rpY=upi(rpY),*d_colY=upi(colY);
    double *d_tmp=al(n2q*nz),*d_uiq=al(NQ),*d_gx=al(NQ),*d_gy=al(NQ),*d_gz=al(NQ),*d_conv=al(NQ),*d_Fx=al(NQ),*d_Fy=al(NQ),*d_Fz=al(NQ);
    double *d_tz=al(n2q*nz),*d_la=al(NC),*d_dgx=al(NC),*d_dgy=al(NC),*d_dgz=al(NC),*d_pre=al(NC),*d_t1=al(NC),*d_out=al(NC);
    int B=128; auto Gq=[&](int n){return (n+B-1)/B;};
    k_xy<<<Gq(n2q*nz),B>>>(n2q,nz,d_rpP,d_colP,d_vP,d_ui,d_tmp); k_z<<<Gq(NQ),B>>>(n2q,nz,nzq,d_Vz,d_tmp,d_uiq);
    k_xy<<<Gq(n2q*nz),B>>>(n2q,nz,d_rpX,d_colX,d_vX,d_ui,d_tmp); k_z<<<Gq(NQ),B>>>(n2q,nz,nzq,d_Vz,d_tmp,d_gx);
    k_xy<<<Gq(n2q*nz),B>>>(n2q,nz,d_rpY,d_colY,d_vY,d_ui,d_tmp); k_z<<<Gq(NQ),B>>>(n2q,nz,nzq,d_Vz,d_tmp,d_gy);
    k_xy<<<Gq(n2q*nz),B>>>(n2q,nz,d_rpP,d_colP,d_vP,d_ui,d_tmp); k_z<<<Gq(NQ),B>>>(n2q,nz,nzq,d_dVz,d_tmp,d_gz);
    k_pw1<<<Gq(NQ),B>>>(NQ,d_wx,d_wy,d_wz,d_uiq,d_gx,d_gy,d_gz,d_conv,d_Fx,d_Fy,d_Fz);
    cudaMemset(d_la,0,NC*sizeof(double));  k_zint<<<Gq(n2q*nz),B>>>(n2q,nz,nzq,d_Vz,d_wzq,d_conv,d_tz); k_scatter<<<Gq(n2q*nz),B>>>(n2q,nz,d_rpP,d_colP,d_vP,d_w2,d_tz,d_la);
    cudaMemset(d_dgx,0,NC*sizeof(double)); k_zint<<<Gq(n2q*nz),B>>>(n2q,nz,nzq,d_Vz,d_wzq,d_Fx,d_tz);  k_scatter<<<Gq(n2q*nz),B>>>(n2q,nz,d_rpX,d_colX,d_vX,d_w2,d_tz,d_dgx);
    cudaMemset(d_dgy,0,NC*sizeof(double)); k_zint<<<Gq(n2q*nz),B>>>(n2q,nz,nzq,d_Vz,d_wzq,d_Fy,d_tz);  k_scatter<<<Gq(n2q*nz),B>>>(n2q,nz,d_rpY,d_colY,d_vY,d_w2,d_tz,d_dgy);
    cudaMemset(d_dgz,0,NC*sizeof(double)); k_zint<<<Gq(n2q*nz),B>>>(n2q,nz,nzq,d_dVz,d_wzq,d_Fz,d_tz); k_scatter<<<Gq(n2q*nz),B>>>(n2q,nz,d_rpP,d_colP,d_vP,d_w2,d_tz,d_dgz);
    k_pw2<<<Gq(NC),B>>>(NC,d_la,d_dgx,d_dgy,d_dgz,d_pre);
    // mass_solve
    k_apply2d<<<Gq(NC),B>>>(nr2,nz,d_M2i,d_pre,d_t1);
    k_applyz <<<Gq(NC),B>>>(nr2,nz,d_Mzi,d_t1,d_out);
    cudaError_t err=cudaDeviceSynchronize(); if(err!=cudaSuccess){ std::printf("CUDA error: %s\n",cudaGetErrorString(err)); return 1; }

    std::vector<double> out(NC); cudaMemcpy(out.data(),d_out,NC*sizeof(double),cudaMemcpyDeviceToHost);
    double maxd=0,maxr=0; for(int k=0;k<NC;++k){ maxd=std::max(maxd,std::fabs(out[k]-ref.v[k])); maxr=std::max(maxr,std::fabs(ref.v[k])); }
    std::printf("FULL device hill_skew_conv vs CPU: nr2=%d nz=%d nnz=%d  max|gpu-cpu|=%.3e  rel=%.3e\n",
                nr2,nz,nnz,maxd,maxd/std::max(maxr,1e-300));
    return 0;
}
