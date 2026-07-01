// gpu_dns_lowmem.cu
//
// Low-memory variant of gpu_dns.cu: identical periodic-hill DNS driver, with the
// four advection product buffers aliased onto gradient and sample buffers that
// are dead at that point in the step, which reduces the device footprint.

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include "hill3d.h"
#include "hill_fastdiag.h"
#include "stats_hill.h"
#include <string>

// ---- kernels (bricks 2,3,6,7,8,9) ----
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
__global__ void k_apply2d(int nr2,int nz,const double* M2inv,const double* in,double* out){
    int t=blockIdx.x*blockDim.x+threadIdx.x; if(t>=nr2*nz) return; int i=t/nz,iz=t%nz; double s=0;
    for(int j=0;j<nr2;++j) s+=M2inv[(size_t)i*nr2+j]*in[(size_t)j*nz+iz]; out[(size_t)i*nz+iz]=s; }
__global__ void k_applyz(int nr2,int nz,const double* Mzinv,const double* in,double* out){
    int t=blockIdx.x*blockDim.x+threadIdx.x; if(t>=nr2*nz) return; int m=t/nz,iz=t%nz; double s=0;
    for(int jz=0;jz<nz;++jz) s+=Mzinv[(size_t)iz*nz+jz]*in[(size_t)m*nz+jz]; out[(size_t)m*nz+iz]=s; }
__global__ void k_dft(int nr,int nz,const double* f,double* Xr,double* Xi){
    int t=blockIdx.x*blockDim.x+threadIdx.x; if(t>=nr*nz) return; int k=t/nr,m=t%nr; double re=0,im=0;
    for(int iz=0;iz<nz;++iz){ double an=-2.0*M_PI*iz*k/nz; double v=f[(size_t)m*nz+iz]; re+=v*cos(an); im+=v*sin(an);} Xr[(size_t)k*nr+m]=re; Xi[(size_t)k*nr+m]=im; }
__global__ void k_idft(int nr,int nz,const double* Ur,const double* Ui,double* f){
    int t=blockIdx.x*blockDim.x+threadIdx.x; if(t>=nr*nz) return; int m=t/nz,iz=t%nz; double s=0;
    for(int k=0;k<nz;++k){ double an=2.0*M_PI*iz*k/nz; s+=Ur[(size_t)k*nr+m]*cos(an)-Ui[(size_t)k*nr+m]*sin(an);} f[(size_t)m*nz+iz]=s/nz; }
__global__ void k_scale_mu(int nr,int nm,double* x,const double* mu){ int t=blockIdx.x*blockDim.x+threadIdx.x; if(t>=nr*nm) return; x[t]*=mu[t/nr]; }
__global__ void k_div(int nr,int nm,double* dr,double* di,const double* axr,const double* axi,const double* ayr,const double* ayi,const double* azr,const double* azi,const double* mu,const double* kap){
    int t=blockIdx.x*blockDim.x+threadIdx.x; if(t>=nr*nm) return; int k=t/nr; double m=mu[k],kp=kap[k]; dr[t]=m*axr[t]+m*ayr[t]-kp*azi[t]; di[t]=m*axi[t]+m*ayi[t]+kp*azr[t]; }
__global__ void k_gp(int nr,int nm,double* gxr,double* gxi,double* gyr,double* gyi,double* gzr,double* gzi,const double* qxr,const double* qxi,const double* qyr,const double* qyi,const double* m2pr,const double* m2pi,const double* mu,const double* kap){
    int t=blockIdx.x*blockDim.x+threadIdx.x; if(t>=nr*nm) return; int k=t/nr; double m=mu[k],kp=kap[k]; gxr[t]=m*qxr[t]; gxi[t]=m*qxi[t]; gyr[t]=m*qyr[t]; gyi[t]=m*qyi[t]; gzr[t]=kp*m2pi[t]; gzi[t]=-kp*m2pr[t]; }
__global__ void k_sub(int n,double* U,const double* ut,const double* corr){ int t=blockIdx.x*blockDim.x+threadIdx.x; if(t>=n) return; U[t]=ut[t]-corr[t]; }
// elementwise RK3 glue
__global__ void k_stage(int n,double* o,const double* u,const double* L,double dt){ int t=blockIdx.x*blockDim.x+threadIdx.x; if(t>=n) return; o[t]=u[t]+dt*L[t]; }
__global__ void k_comb(int n,double* o,double c1,const double* A,double c2,const double* Bb){ int t=blockIdx.x*blockDim.x+threadIdx.x; if(t>=n) return; o[t]=c1*A[t]+c2*Bb[t]; }
__global__ void k_neg(int n,double* x){ int t=blockIdx.x*blockDim.x+threadIdx.x; if(t>=n) return; x[t]=-x[t]; }
__global__ void k_addfx(int n,double* x,double v){ int t=blockIdx.x*blockDim.x+threadIdx.x; if(t>=n) return; x[t]+=v; }

static void csr(const Hill3D& H,const std::vector<std::vector<double>>& SV,std::vector<int>& rp,std::vector<int>& col,std::vector<double>& val){
    int n2q=H.n2q; rp.assign(n2q+1,0); col.clear(); val.clear();
    for(int q=0;q<n2q;++q){ rp[q+1]=rp[q]+(int)H.sp_idx[q].size(); for(size_t a=0;a<H.sp_idx[q].size();++a){ col.push_back(H.sp_idx[q][a]); val.push_back(SV[q][a]); } } }

struct Ctx {
    int n2q,nzq,nr2,nz,NQ,NC; int B=128; cublasHandle_t hb;
    int *rp,*col; double *vP,*vX,*vY,*Vz,*dVz,*wz,*w2,*M2inv,*Mzinv;          // skewconv ops
    double *M2,*Qx,*Qy,*QxT,*QyT,*mu,*kap,*Ai,*Si;                            // solve ops
    // skewconv scratch
    double *s_tmp,*s_uiq,*s_gx,*s_gy,*s_gz,*s_conv,*s_Fx,*s_Fy,*s_Fz,*s_tz,*s_la,*s_dgx,*s_dgy,*s_dgz,*s_pre,*s_t1;
    double *wxQ,*wyQ,*wzQ;
    // solve scratch
    double *Xr,*Xi,*Yr,*Yi,*Zr,*Zi,*t1,*t2,*uxr,*uxi,*uyr,*uyi,*uzr,*uzi,*axr,*axi,*ayr,*ayi,*azr,*azi,*dr,*di,*pr,*pi,*gxr,*gxi,*gyr,*gyi,*gzr,*gzi,*cxr,*cxi,*cyr,*cyi,*czr,*czi,*Uxr,*Uxi,*Uyr,*Uyi,*Uzr,*Uzi;
    int G(int n){return (n+B-1)/B;}
};

void g_sample(Ctx& c,const double* in,double* outQ){
    k_xy<<<c.G(c.n2q*c.nz),c.B>>>(c.n2q,c.nz,c.rp,c.col,c.vP,in,c.s_tmp);
    k_z<<<c.G(c.NQ),c.B>>>(c.n2q,c.nz,c.nzq,c.Vz,c.s_tmp,outQ);
}
void g_skewconv(Ctx& c,const double* wx,const double* wy,const double* wz,const double* ui,double* out){
    int B=c.B; auto Gq=[&](int n){return c.G(n);};
    k_xy<<<Gq(c.n2q*c.nz),B>>>(c.n2q,c.nz,c.rp,c.col,c.vP,ui,c.s_tmp); k_z<<<Gq(c.NQ),B>>>(c.n2q,c.nz,c.nzq,c.Vz,c.s_tmp,c.s_uiq);
    k_xy<<<Gq(c.n2q*c.nz),B>>>(c.n2q,c.nz,c.rp,c.col,c.vX,ui,c.s_tmp); k_z<<<Gq(c.NQ),B>>>(c.n2q,c.nz,c.nzq,c.Vz,c.s_tmp,c.s_gx);
    k_xy<<<Gq(c.n2q*c.nz),B>>>(c.n2q,c.nz,c.rp,c.col,c.vY,ui,c.s_tmp); k_z<<<Gq(c.NQ),B>>>(c.n2q,c.nz,c.nzq,c.Vz,c.s_tmp,c.s_gy);
    k_xy<<<Gq(c.n2q*c.nz),B>>>(c.n2q,c.nz,c.rp,c.col,c.vP,ui,c.s_tmp); k_z<<<Gq(c.NQ),B>>>(c.n2q,c.nz,c.nzq,c.dVz,c.s_tmp,c.s_gz);
    k_pw1<<<Gq(c.NQ),B>>>(c.NQ,wx,wy,wz,c.s_uiq,c.s_gx,c.s_gy,c.s_gz,c.s_conv,c.s_Fx,c.s_Fy,c.s_Fz);
    cudaMemset(c.s_la,0,c.NC*sizeof(double));  k_zint<<<Gq(c.n2q*c.nz),B>>>(c.n2q,c.nz,c.nzq,c.Vz,c.wz,c.s_conv,c.s_tz); k_scatter<<<Gq(c.n2q*c.nz),B>>>(c.n2q,c.nz,c.rp,c.col,c.vP,c.w2,c.s_tz,c.s_la);
    cudaMemset(c.s_dgx,0,c.NC*sizeof(double)); k_zint<<<Gq(c.n2q*c.nz),B>>>(c.n2q,c.nz,c.nzq,c.Vz,c.wz,c.s_Fx,c.s_tz);  k_scatter<<<Gq(c.n2q*c.nz),B>>>(c.n2q,c.nz,c.rp,c.col,c.vX,c.w2,c.s_tz,c.s_dgx);
    cudaMemset(c.s_dgy,0,c.NC*sizeof(double)); k_zint<<<Gq(c.n2q*c.nz),B>>>(c.n2q,c.nz,c.nzq,c.Vz,c.wz,c.s_Fy,c.s_tz);  k_scatter<<<Gq(c.n2q*c.nz),B>>>(c.n2q,c.nz,c.rp,c.col,c.vY,c.w2,c.s_tz,c.s_dgy);
    cudaMemset(c.s_dgz,0,c.NC*sizeof(double)); k_zint<<<Gq(c.n2q*c.nz),B>>>(c.n2q,c.nz,c.nzq,c.dVz,c.wz,c.s_Fz,c.s_tz); k_scatter<<<Gq(c.n2q*c.nz),B>>>(c.n2q,c.nz,c.rp,c.col,c.vP,c.w2,c.s_tz,c.s_dgz);
    k_pw2<<<Gq(c.NC),B>>>(c.NC,c.s_la,c.s_dgx,c.s_dgy,c.s_dgz,c.s_pre);
    k_apply2d<<<Gq(c.NC),B>>>(c.nr2,c.nz,c.M2inv,c.s_pre,c.s_t1);
    k_applyz <<<Gq(c.NC),B>>>(c.nr2,c.nz,c.Mzinv,c.s_t1,out);
}
void g_solve(Ctx& c,const double* ux,const double* uy,const double* uz,double* ox,double* oy,double* oz){
    int nr=c.nr2,nm=c.nz,NM=nr*nm,B=c.B; auto Gq=[&](int n){return c.G(n);}; const double one=1.0,zero=0.0;
    auto SH=[&](double* A,const double* IN,double* OUT){ cublasDgemm(c.hb,CUBLAS_OP_T,CUBLAS_OP_N,nr,nm,nr,&one,A,nr,IN,nr,&zero,OUT,nr); };
    auto BA=[&](double* Mp,const double* IN,double* OUT){ cublasDgemmStridedBatched(c.hb,CUBLAS_OP_T,CUBLAS_OP_N,nr,1,nr,&one,Mp,nr,(long long)nr*nr,IN,nr,(long long)nr,&zero,OUT,nr,(long long)nr,nm); };
    k_dft<<<Gq(nr*nm),B>>>(nr,nm,ux,c.Xr,c.Xi); k_dft<<<Gq(nr*nm),B>>>(nr,nm,uy,c.Yr,c.Yi); k_dft<<<Gq(nr*nm),B>>>(nr,nm,uz,c.Zr,c.Zi);
    SH(c.M2,c.Xr,c.t1); k_scale_mu<<<Gq(NM),B>>>(nr,nm,c.t1,c.mu); BA(c.Ai,c.t1,c.uxr);
    SH(c.M2,c.Xi,c.t2); k_scale_mu<<<Gq(NM),B>>>(nr,nm,c.t2,c.mu); BA(c.Ai,c.t2,c.uxi);
    SH(c.M2,c.Yr,c.t1); k_scale_mu<<<Gq(NM),B>>>(nr,nm,c.t1,c.mu); BA(c.Ai,c.t1,c.uyr);
    SH(c.M2,c.Yi,c.t2); k_scale_mu<<<Gq(NM),B>>>(nr,nm,c.t2,c.mu); BA(c.Ai,c.t2,c.uyi);
    SH(c.M2,c.Zr,c.t1); k_scale_mu<<<Gq(NM),B>>>(nr,nm,c.t1,c.mu); BA(c.Ai,c.t1,c.uzr);
    SH(c.M2,c.Zi,c.t2); k_scale_mu<<<Gq(NM),B>>>(nr,nm,c.t2,c.mu); BA(c.Ai,c.t2,c.uzi);
    SH(c.Qx,c.uxr,c.axr); SH(c.Qx,c.uxi,c.axi); SH(c.Qy,c.uyr,c.ayr); SH(c.Qy,c.uyi,c.ayi); SH(c.M2,c.uzr,c.azr); SH(c.M2,c.uzi,c.azi);
    k_div<<<Gq(NM),B>>>(nr,nm,c.dr,c.di,c.axr,c.axi,c.ayr,c.ayi,c.azr,c.azi,c.mu,c.kap);
    BA(c.Si,c.dr,c.pr); BA(c.Si,c.di,c.pi);
    SH(c.QxT,c.pr,c.axr); SH(c.QxT,c.pi,c.axi); SH(c.QyT,c.pr,c.ayr); SH(c.QyT,c.pi,c.ayi); SH(c.M2,c.pr,c.azr); SH(c.M2,c.pi,c.azi);
    k_gp<<<Gq(NM),B>>>(nr,nm,c.gxr,c.gxi,c.gyr,c.gyi,c.gzr,c.gzi,c.axr,c.axi,c.ayr,c.ayi,c.azr,c.azi,c.mu,c.kap);
    BA(c.Ai,c.gxr,c.cxr); BA(c.Ai,c.gxi,c.cxi); BA(c.Ai,c.gyr,c.cyr); BA(c.Ai,c.gyi,c.cyi); BA(c.Ai,c.gzr,c.czr); BA(c.Ai,c.gzi,c.czi);
    k_sub<<<Gq(NM),B>>>(NM,c.Uxr,c.uxr,c.cxr); k_sub<<<Gq(NM),B>>>(NM,c.Uxi,c.uxi,c.cxi);
    k_sub<<<Gq(NM),B>>>(NM,c.Uyr,c.uyr,c.cyr); k_sub<<<Gq(NM),B>>>(NM,c.Uyi,c.uyi,c.cyi);
    k_sub<<<Gq(NM),B>>>(NM,c.Uzr,c.uzr,c.czr); k_sub<<<Gq(NM),B>>>(NM,c.Uzi,c.uzi,c.czi);
    k_idft<<<Gq(nr*nm),B>>>(nr,nm,c.Uxr,c.Uxi,ox); k_idft<<<Gq(nr*nm),B>>>(nr,nm,c.Uyr,c.Uyi,oy); k_idft<<<Gq(nr*nm),B>>>(nr,nm,c.Uzr,c.Uzi,oz);
}

// device bulk streamwise velocity U_b = (1/Vol) int u_x dV via a weighted dot
static double g_bulk(Ctx& c,const double* ux,const double* d_W,double vol){
    g_sample(c, ux, c.wxQ);
    double s=0; cublasDdot(c.hb, c.NQ, d_W,1, c.wxQ,1, &s); return s/vol;
}
// Periodic-hill turbulent DNS driver: physical metric, streamwise body force (constant
// or PI-controlled to hold a target bulk velocity = fixed Re), online diagnostics.
// binary checkpoint of the full run state (field + controller + stats) for resume
static void save_ckpt(const std::string& path,int step,double fx,double integ,
                      double* const cur[3],int NC,const StationStats& stats){
    FILE* f=std::fopen((path+".tmp").c_str(),"wb"); if(!f) return;
    std::fwrite(&step,sizeof(int),1,f); std::fwrite(&fx,sizeof(double),1,f); std::fwrite(&integ,sizeof(double),1,f);
    std::fwrite(&NC,sizeof(int),1,f);
    std::vector<double> h(NC);
    for(int cc=0;cc<3;++cc){ cudaMemcpy(h.data(),cur[cc],NC*sizeof(double),cudaMemcpyDeviceToHost); std::fwrite(h.data(),sizeof(double),NC,f); }
    stats.save(f); std::fclose(f); std::rename((path+".tmp").c_str(),path.c_str());   // atomic
}
static bool load_ckpt(const std::string& path,int& step,double& fx,double& integ,
                      double* cur[3],int NC,StationStats& stats){
    FILE* f=std::fopen(path.c_str(),"rb"); if(!f) return false;
    int nc2=0; size_t g;
    g=std::fread(&step,sizeof(int),1,f); g=std::fread(&fx,sizeof(double),1,f); g=std::fread(&integ,sizeof(double),1,f);
    g=std::fread(&nc2,sizeof(int),1,f); (void)g; if(nc2!=NC){ std::fclose(f); return false; }
    std::vector<double> h(NC);
    for(int cc=0;cc<3;++cc){ size_t r=std::fread(h.data(),sizeof(double),NC,f); (void)r; cudaMemcpy(cur[cc],h.data(),NC*sizeof(double),cudaMemcpyHostToDevice); }
    stats.load(f); std::fclose(f); return true;
}
static double g_ic_amp = 0.05;   // IC broadband perturbation amplitude (set from CLI)
static double g_dt = 0.02;       // time step (set from CLI)
static std::string g_ckpt;       // checkpoint path ("" => none)
static int g_ckpt_every = 500;   // checkpoint cadence (steps)
static void dns(int ne,int nz,double nu,double fx0,double target_ub,int nsteps,int pe,
                int avg_start,const std::string& stats_path){
    const int p=3; const double Lz=4.5; const double dt=g_dt; const double a=dt*nu;
    const bool control = target_ub>0.0;
    Hill3D H; H.build(ne,p,nz,Lz,/*physical=*/true);
    FastDiagHill fd; fd.build(H,a);
    const int n2q=H.n2q,nzq=H.nzq,nr2=H.nr2,NQ=n2q*nzq,NC=nr2*nz;
    std::printf("GPU DNS: ne=%d nz=%d nr2=%d  nu=%.4g fx0=%.3g target_ub=%.3g (Re=%.0f) control=%d\n",
                ne,nz,nr2,nu,fx0,target_ub,(control?target_ub/nu:0.0),(int)control);
    HillNS s0; s0.ux=FieldH(nr2,nz); s0.uy=FieldH(nr2,nz); s0.uz=FieldH(nr2,nz); srand(20260629);
    // broadband perturbation IC to seed transition to turbulence
    const double amp=g_ic_amp;
    for(auto&v:s0.ux.v)v=amp*((double)rand()/RAND_MAX-0.5); for(auto&v:s0.uy.v)v=amp*((double)rand()/RAND_MAX-0.5); for(auto&v:s0.uz.v)v=amp*((double)rand()/RAND_MAX-0.5);

    Ctx c; c.n2q=n2q; c.nzq=nzq; c.nr2=nr2; c.nz=nz; c.NQ=NQ; c.NC=NC; cublasCreate(&c.hb);
    auto chk=[&](cudaError_t e,double mb,const char* what){ if(e!=cudaSuccess){ size_t fr=0,tot=0; cudaMemGetInfo(&fr,&tot); std::printf("cudaMalloc FAILED (%s, %.1f MB; free %.0f/%0.f MB): %s\n",what,mb,fr/1e6,tot/1e6,cudaGetErrorString(e)); std::exit(2);} };
    auto up=[&](const std::vector<double>&h){ double*d; chk(cudaMalloc(&d,h.size()*sizeof(double)),h.size()*8.0/1e6,"up"); cudaMemcpy(d,h.data(),h.size()*sizeof(double),cudaMemcpyHostToDevice); return d; };
    auto upi=[&](const std::vector<int>&h){ int*d; chk(cudaMalloc(&d,h.size()*sizeof(int)),h.size()*4.0/1e6,"upi"); cudaMemcpy(d,h.data(),h.size()*sizeof(int),cudaMemcpyHostToDevice); return d; };
    auto al=[&](int n){ double*d; chk(cudaMalloc(&d,(size_t)n*sizeof(double)),(double)n*8.0/1e6,"al"); return d; };
    std::vector<int> rp,col,r2,c2; std::vector<double> vP,vX,vY;
    csr(H,H.sp_phi,rp,col,vP); csr(H,H.sp_dpx,r2,c2,vX); csr(H,H.sp_dpy,r2,c2,vY);
    c.rp=upi(rp); c.col=upi(col); c.vP=up(vP); c.vX=up(vX); c.vY=up(vY);
    c.Vz=up(H.Vz.a); c.dVz=up(H.dVz.a); c.wz=up(H.wz); c.w2=up(H.w2); c.M2inv=up(H.M2inv.a); c.Mzinv=up(H.Mzinv.a);
    c.M2=up(fd.M2.a); c.Qx=up(fd.Qx2.a); c.Qy=up(fd.Qy2.a); c.QxT=up(fd.Qx2T.a); c.QyT=up(fd.Qy2T.a); c.mu=up(fd.mu); c.kap=up(fd.kap);
    std::vector<double> Apk((size_t)nr2*nr2*nz),Spk((size_t)nr2*nr2*nz);
    for(int k=0;k<nz;++k) for(int i=0;i<nr2*nr2;++i){ Apk[(size_t)k*nr2*nr2+i]=fd.Ainv[k].a[i]; Spk[(size_t)k*nr2*nr2+i]=fd.Sinv[k].a[i]; }
    c.Ai=up(Apk); c.Si=up(Spk);
    c.s_tmp=al(n2q*nz); c.s_uiq=al(NQ); c.s_gx=al(NQ); c.s_gy=al(NQ); c.s_gz=al(NQ);
    // LOWMEM: alias the 4 product buffers onto the (dead-after-k_pw1) grad/sample
    // buffers. k_pw1 is alias-safe in this order: conv=wx*gx+wy*gy+wz*gz consumes
    // gx,gy,gz first; then Fx,Fy,Fz=w*uiq with Fz the LAST read of uiq. Saves 4*NQ.
    c.s_conv=c.s_gx; c.s_Fx=c.s_gy; c.s_Fy=c.s_gz; c.s_Fz=c.s_uiq;
    c.s_tz=al(n2q*nz); c.s_la=al(NC); c.s_dgx=al(NC); c.s_dgy=al(NC); c.s_dgz=al(NC); c.s_pre=al(NC); c.s_t1=al(NC);
    c.wxQ=al(NQ); c.wyQ=al(NQ); c.wzQ=al(NQ);
    for(double** q:{&c.Xr,&c.Xi,&c.Yr,&c.Yi,&c.Zr,&c.Zi,&c.t1,&c.t2,&c.uxr,&c.uxi,&c.uyr,&c.uyi,&c.uzr,&c.uzi,&c.axr,&c.axi,&c.ayr,&c.ayi,&c.azr,&c.azi,&c.dr,&c.di,&c.pr,&c.pi,&c.gxr,&c.gxi,&c.gyr,&c.gyi,&c.gzr,&c.gzi,&c.cxr,&c.cxi,&c.cyr,&c.cyi,&c.czr,&c.czi,&c.Uxr,&c.Uxi,&c.Uyr,&c.Uyi,&c.Uzr,&c.Uzi}) *q=al(NC);
    // bulk-reduction weights W[q,qz]=w2[q]*wz[qz] + fluid volume
    std::vector<double> Wh(NQ); double vol=0;
    for(int q=0;q<n2q;++q) for(int iz=0;iz<nzq;++iz){ double w=H.w2[q]*H.wz[iz]; Wh[(size_t)q*nzq+iz]=w; vol+=w; }
    double* d_W=up(Wh);

    // device state buffers
    double *cur[3],*u1[3],*u2[3],*us[3],*Lb[3],*tb[3],*ou[3];
    for(int i=0;i<3;++i){ cur[i]=al(NC); u1[i]=al(NC); u2[i]=al(NC); us[i]=al(NC); Lb[i]=al(NC); tb[i]=al(NC); ou[i]=al(NC); }
    cudaMemcpy(cur[0],s0.ux.v.data(),NC*sizeof(double),cudaMemcpyHostToDevice);
    cudaMemcpy(cur[1],s0.uy.v.data(),NC*sizeof(double),cudaMemcpyHostToDevice);
    cudaMemcpy(cur[2],s0.uz.v.data(),NC*sizeof(double),cudaMemcpyHostToDevice);
    int B=c.B; auto G=[&](int n){return c.G(n);};
    auto advect=[&](double* u[3],double* out[3]){
        g_sample(c,u[0],c.wxQ); g_sample(c,u[1],c.wyQ); g_sample(c,u[2],c.wzQ);
        for(int i=0;i<3;++i){ g_skewconv(c,c.wxQ,c.wyQ,c.wzQ,u[i],out[i]); k_neg<<<G(NC),B>>>(NC,out[i]); }
    };
    double fx=fx0, integ=0.0; const double Kp=4.0, Ki=3.0;   // PI flux controller (bulk velocity)
    auto onestep=[&](double fxn){
        advect(cur,Lb); for(int i=0;i<3;++i) k_stage<<<G(NC),B>>>(NC,u1[i],cur[i],Lb[i],dt);
        advect(u1,Lb);  for(int i=0;i<3;++i){ k_stage<<<G(NC),B>>>(NC,tb[i],u1[i],Lb[i],dt); k_comb<<<G(NC),B>>>(NC,u2[i],0.75,cur[i],0.25,tb[i]); }
        advect(u2,Lb);  for(int i=0;i<3;++i){ k_stage<<<G(NC),B>>>(NC,tb[i],u2[i],Lb[i],dt); k_comb<<<G(NC),B>>>(NC,us[i],1.0/3.0,cur[i],2.0/3.0,tb[i]); }
        k_addfx<<<G(NC),B>>>(NC,us[0],dt*fxn);
        g_solve(c,us[0],us[1],us[2],ou[0],ou[1],ou[2]);
        for(int i=0;i<3;++i){ double* t=cur[i]; cur[i]=ou[i]; ou[i]=t; }   // cur <- new state
    };
    StationStats stats; const bool do_stats=!stats_path.empty(); if(do_stats) stats.build(H);
    const int se=5;   // stats sample cadence (steps)
    double ub_last=0.0;
    // resume from checkpoint if present (field + controller + stats); else start fresh
    int start_step=0;
    bool resumed=false;
    if(!g_ckpt.empty() && load_ckpt(g_ckpt,start_step,fx,integ,cur,NC,stats)){
        std::printf("RESUMED from %s at step %d (fx=%.4f integ=%.4f nsamp=%ld)\n",
                    g_ckpt.c_str(),start_step,fx,integ,stats.n); resumed=true;
    }
    std::printf(" step   U_bulk    Re_H        KE        weak_div   fx       nsamp\n");
    std::vector<double> hx(NC),hy(NC),hz(NC);
    HillNS st; st.ux=FieldH(nr2,nz); st.uy=FieldH(nr2,nz); st.uz=FieldH(nr2,nz);
    for(int it=start_step; it<=nsteps; ++it){
        double ub=g_bulk(c,cur[0],d_W,vol); ub_last=ub;
        const bool prnt=(it%pe==0), acc=(do_stats && it>=avg_start && it%se==0 && !(resumed && it==start_step));
        if(prnt||acc){
            cudaMemcpy(hx.data(),cur[0],NC*sizeof(double),cudaMemcpyDeviceToHost);
            cudaMemcpy(hy.data(),cur[1],NC*sizeof(double),cudaMemcpyDeviceToHost);
            cudaMemcpy(hz.data(),cur[2],NC*sizeof(double),cudaMemcpyDeviceToHost);
            st.ux.v=hx; st.uy.v=hy; st.uz.v=hz;
        }
        if(prnt){ FieldH dv=hill_weak_div(H,st.ux,st.uy,st.uz); double md=0; for(double v:dv.v)md=std::max(md,std::fabs(v));
            std::printf(" %5d  %8.5f  %8.1f  %10.4f  %.2e  %7.4f  %ld\n", it, ub, ub/nu, hill_ke(H,st), md, fx, stats.n); std::fflush(stdout);
            if(do_stats && stats.n>0){ const double ubn=control?target_ub:ub; stats.write_json(stats_path,H,ubn,(int)std::lround(ubn/nu),it,it*dt,0.0); } }
        if(acc) stats.accumulate(H,st.ux,st.uy,st.uz);
        if(!g_ckpt.empty() && it%g_ckpt_every==0) save_ckpt(g_ckpt,it,fx,integ,cur,NC,stats);
        if(it<nsteps){
            if(control){ double e=target_ub-ub; integ+=e*dt; fx=Kp*e+Ki*integ; }
            onestep(fx);
        }
    }
    cudaError_t err=cudaDeviceSynchronize();
    if(err!=cudaSuccess) std::printf("CUDA error: %s\n",cudaGetErrorString(err));
    if(do_stats){ const double ubn=control?target_ub:ub_last; const int reyn=(int)std::lround(ubn/nu);
        stats.write_json(stats_path,H,ubn,reyn,nsteps,nsteps*dt,0.0);
        std::printf("wrote %s  (n_samples=%ld, ub=%.4f Re=%d)\n", stats_path.c_str(), stats.n, ubn, reyn); }
    cublasDestroy(c.hb);
}

int main(int argc,char** argv){
    int ne     = argc>1?atoi(argv[1]):8;
    int nz     = argc>2?atoi(argv[2]):16;
    double nu  = argc>3?atof(argv[3]):0.05;
    double fx0 = argc>4?atof(argv[4]):0.5;
    double tgt = argc>5?atof(argv[5]):0.0;   // >0 => PI-control bulk velocity to tgt (fixed Re); 0 => constant fx0
    int nsteps = argc>6?atoi(argv[6]):100;
    int pe     = argc>7?atoi(argv[7]):20;
    int avg0   = argc>8?atoi(argv[8]):1000000;        // step to begin averaging (default: none)
    std::string sp = argc>9?std::string(argv[9]):std::string();   // stats JSON path ("" => no stats)
    if(argc>10) g_ic_amp = atof(argv[10]);            // IC perturbation amplitude
    if(argc>11) g_dt     = atof(argv[11]);            // time step
    if(argc>12) g_ckpt   = std::string(argv[12]);     // checkpoint path ("" => none)
    if(argc>13) g_ckpt_every = atoi(argv[13]);        // checkpoint cadence (steps)
    dns(ne,nz,nu,fx0,tgt,nsteps,pe,avg0,sp);
    return 0;
}
