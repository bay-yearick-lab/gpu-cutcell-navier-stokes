// gpu_dft.cu
//
// GPU brick 7: the spanwise DFT_z / IDFT_z used by FastDiagHill::solve, as direct
// kernels matching the CPU convention exactly (nz is small, so a dense direct
// transform is cheap and avoids cuBLAS orientation bookkeeping).  Output is
// mode-major Xr[k*nr+m] (contiguous nr-vector per mode) -- the layout the per-mode
// batched solves want.  Verified: (a) forward DFT vs the CPU dft lambda, and
// (b) the f -> DFT -> IDFT -> f round-trip, both to the roundoff floor.

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <cuda_runtime.h>

__global__ void k_dft(int nr,int nz,const double* f,double* Xr,double* Xi){
    int t=blockIdx.x*blockDim.x+threadIdx.x; if(t>=nr*nz) return;
    int k=t/nr, m=t%nr; double re=0,im=0;
    for(int iz=0;iz<nz;++iz){ double an=-2.0*M_PI*iz*k/nz; double v=f[(size_t)m*nz+iz];
        re+=v*cos(an); im+=v*sin(an); }
    Xr[(size_t)k*nr+m]=re; Xi[(size_t)k*nr+m]=im;
}
__global__ void k_idft(int nr,int nz,const double* Ur,const double* Ui,double* f){
    int t=blockIdx.x*blockDim.x+threadIdx.x; if(t>=nr*nz) return;
    int m=t/nz, iz=t%nz; double s=0;
    for(int k=0;k<nz;++k){ double an=2.0*M_PI*iz*k/nz; s+=Ur[(size_t)k*nr+m]*cos(an)-Ui[(size_t)k*nr+m]*sin(an); }
    f[(size_t)m*nz+iz]=s/nz;
}

int main(){
    const int nr=111, nz=16;
    std::vector<double> f((size_t)nr*nz); srand(424242);
    for(auto& v: f) v=(double)rand()/RAND_MAX-0.5;

    // CPU reference forward DFT (mode-major)
    std::vector<double> Cr((size_t)nz*nr), Ci((size_t)nz*nr);
    for(int m=0;m<nr;++m) for(int k=0;k<nz;++k){ double re=0,im=0;
        for(int iz=0;iz<nz;++iz){ double an=-2.0*M_PI*iz*k/nz; re+=f[(size_t)m*nz+iz]*std::cos(an); im+=f[(size_t)m*nz+iz]*std::sin(an); }
        Cr[(size_t)k*nr+m]=re; Ci[(size_t)k*nr+m]=im; }

    double *d_f,*d_Xr,*d_Xi,*d_f2;
    cudaMalloc(&d_f,f.size()*sizeof(double)); cudaMalloc(&d_Xr,(size_t)nz*nr*sizeof(double));
    cudaMalloc(&d_Xi,(size_t)nz*nr*sizeof(double)); cudaMalloc(&d_f2,f.size()*sizeof(double));
    cudaMemcpy(d_f,f.data(),f.size()*sizeof(double),cudaMemcpyHostToDevice);
    int B=128;
    k_dft <<<(nr*nz+B-1)/B,B>>>(nr,nz,d_f,d_Xr,d_Xi);
    k_idft<<<(nr*nz+B-1)/B,B>>>(nr,nz,d_Xr,d_Xi,d_f2);
    cudaDeviceSynchronize();

    std::vector<double> Xr((size_t)nz*nr),Xi((size_t)nz*nr),f2(f.size());
    cudaMemcpy(Xr.data(),d_Xr,Xr.size()*sizeof(double),cudaMemcpyDeviceToHost);
    cudaMemcpy(Xi.data(),d_Xi,Xi.size()*sizeof(double),cudaMemcpyDeviceToHost);
    cudaMemcpy(f2.data(),d_f2,f2.size()*sizeof(double),cudaMemcpyDeviceToHost);

    double dft_err=0, rt_err=0, fscale=0;
    for(size_t i=0;i<Cr.size();++i) dft_err=std::max(dft_err,std::max(std::fabs(Xr[i]-Cr[i]),std::fabs(Xi[i]-Ci[i])));
    for(size_t i=0;i<f.size();++i){ rt_err=std::max(rt_err,std::fabs(f2[i]-f[i])); fscale=std::max(fscale,std::fabs(f[i])); }
    std::printf("DFT_z forward GPU vs CPU: max=%.3e ; f->DFT->IDFT->f round-trip: max=%.3e (rel %.3e)\n",
                dft_err, rt_err, rt_err/std::max(fscale,1e-300));
    return 0;
}
