// gpu_load.cu
//
// GPU brick 3: the Galerkin load hill_load (ADJOINT of hill_sample), verified
// against the CPU reference to the roundoff floor.  hill_load is:
//   tmp(q,iz) = sum_qz Vz(qz,iz) * wz[qz] * fq(q,qz)         (dense z-integrate)
//   red(m,iz) += sp_phi[q][a] * (w2[q]*tmp(q,iz))  for m=sp_idx[q][a]  (sparse SCATTER)
// The scatter has overlapping basis supports, so the GPU uses atomicAdd; the
// reduction order differs from the CPU's, so the gate is a roundoff match (not
// bit-identical).  Built on a REAL Hill3D.

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <cuda_runtime.h>
#include "hill3d.h"

__global__ void k_zint(int n2q,int nz,int nzq,const double* Vz,const double* wz,
                       const double* fq,double* tmp){
    int t = blockIdx.x*blockDim.x + threadIdx.x;
    if(t >= n2q*nz) return;
    int q = t/nz, iz = t%nz;
    double s = 0.0;
    for(int qz=0; qz<nzq; ++qz) s += Vz[(size_t)qz*nz+iz]*wz[qz]*fq[(size_t)q*nzq+qz];
    tmp[(size_t)q*nz+iz] = s;
}
__global__ void k_scatter(int n2q,int nz,const int* row_ptr,const int* col,const double* val,
                          const double* w2,const double* tmp,double* red){
    int t = blockIdx.x*blockDim.x + threadIdx.x;
    if(t >= n2q*nz) return;
    int q = t/nz, iz = t%nz;
    double tv = w2[q]*tmp[(size_t)q*nz+iz];
    for(int a=row_ptr[q]; a<row_ptr[q+1]; ++a)
        atomicAdd(&red[(size_t)col[a]*nz+iz], val[a]*tv);
}

int main(){
    const int ne=8, p=3, nz=16; const double Lz=4.5;
    Hill3D H; H.build(ne,p,nz,Lz);
    const int n2q=H.n2q, nzq=H.nzq, nr2=H.nr2;

    // random quadrature field fq (n2q x nzq)
    QField fq(n2q, nzq); srand(2024);
    for(auto& v: fq.v) v = (double)rand()/RAND_MAX - 0.5;

    // CPU reference
    FieldH ref = hill_load(H, fq);

    // CSR
    std::vector<int> row_ptr(n2q+1,0), col; std::vector<double> val;
    for(int q=0;q<n2q;++q){ row_ptr[q+1]=row_ptr[q]+(int)H.sp_idx[q].size();
        for(size_t a=0;a<H.sp_idx[q].size();++a){ col.push_back(H.sp_idx[q][a]); val.push_back(H.sp_phi[q][a]); } }
    const int nnz=(int)col.size();

    int *d_rp,*d_col; double *d_val,*d_fq,*d_tmp,*d_Vz,*d_wz,*d_w2,*d_red;
    cudaMalloc(&d_rp,(n2q+1)*sizeof(int)); cudaMalloc(&d_col,nnz*sizeof(int)); cudaMalloc(&d_val,nnz*sizeof(double));
    cudaMalloc(&d_fq,fq.v.size()*sizeof(double)); cudaMalloc(&d_tmp,(size_t)n2q*nz*sizeof(double));
    cudaMalloc(&d_Vz,H.Vz.a.size()*sizeof(double)); cudaMalloc(&d_wz,H.wz.size()*sizeof(double));
    cudaMalloc(&d_w2,H.w2.size()*sizeof(double)); cudaMalloc(&d_red,(size_t)nr2*nz*sizeof(double));
    cudaMemcpy(d_rp,row_ptr.data(),(n2q+1)*sizeof(int),cudaMemcpyHostToDevice);
    cudaMemcpy(d_col,col.data(),nnz*sizeof(int),cudaMemcpyHostToDevice);
    cudaMemcpy(d_val,val.data(),nnz*sizeof(double),cudaMemcpyHostToDevice);
    cudaMemcpy(d_fq,fq.v.data(),fq.v.size()*sizeof(double),cudaMemcpyHostToDevice);
    cudaMemcpy(d_Vz,H.Vz.a.data(),H.Vz.a.size()*sizeof(double),cudaMemcpyHostToDevice);
    cudaMemcpy(d_wz,H.wz.data(),H.wz.size()*sizeof(double),cudaMemcpyHostToDevice);
    cudaMemcpy(d_w2,H.w2.data(),H.w2.size()*sizeof(double),cudaMemcpyHostToDevice);
    cudaMemset(d_red,0,(size_t)nr2*nz*sizeof(double));

    int B=128;
    k_zint   <<<(n2q*nz+B-1)/B,B>>>(n2q,nz,nzq,d_Vz,d_wz,d_fq,d_tmp);
    k_scatter<<<(n2q*nz+B-1)/B,B>>>(n2q,nz,d_rp,d_col,d_val,d_w2,d_tmp,d_red);
    cudaError_t err=cudaDeviceSynchronize();
    if(err!=cudaSuccess){ std::printf("CUDA error: %s\n",cudaGetErrorString(err)); return 1; }

    std::vector<double> red((size_t)nr2*nz);
    cudaMemcpy(red.data(),d_red,red.size()*sizeof(double),cudaMemcpyDeviceToHost);

    double maxd=0,maxr=0;
    for(size_t i=0;i<red.size();++i){ maxd=std::max(maxd,std::fabs(red[i]-ref.v[i])); maxr=std::max(maxr,std::fabs(ref.v[i])); }
    std::printf("hill_load GPU(atomic) vs CPU: nr2=%d nz=%d nnz=%d  max|gpu-cpu|=%.3e  rel=%.3e  (ref-scale=%.3e)\n",
                nr2, nz, nnz, maxd, maxd/std::max(maxr,1e-300), maxr);
    return 0;
}
