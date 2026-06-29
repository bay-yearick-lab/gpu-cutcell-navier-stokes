// gpu_sample.cu
//
// GPU brick 2: the SPARSE sample (hill_sample) on the curved-cut hill, verified
// against the CPU reference to the roundoff floor.  hill_sample is a tensor:
//   tmp(q,iz) = sum_a sp_phi[q][a] * c(sp_idx[q][a], iz)      (sparse x-y, CSR)
//   out(q,qz) = sum_iz Vz(qz,iz) * tmp(q,iz)                  (dense z)
// Stage 1 is the custom CSR kernel (k_xy); stage 2 the dense z kernel (k_z).
// Builds a REAL Hill3D so the sparsity, basis values and Vz are the production
// ones, not a mock.

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <cuda_runtime.h>
#include "hill3d.h"

__global__ void k_xy(int n2q,int nz,const int* row_ptr,const int* col,const double* val,
                     const double* c,double* tmp){
    int t = blockIdx.x*blockDim.x + threadIdx.x;
    if(t >= n2q*nz) return;
    int q = t/nz, iz = t%nz;
    double s = 0.0;
    for(int a=row_ptr[q]; a<row_ptr[q+1]; ++a) s += val[a]*c[(size_t)col[a]*nz+iz];
    tmp[(size_t)q*nz+iz] = s;
}
__global__ void k_z(int n2q,int nz,int nzq,const double* Vz,const double* tmp,double* out){
    int t = blockIdx.x*blockDim.x + threadIdx.x;
    if(t >= n2q*nzq) return;
    int q = t/nzq, qz = t%nzq;
    double s = 0.0;
    for(int iz=0; iz<nz; ++iz) s += Vz[(size_t)qz*nz+iz]*tmp[(size_t)q*nz+iz];
    out[(size_t)q*nzq+qz] = s;
}

int main(){
    const int ne=8, p=3, nz=16; const double Lz=4.5;
    Hill3D H; H.build(ne,p,nz,Lz);
    const int n2q=H.n2q, nzq=H.nzq, nr2=H.nr2;
    std::printf("Hill3D: nr2=%d nz=%d  n2q=%d nzq=%d\n", nr2, nz, n2q, nzq);

    // random reduced field c (nr2 x nz)
    FieldH c(nr2, nz); srand(777);
    for(auto& v: c.v) v = (double)rand()/RAND_MAX - 0.5;

    // CPU reference
    QField ref = hill_sample(H, c);

    // CSR from sp_idx/sp_phi
    std::vector<int> row_ptr(n2q+1,0), col; std::vector<double> val;
    for(int q=0;q<n2q;++q){ row_ptr[q+1]=row_ptr[q]+(int)H.sp_idx[q].size();
        for(size_t a=0;a<H.sp_idx[q].size();++a){ col.push_back(H.sp_idx[q][a]); val.push_back(H.sp_phi[q][a]); } }
    const int nnz = (int)col.size();

    // device
    int *d_rp,*d_col; double *d_val,*d_c,*d_tmp,*d_Vz,*d_out;
    cudaMalloc(&d_rp,(n2q+1)*sizeof(int)); cudaMalloc(&d_col,nnz*sizeof(int));
    cudaMalloc(&d_val,nnz*sizeof(double)); cudaMalloc(&d_c,c.v.size()*sizeof(double));
    cudaMalloc(&d_tmp,(size_t)n2q*nz*sizeof(double)); cudaMalloc(&d_Vz,H.Vz.a.size()*sizeof(double));
    cudaMalloc(&d_out,(size_t)n2q*nzq*sizeof(double));
    cudaMemcpy(d_rp,row_ptr.data(),(n2q+1)*sizeof(int),cudaMemcpyHostToDevice);
    cudaMemcpy(d_col,col.data(),nnz*sizeof(int),cudaMemcpyHostToDevice);
    cudaMemcpy(d_val,val.data(),nnz*sizeof(double),cudaMemcpyHostToDevice);
    cudaMemcpy(d_c,c.v.data(),c.v.size()*sizeof(double),cudaMemcpyHostToDevice);
    cudaMemcpy(d_Vz,H.Vz.a.data(),H.Vz.a.size()*sizeof(double),cudaMemcpyHostToDevice);

    int B=128;
    k_xy<<<(n2q*nz+B-1)/B,B>>>(n2q,nz,d_rp,d_col,d_val,d_c,d_tmp);
    k_z <<<(n2q*nzq+B-1)/B,B>>>(n2q,nz,nzq,d_Vz,d_tmp,d_out);
    cudaError_t err=cudaDeviceSynchronize();
    if(err!=cudaSuccess){ std::printf("CUDA error: %s\n",cudaGetErrorString(err)); return 1; }

    std::vector<double> out((size_t)n2q*nzq);
    cudaMemcpy(out.data(),d_out,out.size()*sizeof(double),cudaMemcpyDeviceToHost);

    double maxd=0,maxr=0;
    for(size_t i=0;i<out.size();++i){ maxd=std::max(maxd,std::fabs(out[i]-ref.v[i])); maxr=std::max(maxr,std::fabs(ref.v[i])); }
    std::printf("hill_sample GPU vs CPU: nnz=%d  max|gpu-cpu|=%.3e  rel=%.3e  (ref-scale=%.3e)\n",
                nnz, maxd, maxd/std::max(maxr,1e-300), maxr);

    cudaFree(d_rp);cudaFree(d_col);cudaFree(d_val);cudaFree(d_c);cudaFree(d_tmp);cudaFree(d_Vz);cudaFree(d_out);
    return 0;
}
