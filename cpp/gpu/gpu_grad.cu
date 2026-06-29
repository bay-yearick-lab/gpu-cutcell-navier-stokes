// gpu_grad.cu
//
// GPU brick 5: hill_grad (sampled gradient, all 3 directions) reusing the SAME
// brick-2 kernels (k_xy CSR-gather + k_z dense-z).  hill_grad is structurally
// identical to hill_sample; only the CSR value array (sp_dpx / sp_dpy / sp_phi)
// and the z-matrix (Vz for x,y ; dVz for z) change.  This proves the advection
// sample-side needs NO new kernel beyond a pointwise multiply.  Verified vs CPU
// hill_grad to roundoff for dir = 0,1,2 on a real Hill3D.

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <cuda_runtime.h>
#include "hill3d.h"

__global__ void k_xy(int n2q,int nz,const int* row_ptr,const int* col,const double* val,
                     const double* c,double* tmp){
    int t = blockIdx.x*blockDim.x + threadIdx.x; if(t>=n2q*nz) return;
    int q=t/nz, iz=t%nz; double s=0.0;
    for(int a=row_ptr[q]; a<row_ptr[q+1]; ++a) s += val[a]*c[(size_t)col[a]*nz+iz];
    tmp[(size_t)q*nz+iz]=s;
}
__global__ void k_z(int n2q,int nz,int nzq,const double* B,const double* tmp,double* out){
    int t = blockIdx.x*blockDim.x + threadIdx.x; if(t>=n2q*nzq) return;
    int q=t/nzq, qz=t%nzq; double s=0.0;
    for(int iz=0; iz<nz; ++iz) s += B[(size_t)qz*nz+iz]*tmp[(size_t)q*nz+iz];
    out[(size_t)q*nzq+qz]=s;
}

int main(){
    const int ne=8, p=3, nz=16; const double Lz=4.5;
    Hill3D H; H.build(ne,p,nz,Lz);
    const int n2q=H.n2q, nzq=H.nzq, nr2=H.nr2;

    FieldH c(nr2, nz); srand(555);
    for(auto& v: c.v) v=(double)rand()/RAND_MAX-0.5;

    // device buffers reused across dirs
    double *d_c,*d_tmp,*d_B,*d_out,*d_val; int *d_rp,*d_col;
    cudaMalloc(&d_c,c.v.size()*sizeof(double)); cudaMemcpy(d_c,c.v.data(),c.v.size()*sizeof(double),cudaMemcpyHostToDevice);
    cudaMalloc(&d_tmp,(size_t)n2q*nz*sizeof(double));
    cudaMalloc(&d_out,(size_t)n2q*nzq*sizeof(double));
    cudaMalloc(&d_rp,(n2q+1)*sizeof(int));
    int B=128;

    for(int dir=0; dir<3; ++dir){
        const std::vector<std::vector<double>>& SV = (dir==0)? H.sp_dpx : (dir==1)? H.sp_dpy : H.sp_phi;
        const Mat& Bz = (dir==2)? H.dVz : H.Vz;
        QField ref = hill_grad(H, c, dir);

        std::vector<int> row_ptr(n2q+1,0), col; std::vector<double> val;
        for(int q=0;q<n2q;++q){ row_ptr[q+1]=row_ptr[q]+(int)H.sp_idx[q].size();
            for(size_t a=0;a<H.sp_idx[q].size();++a){ col.push_back(H.sp_idx[q][a]); val.push_back(SV[q][a]); } }
        const int nnz=(int)col.size();
        cudaMemcpy(d_rp,row_ptr.data(),(n2q+1)*sizeof(int),cudaMemcpyHostToDevice);
        cudaMalloc(&d_col,nnz*sizeof(int)); cudaMalloc(&d_val,nnz*sizeof(double)); cudaMalloc(&d_B,Bz.a.size()*sizeof(double));
        cudaMemcpy(d_col,col.data(),nnz*sizeof(int),cudaMemcpyHostToDevice);
        cudaMemcpy(d_val,val.data(),nnz*sizeof(double),cudaMemcpyHostToDevice);
        cudaMemcpy(d_B,Bz.a.data(),Bz.a.size()*sizeof(double),cudaMemcpyHostToDevice);

        k_xy<<<(n2q*nz+B-1)/B,B>>>(n2q,nz,d_rp,d_col,d_val,d_c,d_tmp);
        k_z <<<(n2q*nzq+B-1)/B,B>>>(n2q,nz,nzq,d_B,d_tmp,d_out);
        cudaDeviceSynchronize();
        std::vector<double> out((size_t)n2q*nzq);
        cudaMemcpy(out.data(),d_out,out.size()*sizeof(double),cudaMemcpyDeviceToHost);
        double maxd=0,maxr=0;
        for(size_t i=0;i<out.size();++i){ maxd=std::max(maxd,std::fabs(out[i]-ref.v[i])); maxr=std::max(maxr,std::fabs(ref.v[i])); }
        const char* nm = (dir==0)?"d/dx":(dir==1)?"d/dy":"d/dz";
        std::printf("hill_grad dir=%d (%s): nnz=%d  max|gpu-cpu|=%.3e  rel=%.3e\n", dir, nm, nnz, maxd, maxd/std::max(maxr,1e-300));
        cudaFree(d_col); cudaFree(d_val); cudaFree(d_B);
    }
    cudaFree(d_c);cudaFree(d_tmp);cudaFree(d_out);cudaFree(d_rp);
    return 0;
}
