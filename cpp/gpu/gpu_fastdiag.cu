// gpu_fastdiag.cu
//
// GPU brick 4: the fast-diag's new primitive -- the BATCHED per-z-mode dense
// solve.  FastDiagHill::solve applies a DIFFERENT nr x nr matrix per Fourier
// mode k (Ainv[k] for the viscous tent, Sinv[k] for the Schur pressure).  On the
// GPU this is cublasDgemmStridedBatched (one matrix per batch).  Verified vs the
// CPU per-mode matvec loop to the roundoff floor, on a REAL FastDiagHill.
//
// cuBLAS is column-major; Mat.a is row-major (= transpose when read col-major),
// so transa=CUBLAS_OP_T recovers the true Ainv[k]*rhs.

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include "hill3d.h"
#include "hill_fastdiag.h"

int main(){
    const int ne=6, p=3, nz=12; const double Lz=4.5;
    Hill3D H; H.build(ne,p,nz,Lz);
    FastDiagHill fd; fd.build(H, 0.01);   // a = dt*nu
    const int nr=fd.nr, nm=fd.nz;
    std::printf("FastDiagHill: nr=%d  nz(modes)=%d\n", nr, nm);

    // random per-mode RHS (nr x nm), column k = rhs for mode k
    std::vector<double> rhs((size_t)nr*nm); srand(99);
    for(auto& v: rhs) v = (double)rand()/RAND_MAX - 0.5;

    // CPU reference: out_k = Ainv[k] . rhs_k
    std::vector<double> ref((size_t)nr*nm);
    for(int k=0;k<nm;++k){ std::vector<double> r(nr); for(int i=0;i<nr;++i) r[i]=rhs[(size_t)k*nr+i];
        std::vector<double> o = matvec(fd.Ainv[k], r);
        for(int i=0;i<nr;++i) ref[(size_t)k*nr+i]=o[i]; }

    // pack Ainv[k] contiguously (each row-major nr*nr)
    std::vector<double> Apack((size_t)nr*nr*nm);
    for(int k=0;k<nm;++k) for(int i=0;i<(int)fd.Ainv[k].a.size();++i) Apack[(size_t)k*nr*nr+i]=fd.Ainv[k].a[i];

    double *dA,*dB,*dC;
    cudaMalloc(&dA,Apack.size()*sizeof(double)); cudaMalloc(&dB,rhs.size()*sizeof(double)); cudaMalloc(&dC,ref.size()*sizeof(double));
    cudaMemcpy(dA,Apack.data(),Apack.size()*sizeof(double),cudaMemcpyHostToDevice);
    cudaMemcpy(dB,rhs.data(),rhs.size()*sizeof(double),cudaMemcpyHostToDevice);

    cublasHandle_t h; cublasCreate(&h);
    const double one=1.0, zero=0.0;
    // C_k(nr x 1) = op(A_k)(nr x nr) . B_k(nr x 1) ; transa=T to undo row-major
    cublasStatus_t st = cublasDgemmStridedBatched(h, CUBLAS_OP_T, CUBLAS_OP_N,
        nr, 1, nr, &one,
        dA, nr, (long long)nr*nr,
        dB, nr, (long long)nr,
        &zero, dC, nr, (long long)nr, nm);
    cudaDeviceSynchronize();

    std::vector<double> out(ref.size());
    cudaMemcpy(out.data(),dC,out.size()*sizeof(double),cudaMemcpyDeviceToHost);

    double maxd=0,maxr=0;
    for(size_t i=0;i<out.size();++i){ maxd=std::max(maxd,std::fabs(out[i]-ref[i])); maxr=std::max(maxr,std::fabs(ref[i])); }
    std::printf("batched per-mode Ainv solve GPU vs CPU: status=%d  max|gpu-cpu|=%.3e  rel=%.3e\n",
                (int)st, maxd, maxd/std::max(maxr,1e-300));

    cublasDestroy(h); cudaFree(dA); cudaFree(dB); cudaFree(dC);
    return 0;
}
