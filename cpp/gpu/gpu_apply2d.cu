// gpu_apply2d.cu
//
// GPU brick for the DNS-scale port: the dense 2D operator apply that ALL of the
// hill step routes through -- hill_apply2d computes, for every spanwise z-slice,
//   OUT(:,iz) = A . FIELD(:,iz),
// i.e. one GEMM  OUT(nr2 x nz) = A(nr2 x nr2) . FIELD(nr2 x nz)  applied with
// A in {M2inv, Qx2, Qy2, KN2, M2}.  This verifies cuBLAS dgemm + sm_120 linkage
// against a plain CPU matmul to the roundoff floor (column-major throughout).

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <cuda_runtime.h>
#include <cublas_v2.h>

int main(){
    const int nr2 = 111;   // representative reduced 2D DOF (ne=8,p=3)
    const int nz  = 16;    // spanwise slices
    std::vector<double> A((size_t)nr2*nr2), X((size_t)nr2*nz), Cgpu((size_t)nr2*nz), Ccpu((size_t)nr2*nz);
    srand(12345);
    auto rnd=[&](){ return (double)rand()/RAND_MAX - 0.5; };
    for(auto&a:A) a=rnd();
    for(auto&x:X) x=rnd();

    // CPU reference (column-major): C[i+j*nr2] = sum_l A[i+l*nr2]*X[l+j*nr2]
    for(int j=0;j<nz;++j) for(int i=0;i<nr2;++i){ double s=0;
        for(int l=0;l<nr2;++l) s += A[(size_t)i+(size_t)l*nr2]*X[(size_t)l+(size_t)j*nr2];
        Ccpu[(size_t)i+(size_t)j*nr2]=s; }

    // GPU via cuBLAS
    double *dA,*dX,*dC;
    cudaMalloc(&dA,A.size()*sizeof(double)); cudaMalloc(&dX,X.size()*sizeof(double)); cudaMalloc(&dC,Cgpu.size()*sizeof(double));
    cudaMemcpy(dA,A.data(),A.size()*sizeof(double),cudaMemcpyHostToDevice);
    cudaMemcpy(dX,X.data(),X.size()*sizeof(double),cudaMemcpyHostToDevice);
    cublasHandle_t h; cublasCreate(&h);
    const double one=1.0, zero=0.0;
    cublasStatus_t st = cublasDgemm(h, CUBLAS_OP_N, CUBLAS_OP_N, nr2, nz, nr2,
                                    &one, dA, nr2, dX, nr2, &zero, dC, nr2);
    cudaDeviceSynchronize();
    cudaMemcpy(Cgpu.data(),dC,Cgpu.size()*sizeof(double),cudaMemcpyDeviceToHost);

    double maxd=0, maxc=0;
    for(size_t i=0;i<Ccpu.size();++i){ maxd=std::max(maxd,std::fabs(Ccpu[i]-Cgpu[i])); maxc=std::max(maxc,std::fabs(Ccpu[i])); }
    std::printf("cuBLAS dgemm status=%d  nr2=%d nz=%d  max|gpu-cpu|=%.3e  rel=%.3e\n",
                (int)st, nr2, nz, maxd, maxd/std::max(maxc,1e-300));

    // throughput: many GEMMs (the dense applies per step ~ O(10))
    cudaEvent_t e0,e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
    const int reps=2000; cudaEventRecord(e0);
    for(int r=0;r<reps;++r) cublasDgemm(h,CUBLAS_OP_N,CUBLAS_OP_N,nr2,nz,nr2,&one,dA,nr2,dX,nr2,&zero,dC,nr2);
    cudaEventRecord(e1); cudaEventSynchronize(e1);
    float ms=0; cudaEventElapsedTime(&ms,e0,e1);
    std::printf("  %d GEMMs in %.2f ms => %.1f us/GEMM (%.1f GFLOP/s)\n",
                reps, ms, 1e3*ms/reps, 2.0*nr2*nr2*nz*reps/(ms*1e-3)/1e9);

    cublasDestroy(h); cudaFree(dA); cudaFree(dX); cudaFree(dC);
    return 0;
}
