// mms_hill_metric.cpp
//
// Metric-correctness gate for the PHYSICAL (Lx:Ly) hill operators.  The standard
// best-approximation MMS is scale-invariant (pure mass projection) and cannot
// catch a metric bug.  This gate proves the physical operators are the EXACT
// chain-rule scaling of the verified unit-square operators:
//
//   physical x = Lx x',  y = Ly y'  =>  volume Jacobian Lx*Ly,  d/dx = (1/Lx) d/dx'
//   weak forms:  M_phys  = (Lx*Ly) M_unit
//                Qx_phys = (Lx*Ly)(1/Lx) Qx_unit = Ly Qx_unit
//                Qy_phys = (Lx*Ly)(1/Ly) Qy_unit = Lx Qy_unit
//
// If these hold to roundoff, the physical first-derivative + mass metric (which
// drive advection, divergence, pressure-gradient in the DNS) inherit the
// unit-square operators' verification.  The aggregation/active-set is metric-
// invariant, so both builds share the reduced space (same nr2), making the
// comparison element-wise.

#include <cmath>
#include <cstdio>
#include "hill3d.h"

static double reldiff(const Mat& A, const Mat& B, double s){
    double mx=0, me=0;
    for (int i=0;i<A.r;++i) for (int j=0;j<A.c;++j){
        mx=std::max(mx,std::fabs(A(i,j)));
        me=std::max(me,std::fabs(A(i,j)-s*B(i,j)));
    }
    return me/(mx+1e-300);
}

int main(){
    std::printf("=================================================================\n");
    std::printf(" MMS gate: PHYSICAL Lx:Ly metric on the hill operators\n");
    std::printf(" physical_op must equal chain-rule scaling of the verified unit op\n");
    std::printf("=================================================================\n\n");
    bool all_ok = true;
    for (int ne : {6,8,12}) for (int p : {2,3,4}){
        const int nz=8; const double Lz=4.5;
        Hill3D Hu; Hu.build(ne,p,nz,Lz,/*physical=*/false);
        Hill3D Hp; Hp.build(ne,p,nz,Lz,/*physical=*/true);
        const double Lx=Hp.HB.Lx, Ly=Hp.HB.Ly;
        if (Hp.nr2 != Hu.nr2){ std::printf("  nr2 mismatch (metric changed aggregation!) %d vs %d\n",Hp.nr2,Hu.nr2); all_ok=false; continue; }
        const double rM  = reldiff(Hp.M2,  Hu.M2,  Lx*Ly);
        const double rQx = reldiff(Hp.Qx2, Hu.Qx2, Ly);
        const double rQy = reldiff(Hp.Qy2, Hu.Qy2, Lx);
        const bool ok = (rM<1e-12 && rQx<1e-12 && rQy<1e-12);
        all_ok = all_ok && ok;
        std::printf("  ne=%2d p=%d nr2=%4d : M2 %.2e  Qx2 %.2e  Qy2 %.2e   %s\n",
                    ne,p,Hp.nr2,rM,rQx,rQy, ok?"PASS":"FAIL");
    }
    std::printf("\n => physical metric %s (chain-rule scaling of verified unit operators)\n",
                all_ok?"VERIFIED to roundoff":"FAILED");
    return all_ok?0:1;
}
