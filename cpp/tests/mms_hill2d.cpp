// mms_hill2d.cpp
//
// MMS / identity gate for the 2D curved periodic-hill cut (hill_cut2d.h), the
// capstone geometry (sec:s6-hill).  Reuses the verified cut2d aggregation.
//   (1) curved-wall Green identities ||Qx+Qx^T-Sx||, ||Qy+Qy^T-Sy||;
//   (2) tensor-monomial reproduction by E (pure aggregation -> roundoff);
//   (3) best-approximation order p+1 on the height-function cut.

#include <cmath>
#include <cstdio>
#include <vector>
#include "hill_cut2d.h"

static double u_exact(double x,double y){ return std::sin(2.3*x+0.7*y+0.4)+0.5*std::cos(1.1*x-1.7*y); }

int main(){
    std::printf("=================================================================\n");
    std::printf(" MMS gate: 2D curved periodic-hill cut (hill_cut2d.h)\n");
    std::printf(" fluid { y >= h(x) } on the unit square, hbar(x)=h(Lx x)/Ly\n");
    std::printf("=================================================================\n\n");
    HillBar H;   // default periodic hill, Lx=9, Ly=3.036
    for (int p=2;p<=4;++p){
        std::printf("  degree p=%d (formal best-approx p+1=%d)\n", p, p+1);
        std::printf("   N_el   h        gamma_min  #small  Green_x    Green_y    polyE      L2err      order\n");
        double prev=0,prevh=0,mg=0,mp=0;
        for (int ne : {8,12,16,24}){
            const double hh=1.0/ne;
            CutForms2D cf = assemble_hill_cut_2d(ne, p, H);
            ActiveSets2D as = cut2d_active_sets(cf);
            Extension2D ext = cut2d_extension(cf, as);
            // Gate 1: curved-wall Green identities.
            double gx=0,gy=0;
            for(int i=0;i<cf.n;++i)for(int j=0;j<cf.n;++j){
                gx=std::max(gx,std::fabs(cf.Qx(i,j)+cf.Qx(j,i)-cf.Sx(i,j)));
                gy=std::max(gy,std::fabs(cf.Qy(i,j)+cf.Qy(j,i)-cf.Sy(i,j)));
            }
            // Gate 3: reduced-space best approximation.
            Mat Hr = cut2d_pullback(cf.H, ext.E);
            FluidQuad2D fq = hill_fluid_quadrature(ne, p, H);
            std::vector<double> bfull(cf.n,0.0);
            for(size_t qq=0;qq<fq.w.size();++qq){ std::vector<int> rows; std::vector<double> phi;
                cut2d_point_basis(cf, fq.x[qq], fq.y[qq], rows, phi); const double uq=u_exact(fq.x[qq],fq.y[qq]);
                for(size_t a=0;a<rows.size();++a) bfull[rows[a]]+=fq.w[qq]*phi[a]*uq; }
            std::vector<double> br=matvec(transpose(ext.E),bfull);
            DenseMatrix Hrd=to_dense(Hr); std::vector<double> c; dense_lu_solve(Hrd,br,c);
            std::vector<double> alpha=matvec(ext.E,c);
            double e2=0; for(size_t qq=0;qq<fq.w.size();++qq){ std::vector<int> rows; std::vector<double> phi;
                cut2d_point_basis(cf, fq.x[qq], fq.y[qq], rows, phi); double uh=0; for(size_t a=0;a<rows.size();++a) uh+=alpha[rows[a]]*phi[a];
                const double d=u_exact(fq.x[qq],fq.y[qq])-uh; e2+=fq.w[qq]*d*d; }
            const double err=std::sqrt(e2);
            double gmin=1e30; for(int i:as.active) gmin=std::min(gmin,as.gamma[i]);
            const double ord=(prev>0)?std::log(prev/err)/std::log(prevh/hh):0;
            std::printf("   %4d  %.5f  %.2e   %3d    %.2e   %.2e   %.2e   %.2e   %s%.2f\n",
                ne,hh,gmin,(int)as.small.size(),gx,gy,ext.poly_residual,err,(prev>0)?"":"--",ord);
            prev=err;prevh=hh;mg=std::max(mg,std::max(gx,gy));mp=std::max(mp,ext.poly_residual);
        }
        std::printf("   => Green<=%.1e (curved-panel consistency)  polyE<=%.1e  best-approx ~ p+1\n\n",mg,mp);
    }
    std::printf("Done.\n");
    return 0;
}
