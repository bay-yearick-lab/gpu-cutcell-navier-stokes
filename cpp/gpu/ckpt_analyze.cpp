// ckpt_analyze.cpp -- is the developed state TURBULENT or LAMINAR, and what is the CFL?
// Loads a gpu_dns checkpoint (field), builds the matching Hill3D, and reports:
//   KE_total, KE of the spanwise-mean field, fluctuation fraction E'/E (=0 laminar/2D,
//   large = 3D turbulence), max|u| and the advective CFL = max|u| dt / Dmin.
#include <cstdio>
#include <cmath>
#include <vector>
#include "hill3d.h"

int main(int argc,char**argv){
    const char* ck = argc>1?argv[1]:"ckpt_ne24.bin";
    const int ne=argc>2?atoi(argv[2]):24, nz=argc>3?atoi(argv[3]):16;
    const double dt=argc>4?atof(argv[4]):0.005, Lz=4.5;
    Hill3D H; H.build(ne,3,nz,Lz,true);
    const int NC=H.nr2*nz;
    FILE* f=std::fopen(ck,"rb"); if(!f){ std::printf("no ckpt %s\n",ck); return 1; }
    int step,nc2; double fx,integ; size_t g;
    g=std::fread(&step,sizeof(int),1,f); g=std::fread(&fx,sizeof(double),1,f); g=std::fread(&integ,sizeof(double),1,f); g=std::fread(&nc2,sizeof(int),1,f);
    if(nc2!=NC){ std::printf("NC mismatch ck=%d build=%d (wrong ne/nz?)\n",nc2,NC); return 1; }
    HillNS s; s.ux=FieldH(H.nr2,nz); s.uy=FieldH(H.nr2,nz); s.uz=FieldH(H.nr2,nz);
    g=std::fread(s.ux.v.data(),sizeof(double),NC,f); g=std::fread(s.uy.v.data(),sizeof(double),NC,f); g=std::fread(s.uz.v.data(),sizeof(double),NC,f); (void)g;
    std::fclose(f);

    const double KE=hill_ke(H,s);
    // spanwise-mean field: cbar(m)=(1/nz) sum_iz c(m,iz), replicated in z
    HillNS sm; sm.ux=FieldH(H.nr2,nz); sm.uy=FieldH(H.nr2,nz); sm.uz=FieldH(H.nr2,nz);
    auto zmean=[&](const FieldH& c, FieldH& o){ for(int m=0;m<H.nr2;++m){ double a=0; for(int iz=0;iz<nz;++iz)a+=c.at(m,iz); a/=nz; for(int iz=0;iz<nz;++iz)o.at(m,iz)=a; } };
    zmean(s.ux,sm.ux); zmean(s.uy,sm.uy); zmean(s.uz,sm.uz);
    const double KEm=hill_ke(H,sm);
    const double fluct = 1.0 - KEm/std::max(KE,1e-300);

    // max|u| over the quadrature grid + WHERE it is (wall artifact?) + percentiles
    QField ux=hill_sample(H,s.ux), uy=hill_sample(H,s.uy), uz=hill_sample(H,s.uz);
    double umax=0; size_t iarg=0; std::vector<double> mags(ux.v.size());
    for(size_t i=0;i<ux.v.size();++i){ double m=std::sqrt(ux.v[i]*ux.v[i]+uy.v[i]*uy.v[i]+uz.v[i]*uz.v[i]); mags[i]=m; if(m>umax){umax=m;iarg=i;} }
    const double dmin=std::min(std::min(H.HB.Lx/ne, H.HB.Ly/ne), Lz/nz);
    const double cfl = umax*dt/dmin;
    // location of the max: quad index iarg -> (q2, qz); physical (x,y), wall distance
    const int q2=(int)(iarg/H.nzq);
    const double xp=H.fq2.x[q2], yp=H.fq2.y[q2];                 // parametric [0,1]
    const double xphys=xp*H.HB.Lx, yphys=yp*H.HB.Ly, wallh=H.HB.hill.h(xphys);
    std::vector<double> sm2=mags; std::sort(sm2.begin(),sm2.end());
    auto pct=[&](double p){ return sm2[(size_t)(p*(sm2.size()-1))]; };
    std::printf("  |u| percentiles: 50%%=%.3f 99%%=%.3f 99.9%%=%.3f max=%.3f  (U_b~1; physical hill max ~2-3)\n",
                pct(0.50),pct(0.99),pct(0.999),umax);
    std::printf("  max|u| at x/H=%.3f y=%.4f (wall h=%.4f, dist-from-wall=%.4f) %s\n",
                xphys, yphys, wallh, yphys-wallh, (yphys-wallh<0.05)?"<-- AT THE WALL (no-slip should be ~0: ARTIFACT)":"(interior)");

    std::printf("ckpt %s : step=%d fx=%.4f\n", ck, step, fx);
    std::printf("  KE_total=%.4f  KE_zmean=%.4f  FLUCT_FRAC E'/E=%.4f  => %s\n",
                KE, KEm, fluct, fluct<0.02?"LAMINAR/2D (z-uniform, NOT turbulent)":(fluct<0.1?"weak 3D":"TURBULENT (substantial 3D fluctuations)"));
    std::printf("  max|u|=%.3f  Dmin=%.4f (Ly/ne)  CFL=max|u|*dt/Dmin=%.3f  (RK3 stable <~1.4)\n", umax, dmin, cfl);
    return 0;
}
