// stats_hill.h -- online Reynolds-stress statistics at the 10 reference periodic-hill
// stations, in the hillturb_stats JSON format used for the
// reference comparison.  Host-side (cheap, every K steps): copy the reduced field, sample u
// at each station's y-profile x all spanwise quad nodes, z-average the first/second
// moments (z is homogeneous), accumulate in time.  Physical units (H=1, normalized by u_b).
#ifndef STATS_HILL_H
#define STATS_HILL_H
#include <array>
#include <vector>
#include <string>
#include <cstdio>
#include <cmath>
#include "hill3d.h"

struct StationStats {
    static const int NS = 10, NY = 64;
    double xH[NS] = {0.05,0.5,1,2,3,4,5,6,7,8};
    struct SampPt { std::vector<int> m; std::vector<double> w; };
    std::vector<std::vector<SampPt>> samp;            // [NS][NY] reduced sample (mode,weight)
    std::vector<std::array<double,NY>> yphys;         // [NS] physical y nodes
    std::vector<std::array<double,NY>> su,sv,sw,suu,svv,sww,suv;
    long n = 0;

    // reduced-basis sample weights at parametric (xp,yp): phi_full . E -> reduced modes
    static void reduced_sample(const Hill3D& H, double xp, double yp,
                               std::vector<int>& ms, std::vector<double>& ws){
        std::vector<int> rows; std::vector<double> phi;
        cut2d_point_basis(H.cf2, xp, yp, rows, phi);
        std::vector<double> acc(H.nr2, 0.0);
        for (size_t a=0;a<rows.size();++a){ const int b=rows[a];
            for (int mm=0; mm<H.nr2; ++mm){ const double e=H.ext2.E(b,mm); if (e!=0.0) acc[mm]+=phi[a]*e; } }
        ms.clear(); ws.clear();
        for (int mm=0; mm<H.nr2; ++mm) if (acc[mm]!=0.0){ ms.push_back(mm); ws.push_back(acc[mm]); }
    }

    void build(const Hill3D& H){
        const double Lx=H.HB.Lx, Ly=H.HB.Ly, eps=1e-4;
        samp.assign(NS, std::vector<SampPt>(NY));
        yphys.assign(NS, std::array<double,NY>{});
        su.assign(NS,std::array<double,NY>{}); sv=su; sw=su; suu=su; svv=su; sww=su; suv=su;
        for (int i=0;i<NS;++i){ const double xs=xH[i]; const double hw=H.HB.hill.h(xs); const double xp=xs/Lx;
            for (int j=0;j<NY;++j){ const double yj=(hw+eps)+((Ly-eps)-(hw+eps))*j/(NY-1); yphys[i][j]=yj;
                reduced_sample(H, xp, yj/Ly, samp[i][j].m, samp[i][j].w); } }
        n=0;
    }

    void accumulate(const Hill3D& H, const FieldH& ux,const FieldH& uy,const FieldH& uz){
        const int nz=H.Z.nz, nzq=H.nzq; const double iLz=1.0/H.Z.Lz;
        std::vector<double> ax(nz),ay(nz),az(nz);
        for (int i=0;i<NS;++i) for (int j=0;j<NY;++j){
            const SampPt& s=samp[i][j];
            for (int iz=0;iz<nz;++iz){ double sx=0,sy=0,sz=0;
                for (size_t a=0;a<s.m.size();++a){ const int m=s.m[a]; const double w=s.w[a];
                    sx+=w*ux.at(m,iz); sy+=w*uy.at(m,iz); sz+=w*uz.at(m,iz); }
                ax[iz]=sx; ay[iz]=sy; az[iz]=sz; }
            double mU=0,mV=0,mW=0,mUU=0,mVV=0,mWW=0,mUV=0;
            for (int qz=0;qz<nzq;++qz){ double ux_=0,uy_=0,uz_=0;
                for (int iz=0;iz<nz;++iz){ const double v=H.Vz(qz,iz); ux_+=v*ax[iz]; uy_+=v*ay[iz]; uz_+=v*az[iz]; }
                const double w=H.wz[qz];
                mU+=w*ux_; mV+=w*uy_; mW+=w*uz_; mUU+=w*ux_*ux_; mVV+=w*uy_*uy_; mWW+=w*uz_*uz_; mUV+=w*ux_*uy_; }
            su[i][j]+=mU*iLz; sv[i][j]+=mV*iLz; sw[i][j]+=mW*iLz;
            suu[i][j]+=mUU*iLz; svv[i][j]+=mVV*iLz; sww[i][j]+=mWW*iLz; suv[i][j]+=mUV*iLz;
        }
        ++n;
    }

    // binary checkpoint of the accumulators (for resume across teardowns)
    void save(FILE* f) const {
        std::fwrite(&n,sizeof(long),1,f);
        auto wr=[&](const std::vector<std::array<double,NY>>& a){ for(auto&r:a) std::fwrite(r.data(),sizeof(double),NY,f); };
        wr(su); wr(sv); wr(sw); wr(suu); wr(svv); wr(sww); wr(suv);
    }
    void load(FILE* f){
        if(std::fread(&n,sizeof(long),1,f)!=1) return;
        auto rd=[&](std::vector<std::array<double,NY>>& a){ for(auto&r:a){ size_t g=std::fread(r.data(),sizeof(double),NY,f); (void)g; } };
        rd(su); rd(sv); rd(sw); rd(suu); rd(svv); rd(sww); rd(suv);
    }

    // U=su/n, uu=max(suu/n-U^2,0), uv=suv/n-U*V ; normalized by u_b (H=1 already in xH,yphys)
    void write_json(const std::string& path, const Hill3D& H, double u_b, int reynolds,
                    int step, double sim_time, double u_tau) const {
        const long nn = n>0? n : 1; const double ub=(u_b!=0.0?u_b:1.0);
        FILE* f=std::fopen(path.c_str(),"w"); if(!f){ std::printf("stats write FAILED %s\n",path.c_str()); return; }
        std::fprintf(f,"{\n  \"status\": \"gpu_dns\",\n");
        std::fprintf(f,"  \"meta\": {\"reynolds\": %d, \"nu\": %.10g, \"p\": %d, \"nx\": %d, \"ny\": %d, \"Lz\": %.6g, \"Nz\": %d, \"n_modes\": %d, \"dt\": 0.02, \"target_bulk\": %.10g, \"dns_source\": \"reference\"},\n",
                     reynolds, (reynolds>0?u_b/reynolds:0.0), H.cf2.degree, H.cf2.n_elements, H.cf2.n_elements, H.Z.Lz, H.Z.nz, H.Z.nz, u_b);
        std::fprintf(f,"  \"averaging_window\": {\"n_samples\": %ld, \"step\": %d, \"sim_time\": %.6g, \"eddy_turnover_time\": 9.0, \"under_averaged\": %s},\n",
                     n, step, sim_time, (n<200?"true":"false"));
        std::fprintf(f,"  \"running_means\": {\"u_tau\": %.8g, \"bulk_velocity\": %.8g, \"reynolds\": %.8g},\n", u_tau, u_b, (double)reynolds);
        std::fprintf(f,"  \"stations\": [\n");
        auto arr=[&](const char* key, const double* a, double sc){ std::fprintf(f,"\"%s\": [",key);
            for(int j=0;j<NY;++j) std::fprintf(f,"%s%.9g", j?", ":"", a[j]*sc); std::fprintf(f,"]"); };
        for (int i=0;i<NS;++i){
            std::array<double,NY> U,V,W,uu,vv,ww,uv,yy;
            for(int j=0;j<NY;++j){ U[j]=su[i][j]/nn; V[j]=sv[i][j]/nn; W[j]=sw[i][j]/nn;
                uu[j]=std::max(suu[i][j]/nn-U[j]*U[j],0.0); vv[j]=std::max(svv[i][j]/nn-V[j]*V[j],0.0);
                ww[j]=std::max(sww[i][j]/nn-W[j]*W[j],0.0); uv[j]=suv[i][j]/nn-U[j]*V[j]; yy[j]=yphys[i][j]; }
            std::fprintf(f,"    {\"x_over_H\": %.6g, ", xH[i]);
            arr("y",yy.data(),1.0); std::fprintf(f,", "); arr("U",U.data(),1.0/ub); std::fprintf(f,", ");
            arr("V",V.data(),1.0/ub); std::fprintf(f,", "); arr("W",W.data(),1.0/ub); std::fprintf(f,", ");
            arr("uu",uu.data(),1.0/(ub*ub)); std::fprintf(f,", "); arr("vv",vv.data(),1.0/(ub*ub)); std::fprintf(f,", ");
            arr("ww",ww.data(),1.0/(ub*ub)); std::fprintf(f,", "); arr("uv",uv.data(),1.0/(ub*ub));
            std::fprintf(f,"}%s\n", i+1<NS?",":"");
        }
        std::fprintf(f,"  ],\n  \"wall\": {\"x_over_H\": [], \"cf\": [], \"cp\": [], \"reattachment_x_over_H\": null}\n}\n");
        std::fclose(f);
    }
};
#endif
