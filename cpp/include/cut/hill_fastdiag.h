// hill_fastdiag.h
//
// Fast-diagonalization DIRECT saddle solver for the periodic-hill incompressible
// step.  The implicit viscous + pressure coupling [A G; G^T 0][u;p]=[M3 u*;0]
// decouples into independent per-z-mode 2D problems (Mz,Sz,Kz are circulant ->
// shared Fourier eigenbasis).  Each mode k gets a REAL 2D viscous A_k and a REAL
// 2D Schur S_k (the skew Kz appears squared), precomputed and inverted ONCE; a
// step is then DFT_z -> per-mode matvecs -> IDFT_z.  No CG-in-CG.  Verified
// piece-by-piece to roundoff (scratchpad/fdz_*.cpp): the direct solve returns a
// velocity with weak divergence at the roundoff floor.

#ifndef HILL_FASTDIAG_H
#define HILL_FASTDIAG_H

#include <array>
#include <cmath>
#include <vector>
#include "hill3d.h"

struct FastDiagHill {
    int nr = 0, nz = 0; double a = 0.0;             // a = dt * nu (rebuild if it changes)
    std::vector<double> mu, sig, kap;               // z symbols: Re(Mz), Re(Sz), Im(Kz)
    std::vector<Mat> Ainv, Sinv;                     // per-mode 2D inverses (nr x nr)
    Mat Qx2, Qy2, Qx2T, Qy2T, M2;

    void build(const Hill3D& H, double a_) {
        nr = H.nr2; nz = H.Z.nz; a = a_;
        Qx2 = H.Qx2; Qy2 = H.Qy2; M2 = H.M2; Qx2T = transpose(H.Qx2); Qy2T = transpose(H.Qy2);
        mu.assign(nz,0); sig.assign(nz,0); kap.assign(nz,0);
        for (int k=0;k<nz;++k){ double m=0,s=0,kp=0;
            for (int j=0;j<nz;++j){ double c=std::cos(2*M_PI*j*k/nz), sn=std::sin(2*M_PI*j*k/nz);
                m+=H.Z.M(0,j)*c; s+=H.Sz(0,j)*c; kp+=H.Kz(0,j)*sn; }
            mu[k]=m; sig[k]=s; kap[k]=kp; }
        Ainv.resize(nz); Sinv.resize(nz);
        for (int k=0;k<nz;++k){
            Mat A(nr,nr); const double c1=mu[k]+a*sig[k], c2=a*mu[k];
            for (int i=0;i<nr;++i) for (int j=0;j<nr;++j) A(i,j)=c1*M2(i,j)+c2*H.KN2(i,j);
            bool ok; Ainv[k]=inverse(A,&ok);
            Mat sx=matmul(Qx2,matmul(Ainv[k],Qx2T)), sy=matmul(Qy2,matmul(Ainv[k],Qy2T)), sz=matmul(M2,matmul(Ainv[k],M2));
            Mat S(nr,nr); for (int i=0;i<nr;++i) for (int j=0;j<nr;++j) S(i,j)=mu[k]*mu[k]*(sx(i,j)+sy(i,j))+kap[k]*kap[k]*sz(i,j);
            Sinv[k]=inverse(S,&ok);
        }
    }

    // Direct saddle solve given the advected velocity u* = (ux,uy,uz); returns the
    // weakly divergence-free velocity (the implicit viscous + pressure projection).
    std::array<FieldH,3> solve(const FieldH& ux, const FieldH& uy, const FieldH& uz) const {
        using V = std::vector<double>;
        auto Ap=[&](const Mat&A,const V&r,const V&i,V&or_,V&oi){ or_=matvec(A,r); oi=matvec(A,i); };
        auto dft=[&](const FieldH&f,std::vector<V>&Fr,std::vector<V>&Fi){ Fr.assign(nz,V(nr,0)); Fi.assign(nz,V(nr,0));
            for(int m=0;m<nr;++m) for(int k=0;k<nz;++k){ double re=0,im=0;
                for(int iz=0;iz<nz;++iz){ double an=-2*M_PI*iz*k/nz; re+=f.at(m,iz)*std::cos(an); im+=f.at(m,iz)*std::sin(an);} Fr[k][m]=re; Fi[k][m]=im; } };
        auto idft=[&](const std::vector<V>&Ur,const std::vector<V>&Ui,FieldH&f){ f=FieldH(nr,nz);
            for(int m=0;m<nr;++m) for(int iz=0;iz<nz;++iz){ double s=0;
                for(int k=0;k<nz;++k){ double an=2*M_PI*iz*k/nz; s+=Ur[k][m]*std::cos(an)-Ui[k][m]*std::sin(an);} f.at(m,iz)=s/nz; } };
        std::vector<V> Xr,Xi,Yr,Yi,Zr,Zi; dft(ux,Xr,Xi); dft(uy,Yr,Yi); dft(uz,Zr,Zi);
        std::vector<V> Uxr(nz,V(nr)),Uxi(nz,V(nr)),Uyr(nz,V(nr)),Uyi(nz,V(nr)),Uzr(nz,V(nr)),Uzi(nz,V(nr));
        for (int k=0;k<nz;++k){ V t1,t2,utxr,utxi,utyr,utyi,utzr,utzi;
            Ap(M2,Xr[k],Xi[k],t1,t2); for(auto&x:t1)x*=mu[k]; for(auto&x:t2)x*=mu[k]; utxr=matvec(Ainv[k],t1); utxi=matvec(Ainv[k],t2);
            Ap(M2,Yr[k],Yi[k],t1,t2); for(auto&x:t1)x*=mu[k]; for(auto&x:t2)x*=mu[k]; utyr=matvec(Ainv[k],t1); utyi=matvec(Ainv[k],t2);
            Ap(M2,Zr[k],Zi[k],t1,t2); for(auto&x:t1)x*=mu[k]; for(auto&x:t2)x*=mu[k]; utzr=matvec(Ainv[k],t1); utzi=matvec(Ainv[k],t2);
            // div_k = mu Qx2 utx + mu Qy2 uty + i*kap M2 utz
            V dr(nr,0),di(nr,0),ax,ay,zr,zi;
            Ap(Qx2,utxr,utxi,ax,ay); for(int i=0;i<nr;++i){dr[i]+=mu[k]*ax[i]; di[i]+=mu[k]*ay[i];}
            Ap(Qy2,utyr,utyi,ax,ay); for(int i=0;i<nr;++i){dr[i]+=mu[k]*ax[i]; di[i]+=mu[k]*ay[i];}
            Ap(M2,utzr,utzi,zr,zi);  for(int i=0;i<nr;++i){dr[i]+= -kap[k]*zi[i]; di[i]+= kap[k]*zr[i];}
            V pr=matvec(Sinv[k],dr), pi=matvec(Sinv[k],di);
            // u = u_tent - A_k^{-1}( G p ); Gx=mu Qx2^T p, Gy=mu Qy2^T p, Gz=-i*kap M2 p
            V gxr,gxi,gyr,gyi,gzr(nr),gzi(nr);
            Ap(Qx2T,pr,pi,gxr,gxi); for(int i=0;i<nr;++i){gxr[i]*=mu[k];gxi[i]*=mu[k];}
            Ap(Qy2T,pr,pi,gyr,gyi); for(int i=0;i<nr;++i){gyr[i]*=mu[k];gyi[i]*=mu[k];}
            Ap(M2,pr,pi,t1,t2);     for(int i=0;i<nr;++i){gzr[i]=kap[k]*t2[i]; gzi[i]=-kap[k]*t1[i];}
            V cxr=matvec(Ainv[k],gxr),cxi=matvec(Ainv[k],gxi),cyr=matvec(Ainv[k],gyr),cyi=matvec(Ainv[k],gyi),czr=matvec(Ainv[k],gzr),czi=matvec(Ainv[k],gzi);
            for(int i=0;i<nr;++i){ Uxr[k][i]=utxr[i]-cxr[i]; Uxi[k][i]=utxi[i]-cxi[i];
                Uyr[k][i]=utyr[i]-cyr[i]; Uyi[k][i]=utyi[i]-cyi[i]; Uzr[k][i]=utzr[i]-czr[i]; Uzi[k][i]=utzi[i]-czi[i]; } }
        FieldH ox,oy,oz; idft(Uxr,Uxi,ox); idft(Uyr,Uyi,oy); idft(Uzr,Uzi,oz);
        return {ox,oy,oz};
    }
};

// One FAST direct-saddle step: explicit SSP-RK3 skew advection (+ forcing, no
// projection), then the direct fast-diag implicit viscous + pressure solve.
inline void hill_fastdiag_step(Hill3D& H, const FastDiagHill& fd, HillNS& s, double dt, double fx = 0.0) {
    auto advect=[&](const HillNS& u){ QField wx=hill_sample(H,u.ux),wy=hill_sample(H,u.uy),wz=hill_sample(H,u.uz);
        const FieldH* U[3]={&u.ux,&u.uy,&u.uz}; HillNS C; FieldH* Co[3]={&C.ux,&C.uy,&C.uz};
        for(int i=0;i<3;++i){ FieldH c=hill_skew_conv(H,wx,wy,wz,*U[i]); *Co[i]=c; for(auto&v:Co[i]->v)v=-v; } return C; };
    auto stage=[&](const HillNS&u,const HillNS&L){ HillNS o=u; for(size_t k=0;k<o.ux.v.size();++k){o.ux.v[k]+=dt*L.ux.v[k];o.uy.v[k]+=dt*L.uy.v[k];o.uz.v[k]+=dt*L.uz.v[k];} return o; };
    auto comb=[&](double c1,const HillNS&A,double c2,const HillNS&B){ HillNS o=A; for(size_t k=0;k<o.ux.v.size();++k){o.ux.v[k]=c1*A.ux.v[k]+c2*B.ux.v[k];o.uy.v[k]=c1*A.uy.v[k]+c2*B.uy.v[k];o.uz.v[k]=c1*A.uz.v[k]+c2*B.uz.v[k];} return o; };
    HillNS u1=stage(s,advect(s));
    HillNS u2=comb(0.75,s,0.25,stage(u1,advect(u1)));
    HillNS us=comb(1.0/3.0,s,2.0/3.0,stage(u2,advect(u2)));
    if(fx!=0.0) for(double& v:us.ux.v) v+=dt*fx;
    auto out=fd.solve(us.ux,us.uy,us.uz);
    s.ux=out[0]; s.uy=out[1]; s.uz=out[2];
}

#endif // HILL_FASTDIAG_H
