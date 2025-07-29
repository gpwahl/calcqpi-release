/*
This file is part of calcQPI.
calcQPI is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published
by the Free Software Foundation, either version 3 of the License,
or (at your option) any later version.
calcQPI is distributed in the hope that it will be useful, but
WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
See the GNU General Public License for more details.
You should have received a copy of the GNU General Public License along with calcQPI.
If not, see <https://www.gnu.org/licenses/>.
*/

#ifndef _tightbinding_h
#define _tightbinding_h

#include <iostream>
#include <fstream>

#include <gsl/gsl_math.h>
#include <gsl/gsl_complex_math.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_eigen.h>
#include <gsl/gsl_blas.h>

#include <gsl/gsl_permutation.h>
#include <gsl/gsl_linalg.h>

#include <vector>
#include <string>
#include <complex>

using namespace std;

enum kzdep {kzdepall,kzdepinplane,kzdepoutofplane};


extern const double kb;
double fermi(double x,double temperature);

class lorentzian {
  double w;
public:
  lorentzian(double iw):w(iw) {}
  double width() { return w; }
  double operator()(double x) {return w/2.0/(x*x+w*w/4.0)/M_PI; }
  double normheight(double x) {return w*w/4.0/(x*x+w*w/4.0); }
};

class tightbindingcontext {
public:
  gsl_matrix_complex *hopping,*evec;
  gsl_vector *eval;
  gsl_eigen_hermv_workspace *w;
  gsl_permutation *p;
  tightbindingcontext(size_t n) {
    hopping=gsl_matrix_complex_alloc(n,n); evec=gsl_matrix_complex_alloc(n,n);
    eval=gsl_vector_alloc(n);
    w=gsl_eigen_hermv_alloc(n);
    p=gsl_permutation_alloc(n);
  }
  void solveeigensystem() {
    gsl_eigen_hermv(hopping, eval, evec, w);
  }
  void calcgreensfunction(gsl_matrix_complex *g0,double omega, double eta) {
    size_t n=hopping->size1;
    gsl_matrix_complex_set_zero(g0);
    for(size_t i=0;i<n;i++) {
      gsl_complex factor=gsl_complex_inverse(gsl_complex_rect(omega-gsl_vector_get(eval,i),eta));
      gsl_vector_complex_view evect=gsl_matrix_complex_column(evec,i);
      gsl_blas_zgerc(factor,(gsl_vector_complex *)&evect,(gsl_vector_complex *)&evect,g0);
    } 
  }
  void calcgreensfunctioninv(gsl_matrix_complex *g0,double omega, double eta) {  
    int signum;
    gsl_matrix_complex_set_identity(g0);
    //(H(k)-omega-i*eta)
    gsl_matrix_complex_scale(g0,gsl_complex_rect(omega,eta));
    gsl_matrix_complex_sub(hopping,g0);
    gsl_matrix_complex_scale(hopping,GSL_COMPLEX_NEGONE);
    //calculate inverse
    gsl_linalg_complex_LU_decomp(hopping,p,&signum);
    gsl_linalg_complex_LU_invert(hopping,p,g0);   
  }
  ~tightbindingcontext() {
    gsl_matrix_complex_free(hopping);
    gsl_vector_free(eval);
    gsl_matrix_complex_free(evec);
    gsl_eigen_hermv_free(w);
    gsl_permutation_free(p);
  }
};

class tightbinding {
protected:
  bool error;
public:
  virtual tightbindingcontext *getnewcontext()=0;
  virtual size_t getbands()=0;
  virtual void setmatrix(vector<double> &k,gsl_matrix_complex *lhopping,kzdep kzdependence)=0;
  virtual void setmatrix(vector<double> &k,gsl_matrix_complex *lhopping)=0;
  virtual void setmatrix(vector<double> &k)=0;
  virtual void calcgreensfunction(gsl_matrix_complex *g0,vector<double> &k,double omega, double eta,tightbindingcontext *tbc)=0;
  virtual void calcgreensfunction(gsl_matrix_complex *g0,vector<double> &k,double omega, double eta)=0;
  virtual void calcgreensfunction(gsl_matrix_complex *g0,gsl_matrix_complex *evec,gsl_vector *eval,double omega, double eta)=0;
  virtual ~tightbinding() {}
  friend bool operator!(tightbinding &tb);
};

class hoppingterm {
public:
  double rx,ry,rz;
  complex<double> t;
  size_t o1,o2;
  bool anti;
  hoppingterm(double rx,double ry, double rz,size_t o1,size_t o2,complex<double> t,bool anti=false):rx(rx),ry(ry),rz(rz),t(t),o1(o1),o2(o2),anti(anti) {}
  hoppingterm(double rx,double ry, double rz,size_t o1,size_t o2,double tr,double ti,bool anti=false):rx(rx),ry(ry),rz(rz),t(complex<double>(tr,ti)),o1(o1),o2(o2),anti(anti) {}
  hoppingterm(double rx,double ry, double rz,size_t o1,size_t o2,gsl_complex t,bool anti=false):rx(rx),ry(ry),rz(rz),t(GSL_REAL(t),GSL_IMAG(t)),o1(o1),o2(o2),anti(anti) {}
  void addhop(gsl_matrix_complex *m,double kx,double ky,double kz) {
    complex<double> val;
    if(anti)
      val=t*exp(complex<double>(0.0,-2.0*M_PI*(rx*kx+ry*ky+rz*kz)));
    else val=t*exp(complex<double>(0.0,2.0*M_PI*(rx*kx+ry*ky+rz*kz)));
    gsl_complex v=gsl_complex_add(gsl_matrix_complex_get(m,o1,o2),gsl_complex_rect(val.real(),val.imag()));
    gsl_matrix_complex_set(m,o1,o2,v);
  }
  void addhop(gsl_matrix_complex *m,vector<double> &k) {
    complex<double> val;
    if(anti)
      val=t*exp(complex<double>(0.0,-2.0*M_PI*(rx*k[0]+ry*k[1]+rz*k[2])));
    else val=t*exp(complex<double>(0.0,2.0*M_PI*(rx*k[0]+ry*k[1]+rz*k[2])));
    gsl_complex v=gsl_complex_add(gsl_matrix_complex_get(m,o1,o2),gsl_complex_rect(val.real(),val.imag()));
    gsl_matrix_complex_set(m,o1,o2,v);
  }
  void addhop(size_t ofs1,size_t ofs2,gsl_matrix_complex *m,vector<double> &k) {
    complex<double> val;
    if(anti) val=t*exp(complex<double>(0.0,-2.0*M_PI*(rx*k[0]+ry*k[1]+rz*k[2])));
    else val=t*exp(complex<double>(0.0,2.0*M_PI*(rx*k[0]+ry*k[1]+rz*k[2])));
    gsl_complex v=gsl_complex_add(gsl_matrix_complex_get(m,ofs1+o1,ofs2+o2),gsl_complex_rect(val.real(),val.imag()));
    gsl_matrix_complex_set(m,ofs1+o1,ofs2+o2,v);
  }
  void subhop(size_t ofs1,size_t ofs2,gsl_matrix_complex *m,vector<double> &k) {
    complex<double> val;
    if(anti) val=t*exp(complex<double>(0.0,-2.0*M_PI*(rx*k[0]+ry*k[1]+rz*k[2])));
    else val=t*exp(complex<double>(0.0,2.0*M_PI*(rx*k[0]+ry*k[1]+rz*k[2])));
    gsl_complex v=gsl_complex_sub(gsl_matrix_complex_get(m,ofs1+o1,ofs2+o2),gsl_complex_rect(val.real(),val.imag()));
    gsl_matrix_complex_set(m,ofs1+o1,ofs2+o2,v);
  }
  void addabshop(gsl_matrix_complex *m) {
    complex<double> val=t*conj(t);
    gsl_complex v=gsl_complex_add(gsl_matrix_complex_get(m,o1,o2),gsl_complex_rect(val.real(),val.imag()));
    gsl_matrix_complex_set(m,o1,o2,v);
  }
  void addhopd(gsl_matrix_complex *m,vector<double> &k,size_t dim) {
    complex<double> val;
    double r[3]={rx,ry,rz};
    if(anti)
      val=-2.0*M_PI*t*complex<double>(0.0,r[dim])*exp(complex<double>(0.0,-2.0*M_PI*(rx*k[0]+ry*k[1]+rz*k[2])));
    else val=2.0*M_PI*t*complex<double>(0.0,r[dim])*exp(complex<double>(0.0,2.0*M_PI*(rx*k[0]+ry*k[1]+rz*k[2])));
    gsl_complex v=gsl_complex_add(gsl_matrix_complex_get(m,o1,o2),gsl_complex_rect(val.real(),val.imag()));
    gsl_matrix_complex_set(m,o1,o2,v);
  }
};

class tightbind:public tightbinding {
protected:
  bool verbose;
  size_t n,neighbourno;
  string date;
  double efermi;
  bool scmodel;
  size_t scmaxbands;
  vector<size_t> degeneracies;
  tightbindingcontext *ctbc;
  vector<hoppingterm> hoppings;
  gsl_complex overlap(size_t nn, size_t i, size_t j);
  void readfile(istream &is);
public:
  tightbind() { error=false; scmodel=false; }
  tightbind(const string &filename, double ifermi=0.0, bool scmodel=false);
  tightbind(istream &is, double ifermi=0.0, bool scmodel=false);
  virtual size_t getbands() {return n;}
  string getname() {return date;}
  virtual tightbindingcontext *getnewcontext() {return new tightbindingcontext(n);} 
  virtual void setmatrix(vector<double> &k,gsl_matrix_complex *lhopping,kzdep kzdependence);
  virtual void setmatrix(vector<double> &k,gsl_matrix_complex *lhopping);
  virtual void setmatrix(vector<double> &k);
  virtual void setmatrixd(vector<double> &k,gsl_matrix_complex *lhopping,size_t dim);
  void inthoppings();
  void debugout(ostream &o);
  void showresults(ostream &o,tightbindingcontext *tbc);
  void showresults(ostream &o);
  void calcenergies(vector<double> &k,tightbindingcontext *tbc,bool sort=false);
  void calcenergies(vector<double> &k,bool sort=false);
  void calckpointlist(vector<vector<double> > &kvects, vector<double> &energies,vector<vector<double> > &oweights,vector<vector<double> > &inkvects,bool sort=false);
  void calckpointlist(vector<vector<double> > &kvects, vector<double> &energies,vector<vector<gsl_complex> > &oweights,vector<vector<double> > &inkvects,bool sort=false);
  void calckpointlist(vector<vector<double> > &kvects, vector<double> &delta,vector<double> &berryphase,double sign=1.0);
  void calckpointlistop(vector<vector<double> > &kvects, gsl_matrix_complex *op,vector<complex<double> > &result,bool spin=true,double sign=1.0);
  void calckpointlistop(vector<vector<double> > &kvects, vector<vector<complex<double> > > &result,gsl_matrix_complex **op,double temperature,size_t ops);
  void calcdir(vector<vector<double> > &kvects, vector<double> &energies,vector<vector<double> > &oweights,vector<double> &k1,vector<double> &dir,int n,size_t maxbands=0);
  void calcline(vector<vector<double> > &kvects, vector<double> &energies,vector<vector<double> > &oweights,vector<double> &k1,vector<double> &k2,int n,size_t maxbands=0);
  void calcdir(vector<vector<double> > &kvects, vector<double> &energies,vector<vector<complex<double> > > &oweights,vector<double> &k1,vector<double> &dir,int n,size_t maxbands=0);
  void calcline(vector<vector<double> > &kvects, vector<double> &energies,vector<vector<complex<double> > > &oweights,vector<double> &k1,vector<double> &k2,int n,size_t maxbands=0);
  void calc2dbandstructure(ostream &os, double kz,size_t n);
  void calcdos2d(vector<vector<double> > &hist,double a,double b,size_t bins,bool orb_res=false,double kstep=0.001);
  void calcdos3d(vector<vector<double> > &hist,double a,double b,size_t bins,bool orb_res=false,double kstep=0.1);
  void projop2d(vector<complex<double> > &res,gsl_matrix_complex **op,double temperature,double kstep=0.001,size_t ops=1);
  void occupations2d(vector<vector<double> > &hist,double a,double b,size_t bins,bool orb_res=false,double kstep=0.001);
  void occupations3d(vector<vector<double> > &hist,double a,double b,size_t bins,bool orb_res=false,double kstep=0.1);
  double getoccupation2d(double fermi=0.0,double kstep=0.001);
  double getoccupation3d(double fermi=0.0,double kstep=0.1);
  void calcbst3dhisthr(const char *fname,size_t pix,double le,double ue,size_t layers,size_t oversamp=2,double ux=1.0,double uy=1.0,size_t maxbands=0);
  void calcbst3dhisthrsel(const char *fname,size_t pix,double le,double ue,size_t layers,size_t oversamp,vector<size_t> &orbs,double ux=1.0,double uy=1.0);
  void calcbst3dhisthr(const char *fname,size_t pix,size_t kint,double le,double ue,size_t layers,size_t oversamp=2,size_t maxbands=0);
  void calcbstcechisthr(const char *fname,double e,size_t pix,size_t zpix,size_t oversamp=2,double width=0.01);
  void calcjdoshr(const char *fname,size_t pix,double le,double ue,size_t layers,size_t oversamp=2);
  double getmaxhopping();
  bool checkprincipallayer() {
    bool principal=true;
    for(size_t l=0;l<hoppings.size();l++)
      if(abs(hoppings[l].rz)>1)
	if(((hoppings[l].t.real())!=0.0)||((hoppings[l].t.imag())!=0.0))
	  principal=false;
    return principal;
  }
  virtual void calcgreensfunction(gsl_matrix_complex *g0,vector<double> &k,double omega, double eta,tightbindingcontext *tbc);
  virtual void calcgreensfunction(gsl_matrix_complex *g0,vector<double> &k,double omega, double eta);
  virtual void calcgreensfunction(gsl_matrix_complex *g0,gsl_matrix_complex *evec,gsl_vector *eval,double omega, double eta);
  double operator()(size_t nb) {return gsl_vector_get(ctbc->eval,nb);}
  complex<double>  get_eigenvect(size_t nb,size_t l) { gsl_complex a=gsl_matrix_complex_get(ctbc->evec,l,nb);
    return complex<double>(GSL_REAL(a),GSL_IMAG(a));
  }
  bool inbz(double kx,double ky) { if((abs(kx)<0.5) && (abs(ky)<0.5)) return true; else return false; }
  void writebs(const char *name,size_t pts,bool sort=false);
  void writebs3d(const char *name,size_t pts,size_t zpts,bool sort=false);
  virtual ~tightbind() {
    if(ctbc) delete ctbc;
  }
  size_t findindex(int x, int y, int z,size_t o1,size_t o2);
  gsl_complex gethopping(int x,int y,int z,size_t i,size_t j) { size_t k=findindex(x,y,z,i,j); if(k<hoppings.size()) return gsl_complex_rect(hoppings[k].t.real(),hoppings[k].t.imag()); else return gsl_complex_rect(0.0,0.0); }
};

#endif
