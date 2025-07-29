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

#ifndef _qpi_h
#define _qpi_h

#include "mpidefs.h"

#include "tightbinding.h"
#include "wannierfunctions.h"
#include <vector>
#include <complex>
#include <fftw3.h>
#include <omp.h>

#include "idl.h"

#ifdef _GPU
#include "gpuqpi.h"
#endif

struct flaglistentry {
  size_t i,j,o1,o2;
  double factor;
};

typedef vector<vector<vector<vector<vector<vector<vector<vector<bool> > > > > > > > flagarray;
typedef vector<vector<vector<vector<vector<flaglistentry> > > > > flaglist;

size_t mkthresholdmap(flagarray &flags, vector<wannierfunctions> &wf,size_t window,size_t oversamp,double threshold=0.01,size_t maxband=0,bool spin=false);
size_t mkthresholdlist(flaglist &flags, vector<wannierfunctions> &wf,size_t window,size_t oversamp,double threshold=0.01,size_t maxband=0,bool spin=false);
size_t mkthresholdlist(flaglist &flags, vector<wannierfunctions> &wf,size_t window,size_t oversamp,double threshold,vector<int> &spinarr);

//do t-matrix calculation for n bands on an mxm matrix
class tmatrix {
  fftw_plan myplan;
  tightbinding *tb;
  size_t n,bands,fftsize,kpoints;
  double eta,limit;
  bool spin;
  complex<double> *fftdata;
  vector<vector<gsl_matrix_complex*> > g0;
  vector<vector<gsl_matrix_complex*> > evect;
  vector<vector<gsl_vector*> > eval;
  vector<vector<double> > ldos;
  vector<complex<double> > vscat;
  gsl_matrix_complex *scat;
  //variables for continuum spectral function
  complex<double> *gfftdata;
  fftw_plan mygplan;
  double gnormconst;
  size_t gfftsize;
  vector<vector<gsl_complex> > contg;
#ifdef _GPU
  GPUQPI *gpuqpi;
#endif
  void alloc();
  void alloc_hamiltonian();
  void allocldos(size_t wanniern);
#ifdef _GPU
  void allocgpumem(size_t wanniern,size_t window,size_t oversamp,size_t maxband, vector<wannierfunctions> &wf);
  void allocgpumem(size_t wanniern,size_t window,size_t oversamp,flaglist &flags);
  void allocgpumem(size_t maxbands,vector<vector<double> > &pos,vector<double> &prearr);
  void freegpumem();
#endif
  void freeg0() {
#pragma omp parallel for
    for(size_t i=0;i<g0.size();i++)
      for(size_t j=0;j<g0[i].size();j++)
	if(g0[i][j]) gsl_matrix_complex_free(g0[i][j]);
#pragma omp parallel for
    for(size_t i=0;i<evect.size();i++)
      for(size_t j=0;j<evect[i].size();j++)
	if(evect[i][j]) gsl_matrix_complex_free(evect[i][j]);
#pragma omp parallel for
    for(size_t i=0;i<eval.size();i++)
      for(size_t j=0;j<eval[i].size();j++)
	if(eval[i][j]) gsl_vector_free(eval[i][j]);
  }
  void allocgfft(size_t wanniern);
  void alloccontg(size_t wanniern);
  void fftalloc();
  double calcwannierldospoint_int(size_t xp,size_t yp,size_t oversamp,size_t window,vector<wannierfunctions> &wf,size_t maxband=0);
public:
  tmatrix(tightbinding *itb,size_t in,size_t ikpoints=0,double ieta=0.00001,bool spin=false) :tb(itb),n(in),kpoints(ikpoints),eta(ieta),limit(0.5),spin(spin) {
    if(!kpoints) kpoints=n;
    bands=tb->getbands(); scat=NULL; vscat.assign(bands,polar(1.0,0.0)); fftdata=NULL; gfftdata=NULL;
#ifdef _GPU
    gpuqpi=NULL;
#endif
    fftalloc();
    fftw_init_threads();
    fftw_plan_with_nthreads(omp_get_max_threads());
    myplan=fftw_plan_dft_2d(kpoints,kpoints,reinterpret_cast<fftw_complex*>(fftdata),reinterpret_cast<fftw_complex*>(fftdata),FFTW_BACKWARD,0);
  }
  void setscatteringphase(complex<double> phase) {
    vscat.assign(bands,phase);
  }
  void setscatteringphase(double ivscat,double iphi) {
    vscat.assign(bands,polar(ivscat,iphi));
  }
  void setscatteringphase(double ivscat,double iphi, vector<size_t> &scat) {
    for(size_t i=0;i<bands;i++) {
      if(i<scat.size()) {
	if(scat[i])
	  vscat[i]=polar(ivscat,iphi);
	 else
	   vscat[i]=complex<double>(0.0,0.0);
      } else
	vscat[i]=complex<double>(0.0,0.0);
    }
  }
  void setscatteringphase(vector<complex<double> > &scat) {
    for(size_t i=0;i<bands;i++)
      if(i<scat.size())
	vscat[i]=scat[i];
      else
	vscat[i]=complex<double>(0.0,0.0);
  }
  void printscattering(ostream &os);
  void setbroadening(double neta) { eta=neta; }
  void setlimit(double nlimit) { limit=nlimit; }
  void calchamiltonian();
  void setgreensfunction(double omega);
  void calcsurfacegreensfunction(double omega,double epserr=1.0e-5,bool surface=true,tightbind *smodel=NULL);
  //calculates G0(0,r,omega)
  void calcrealspacegreensfunction(); 
  void setscatteringmatrix();
  void writeidl(const char *name);
  void writeldos(const char *name);
  void ldos2idl(idl &map,size_t layer);
  void spf2idl(idl &map, size_t layer,size_t maxband=0,bool shift=false);
  void spf2array(double *map,size_t maxband=0,bool shift=false);
  void spf2idl(vector<vector<double> > &pos,vector<double> &prearr,idl &map, size_t layer,size_t maxband=0);
  void spf2array(vector<vector<double> > &pos,vector<double> &prearr,double *map,size_t maxband=0);
  void calcspf(size_t maxband,bool shift=true);
  void calcuspf(size_t maxband,vector<vector<double> > &pos,vector<double> &prearr);
  void ldos2vector(vector<vector <double> > &map);
  void ldos2array(double *map);
  double calcdos(size_t maxband=0);
  void calcwannierldos(size_t oversamp,size_t window,vector<wannierfunctions> &wf,size_t maxband=0);
  void calcwannierldos(size_t oversamp,size_t window,vector<wannierfunctions> &wf,flagarray &flags);
  void calcwannierldos(size_t oversamp,size_t window,flaglist &flags);
  void calcwannierjosephson(size_t oversamp,size_t window,vector<wannierfunctions> &wf,gsl_complex tip);
  double calcwannierldospoint(size_t xp,size_t yp,size_t oversamp,size_t window,vector<wannierfunctions> &wf,size_t maxband=0);
  ~tmatrix() {
    if(fftdata) {
      delete fftdata;
      fftw_destroy_plan(myplan);
    }
    if(gfftdata) {
      delete gfftdata;
      fftw_destroy_plan(mygplan);
    }
    freeg0();
#ifdef _GPU
    if(gpuqpi) freegpumem();
#endif
    if(scat) gsl_matrix_complex_free(scat);
    fftw_cleanup_threads();
  }
};

#endif
