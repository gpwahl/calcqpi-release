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

#ifndef _qpicudakernel_h
#define _qpicudakernel_h

#include <vector>
#include <iostream>
#include <gsl/gsl_complex_math.h>
#include <gsl/gsl_matrix.h>

#include "qpi.h"
#include "wannierfunctions.h"
#include "idl.h"
#include "gpuqpi.h"

using namespace std;

typedef double gpufloat;
typedef uint gpuuint;
typedef int gpuint;

struct dblcomplex {
  gpufloat real, imag;
};

struct qpigpuinfo {
  gpuuint wanniern,kpoints,n,bands,oversamp,window,maxband;
};

struct ctspecgpuinfo {
  gpuuint wanniern,kpoints,n,bands,window,oversamp,maxband;
  gpufloat xorig,yorig;
};

struct qpigpuflaglist {
  gpuuint i,j,o1,o2;
  gpufloat factor;
};

//indexing for 2D arrays
#define IDX2C(i,j,ld) (((j)*(ld))+(i))
//indexing for 4D arrays (Green's function)
#define IDX4C(i,j,k,l,ld0,ld1) ((((j)*(ld0))+(i))*(ld1)*(ld1)+(((l)*(ld1))+(k)))
#define IDX4CC(i,j,k,l,ld) ((((((i)*(ld))+(j))*(ld)+(k))*(ld))+(l))
//indexing for 3D array
#define IDX3C(i,j,k,ld0) (((i)*((ld0)*(ld0)))+(((j)*(ld0))+(k)))

class CudaQPI:public GPUQPI
{
  dblcomplex *gpug0mem,*d_gpug0mem,*gpuscat,*d_gpuscat,*d_gpubuf,*d_gpucldos,*d_gpucontg;
  gpufloat *gpuldosmem,*d_gpuldosmem,*d_gpuwf,*d_mpos;
  gpuuint *d_flagentries,*d_flagofs;
  qpigpuflaglist *d_flaglist;
  size_t scatsize,g0size,ldossize,gpumem,cpumem;
  size_t wanniern,kpoints,nkpts,n,window,oversamp,bands,maxband;
  bool spin;
  //GPU parameters for LDOS calculation
  size_t ldosblocks,ldosblocksize;
  //GPU parameters for clearing LDOS
  //size_t cleartotal,clearblocks,clearblocksize;
  void copygf(vector<vector<gsl_matrix_complex *> > &g0);
 public:
  CudaQPI(size_t wanniern,size_t kpoints,size_t n,size_t window,size_t oversamp,size_t bands,size_t maxband,bool spin,vector<wannierfunctions> &wf);
  CudaQPI(size_t wanniern, size_t kpoints,size_t n,size_t window, size_t oversamp,size_t bands,flaglist &flist);
  CudaQPI(size_t kpoints,size_t n,size_t bands,size_t maxbands,bool spin,vector<vector<double> > &pos,vector<double> &prearr);
  void printinfo(ostream &os);
  void wannierldos(gsl_matrix_complex *scat, vector<vector<gsl_matrix_complex *> > &g0);
  void wannierjosephson(gsl_matrix_complex *scat, vector<vector<gsl_matrix_complex *> > &g0,gsl_complex tip);
  void wannierldoslist(gsl_matrix_complex *scat, vector<vector<gsl_matrix_complex *> > &g0);
  void spf(vector<vector<gsl_matrix_complex *> > &g0);
  void uspf(vector<vector<gsl_matrix_complex *> > &g0);
  void retrieveResult(vector<vector<double> > &ldos);
  void retrieveResult(idl &map,size_t layer);
  void retrieveResult(double *ldos);
  ~CudaQPI();
};

#endif
