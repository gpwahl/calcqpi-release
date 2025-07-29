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

#include "mpidefs.h"

#include <cuda_runtime_api.h>
#include <cuda.h>
#include <math_constants.h>

#ifdef _mpi_version
#include <mpi.h>
#endif

#include "qpicudakernel.h"
#include <iomanip>

#include "idl.h"

using namespace std;

__device__ __inline__ dblcomplex operator+(dblcomplex a,dblcomplex b) { return {a.real+b.real,a.imag+b.imag}; }
__device__ __inline__ dblcomplex operator*(dblcomplex a,dblcomplex b) { return {a.real*b.real-a.imag*b.imag,a.real*b.imag+a.imag*b.real}; }

__device__ __inline__ dblcomplex operator*(const gpufloat a,const dblcomplex b) { return (dblcomplex){a*b.real,a*b.imag}; }

__device__ __inline__ dblcomplex cmpexp(const dblcomplex x) { gpufloat prod=exp(x.real); return (dblcomplex){prod*cos(x.imag),prod*sin(x.imag)}; }

__device__ __inline__ void matrixmultiplication(const dblcomplex *a, const dblcomplex *b, dblcomplex *c, const gpuuint n)
{
  for(gpuuint i=0;i<n*n;i++)
    *(c+i)={0.0f,0.0f};
  for(gpuuint i=0;i<n;i++)
    for(gpuuint j=0;j<n;j++)
      for(gpuuint k=0;k<n;k++)
	*(c+IDX2C(i,j,n))=*(c+IDX2C(i,j,n))+(*(a+IDX2C(i,k,n))*(*(b+IDX2C(k,j,n))));
}

//calculate LDOS using Wannier functions - using cached functions
//g0 - two-dimensional array of complex matrices (kpointsxkpointsxbandsxbands)
//ldos - two dimensioanl array of double values (nxn)
//scat - matrix (bandsxbands)
//wf - array of two-dimensional arrays containing orbitals (bandsx((2*windows+1)*oversamp)x((2*windows+1)*oversamp))
//kpoints - number of kpoints, as specified in g0
//n - number of lattice sites in real-space lattice
//bands - number of bands
//oversamp - oversampling of real-space lattice
//window - window used for orbitals
//maxband - band up to which the orbitals are included (e.g. for superconducting calculation)
__global__ void gpucalcwannierldos(const dblcomplex *g0, gpufloat *ldos,const dblcomplex *scat,const gpufloat *wf,dblcomplex *globalbuffer,dblcomplex *globalcldos,gpuuint kpoints,gpuuint n,gpuuint bands,gpuuint oversamp,gpuuint window,gpuuint maxband) {
  gpuuint wanniern=oversamp*n;
  gpuuint n2=(n>>1);
  gpuuint wfsize=(2*window+1)*oversamp;
  gpuuint globalindex=blockIdx.x * blockDim.x + threadIdx.x;
  gpuuint i=globalindex%n,j=globalindex/n;
  if(j>=n) return;
  //sum over lattice
  dblcomplex *buffer=globalbuffer+bands*bands*globalindex,
    *cldos=globalcldos+bands*bands*globalindex;
  gpuint ipos=(gpuint)i-n2, jpos=(gpuint)j-n2;
  for(gpuuint ii=0;ii<oversamp;ii++)
    for(gpuuint jj=0;jj<oversamp;jj++)
      *(ldos+IDX2C((i*oversamp+ii)%wanniern,(j*oversamp+jj)%wanniern,wanniern))=0.0f;
  //R-loop over nearest neighbours
  for(gpuint nni1=-window;nni1<=(gpuint)window;nni1++)
    for(gpuint nnj1=-window;nnj1<=(gpuint)window;nnj1++) {
      //calculate TG0(0,R)
      matrixmultiplication(scat,g0+IDX4C((kpoints+kpoints-(ipos+nni1))%kpoints,(kpoints+kpoints-(jpos+nnj1))%kpoints,0,0,kpoints,bands),buffer,bands);
      //R'-loop over nearest neighbours
      for(gpuint nni2=-window;nni2<=(gpuint)window;nni2++)
	for(gpuint nnj2=-window;nnj2<=(gpuint)window;nnj2++) {
	  //calculate G0(R',0)TG0(0,R)    
	    matrixmultiplication(g0+IDX4C((kpoints+ipos+nni2)%kpoints,(kpoints+jpos+nnj2)%kpoints,0,0,kpoints,bands),buffer,cldos,bands);	      
	  //sum over sub-unit cell sampling
	  //gpuuint x=ipos*oversamp+ii,y=jpos*oversamp+jj;
	  for(gpuuint ii=0;ii<oversamp;ii++)
	    for(gpuuint jj=0;jj<oversamp;jj++) {
	      gpuuint x=ii+window*oversamp,y=jj+window*oversamp;
	      gpufloat val=*(ldos+IDX2C((i*oversamp+ii)%wanniern,(j*oversamp+jj)%wanniern,wanniern));
	      for(gpuuint o1=0;o1<maxband;o1++)
		for(gpuuint o2=0;o2<maxband;o2++) {
		  gpufloat factor=*(wf+IDX3C(o1,x-nni1*oversamp,y-nnj1*oversamp,wfsize))*(*(wf+IDX3C(o2,x-nni2*oversamp,y-nnj2*oversamp,wfsize)));
		  val-=(cldos+IDX2C(o2,o1,bands))->imag*factor;
		  val-=(g0+IDX4C((kpoints-nni1+nni2)%kpoints,(kpoints-nnj1+nnj2)%kpoints,o2,o1,kpoints,bands))->imag*factor;
		}
	      *(ldos+IDX2C((i*oversamp+ii)%wanniern,(j*oversamp+jj)%wanniern,wanniern))=val;
	    }
	}
    }
}

//calculate LDOS using Wannier functions - using cached functions
//g0 - two-dimensional array of complex matrices (kpointsxkpointsxbandsxbands)
//ldos - two dimensioanl array of double values (nxn)
//scat - matrix (bandsxbands)
//wf - array of two-dimensional arrays containing orbitals (bandsx((2*windows+1)*oversamp)x((2*windows+1)*oversamp))
//kpoints - number of kpoints, as specified in g0
//n - number of lattice sites in real-space lattice
//bands - number of bands
//oversamp - oversampling of real-space lattice
//window - window used for orbitals
//maxband - band up to which the orbitals are included (e.g. for superconducting calculation)
//spin - spin-polarized calculation
__global__ void gpucalcwannierldosspin(const dblcomplex *g0, gpufloat *ldos,const dblcomplex *scat,const gpufloat *wf,dblcomplex *globalbuffer,dblcomplex *globalcldos,gpuuint kpoints,gpuuint n,gpuuint bands,gpuuint oversamp,gpuuint window,gpuuint maxband) {
  gpuuint wanniern=oversamp*n;
  gpuuint n2=(n>>1);
  gpuuint wfsize=(2*window+1)*oversamp;
  gpuuint spinbands=maxband>>1;
  gpuuint globalindex=blockIdx.x * blockDim.x + threadIdx.x;
  gpuuint i=globalindex%n,j=globalindex/n;
  if(j>=n) return;
  //sum over lattice
  dblcomplex *buffer=globalbuffer+bands*bands*globalindex,
    *cldos=globalcldos+bands*bands*globalindex;
  gpuint ipos=(gpuint)i-n2, jpos=(gpuint)j-n2;
  for(gpuuint ii=0;ii<oversamp;ii++)
    for(gpuuint jj=0;jj<oversamp;jj++)
      *(ldos+IDX2C((i*oversamp+ii)%wanniern,(j*oversamp+jj)%wanniern,wanniern))=0.0f;
  //R-loop over nearest neighbours
  for(gpuint nni1=-window;nni1<=(gpuint)window;nni1++)
    for(gpuint nnj1=-window;nnj1<=(gpuint)window;nnj1++) {
      //calculate TG0(0,R)
      matrixmultiplication(scat,g0+IDX4C((kpoints+kpoints-(ipos+nni1))%kpoints,(kpoints+kpoints-(jpos+nnj1))%kpoints,0,0,kpoints,bands),buffer,bands);
      //R'-loop over nearest neighbours
      for(gpuint nni2=-window;nni2<=(gpuint)window;nni2++)
	for(gpuint nnj2=-window;nnj2<=(gpuint)window;nnj2++) {
	  //calculate G0(R',0)TG0(0,R)    
	    matrixmultiplication(g0+IDX4C((kpoints+ipos+nni2)%kpoints,(kpoints+jpos+nnj2)%kpoints,0,0,kpoints,bands),buffer,cldos,bands);
	  //gpuuint x=ipos*oversamp+ii,y=jpos*oversamp+jj;
	  //sum over sub-unit cell sampling
	  for(gpuuint ii=0;ii<oversamp;ii++)
	    for(gpuuint jj=0;jj<oversamp;jj++) {
	      gpuuint x=ii+window*oversamp,y=jj+window*oversamp;
	      gpufloat val=*(ldos+IDX2C((i*oversamp+ii)%wanniern,(j*oversamp+jj)%wanniern,wanniern));
	      for(gpuuint o1=0;o1<spinbands;o1++)
		for(gpuuint o2=0;o2<spinbands;o2++) {
		  gpufloat factor=*(wf+IDX3C(o1,x-nni1*oversamp,y-nnj1*oversamp,wfsize))*(*(wf+IDX3C(o2,x-nni2*oversamp,y-nnj2*oversamp,wfsize)));
		  val-=(cldos+IDX2C(o2,o1,bands))->imag*factor;
		  val-=(g0+IDX4C((kpoints-nni1+nni2)%kpoints,(kpoints-nnj1+nnj2)%kpoints,o2,o1,kpoints,bands))->imag*factor;
		  factor=*(wf+IDX3C(spinbands+o1,x-nni1*oversamp,y-nnj1*oversamp,wfsize))*(*(wf+IDX3C(spinbands+o2,x-nni2*oversamp,y-nnj2*oversamp,wfsize)));
		  val-=(cldos+IDX2C(spinbands+o2,spinbands+o1,bands))->imag*factor;
		  val-=(g0+IDX4C((kpoints-nni1+nni2)%kpoints,(kpoints-nnj1+nnj2)%kpoints,spinbands+o2,spinbands+o1,kpoints,bands))->imag*factor;
		}
	      *(ldos+IDX2C((i*oversamp+ii)%wanniern,(j*oversamp+jj)%wanniern,wanniern))=val;
	    }
	}
    }
}

//calculate LDOS using Wannier functions - using cached functions
//g0 - two-dimensional array of complex matrices (kpointsxkpointsxbandsxbands)
//ldos - two dimensioanl array of double values (nxn)
//scat - matrix (bandsxbands)
//wf - array of two-dimensional arrays containing orbitals (bandsx((2*windows+1)*oversamp)x((2*windows+1)*oversamp))
//kpoints - number of kpoints, as specified in g0
//n - number of lattice sites in real-space lattice
//bands - number of bands
//oversamp - oversampling of real-space lattice
//window - window used for orbitals
//maxband - band up to which the orbitals are included (e.g. for superconducting calculation)
__global__ void gpucalcwannierjosephson(const dblcomplex *g0, gpufloat *ldos,const dblcomplex *scat,const gpufloat *wf,dblcomplex *globalbuffer,dblcomplex *globalcldos,const dblcomplex tip,gpuuint kpoints,gpuuint n,gpuuint bands,gpuuint oversamp,gpuuint window,gpuuint maxband) {
  gpuuint wanniern=oversamp*n;
  gpuuint n2=(n>>1);
  gpuuint wfsize=(2*window+1)*oversamp;
  gpuuint globalindex=blockIdx.x * blockDim.x + threadIdx.x;
  gpuuint i=globalindex%n,j=globalindex/n;
  if(j>=n) return;
  //sum over lattice
  dblcomplex *buffer=globalbuffer+bands*bands*globalindex,
    *cldos=globalcldos+bands*bands*globalindex;
  gpuint ipos=(gpuint)i-n2, jpos=(gpuint)j-n2;
  for(gpuuint ii=0;ii<oversamp;ii++)
    for(gpuuint jj=0;jj<oversamp;jj++)
      *(ldos+IDX2C((i*oversamp+ii)%wanniern,(j*oversamp+jj)%wanniern,wanniern))=0.0f;
  //R-loop over nearest neighbours
  for(gpuint nni1=-window;nni1<=(gpuint)window;nni1++)
    for(gpuint nnj1=-window;nnj1<=(gpuint)window;nnj1++) {
      //calculate TG0(0,R)
      matrixmultiplication(scat,g0+IDX4C((kpoints+kpoints-(ipos+nni1))%kpoints,(kpoints+kpoints-(jpos+nnj1))%kpoints,0,0,kpoints,bands),buffer,bands);
      //R'-loop over nearest neighbours
      for(gpuint nni2=-window;nni2<=(gpuint)window;nni2++)
	for(gpuint nnj2=-window;nnj2<=(gpuint)window;nnj2++) {
	  //calculate G0(R',0)TG0(0,R)    
	    matrixmultiplication(g0+IDX4C((kpoints+ipos+nni2)%kpoints,(kpoints+jpos+nnj2)%kpoints,0,0,kpoints,bands),buffer,cldos,bands);	      
	  //sum over sub-unit cell sampling
	  //gpuuint x=ipos*oversamp+ii,y=jpos*oversamp+jj;
	  for(gpuuint ii=0;ii<oversamp;ii++)
	    for(gpuuint jj=0;jj<oversamp;jj++) {
	      gpuuint x=ii+window*oversamp,y=jj+window*oversamp;
	      gpufloat val=*(ldos+IDX2C((i*oversamp+ii)%wanniern,(j*oversamp+jj)%wanniern,wanniern));
	      for(gpuuint o1=0;o1<maxband;o1++)
		for(gpuuint o2=0;o2<maxband;o2++) {
		  gpufloat factor=*(wf+IDX3C(o1,x-nni1*oversamp,y-nnj1*oversamp,wfsize))*(*(wf+IDX3C(o2,x-nni2*oversamp,y-nnj2*oversamp,wfsize)));
		  dblcomplex cval=*(cldos+IDX2C(o2,o1+maxband,bands))+*(g0+IDX4C((kpoints-nni1+nni2)%kpoints,(kpoints-nnj1+nnj2)%kpoints,o2,o1+maxband,kpoints,bands));
		  val-=(cval*tip).imag*factor;
		}
	      *(ldos+IDX2C((i*oversamp+ii)%wanniern,(j*oversamp+jj)%wanniern,wanniern))=val;
	    }
	}
    }
}

//calculate LDOS using Wannier functions - using cached functions
//g0 - two-dimensional array of complex matrices (kpointsxkpointsxbandsxbands)
//ldos - two dimensioanl array of double values (nxn)
//scat - matrix (bandsxbands)
//wf - array of two-dimensional arrays containing orbitals (bandsx((2*windows+1)*oversamp)x((2*windows+1)*oversamp))
//kpoints - number of kpoints, as specified in g0
//n - number of lattice sites in real-space lattice
//bands - number of bands
//oversamp - oversampling of real-space lattice
//window - window used for orbitals
//maxband - band up to which the orbitals are included (e.g. for superconducting calculation)
//spin - spin-polarized calculation
__global__ void gpucalcwannierjosephsonspin(const dblcomplex *g0, gpufloat *ldos,const dblcomplex *scat,const gpufloat *wf,dblcomplex *globalbuffer,dblcomplex *globalcldos,const dblcomplex tip,gpuuint kpoints,gpuuint n,gpuuint bands,gpuuint oversamp,gpuuint window,gpuuint maxband) {
  gpuuint wanniern=oversamp*n;
  gpuuint n2=(n>>1);
  gpuuint wfsize=(2*window+1)*oversamp;
  gpuuint spinbands=maxband>>1;
  gpuuint globalindex=blockIdx.x * blockDim.x + threadIdx.x;
  gpuuint i=globalindex%n,j=globalindex/n;
  if(j>=n) return;
  //sum over lattice
  dblcomplex *buffer=globalbuffer+bands*bands*globalindex,
    *cldos=globalcldos+bands*bands*globalindex;
  gpuint ipos=(gpuint)i-n2, jpos=(gpuint)j-n2;
  for(gpuuint ii=0;ii<oversamp;ii++)
    for(gpuuint jj=0;jj<oversamp;jj++)
      *(ldos+IDX2C((i*oversamp+ii)%wanniern,(j*oversamp+jj)%wanniern,wanniern))=0.0f;
  //R-loop over nearest neighbours
  for(gpuint nni1=-window;nni1<=(gpuint)window;nni1++)
    for(gpuint nnj1=-window;nnj1<=(gpuint)window;nnj1++) {
      //calculate TG0(0,R)
      matrixmultiplication(scat,g0+IDX4C((kpoints+kpoints-(ipos+nni1))%kpoints,(kpoints+kpoints-(jpos+nnj1))%kpoints,0,0,kpoints,bands),buffer,bands);
      //R'-loop over nearest neighbours
      for(gpuint nni2=-window;nni2<=(gpuint)window;nni2++)
	for(gpuint nnj2=-window;nnj2<=(gpuint)window;nnj2++) {
	  //calculate G0(R',0)TG0(0,R)    
	    matrixmultiplication(g0+IDX4C((kpoints+ipos+nni2)%kpoints,(kpoints+jpos+nnj2)%kpoints,0,0,kpoints,bands),buffer,cldos,bands);
	  //gpuuint x=ipos*oversamp+ii,y=jpos*oversamp+jj;
	  //sum over sub-unit cell sampling
	  for(gpuuint ii=0;ii<oversamp;ii++)
	    for(gpuuint jj=0;jj<oversamp;jj++) {
	      gpuuint x=ii+window*oversamp,y=jj+window*oversamp;
	      gpufloat val=*(ldos+IDX2C((i*oversamp+ii)%wanniern,(j*oversamp+jj)%wanniern,wanniern));
	      for(gpuuint o1=0;o1<spinbands;o1++)
		for(gpuuint o2=0;o2<spinbands;o2++) {
		  gpufloat factor=*(wf+IDX3C(o1,x-nni1*oversamp,y-nnj1*oversamp,wfsize))*(*(wf+IDX3C(o2,x-nni2*oversamp,y-nnj2*oversamp,wfsize)));
		  dblcomplex cval=*(cldos+IDX2C(o2,o1+maxband,bands))+*(g0+IDX4C((kpoints-nni1+nni2)%kpoints,(kpoints-nnj1+nnj2)%kpoints,o2,o1+maxband,kpoints,bands));
		  val-=(cval*tip).imag*factor;
		  factor=*(wf+IDX3C(spinbands+o1,x-nni1*oversamp,y-nnj1*oversamp,wfsize))*(*(wf+IDX3C(spinbands+o2,x-nni2*oversamp,y-nnj2*oversamp,wfsize)));
		  cval=*(cldos+IDX2C(spinbands+o2,spinbands+o1+maxband,bands))+*(g0+IDX4C((kpoints-nni1+nni2)%kpoints,(kpoints-nnj1+nnj2)%kpoints,spinbands+o2,spinbands+o1+maxband,kpoints,bands));
		  val-=(cval*tip).imag*factor;
		}
	      *(ldos+IDX2C((i*oversamp+ii)%wanniern,(j*oversamp+jj)%wanniern,wanniern))=val;
	    }
	}
    }
}

//calculate LDOS using Wannier functions - using cached functions
//kernel void gpucalcwannierldos(device const dblcomplex *g0, device gpufloat *ldos,device const dblcomplex *scat,device const gpufloat *wf,device dblcomplex *gblbuffer,device dblcomplex *gblcldos,constant const qpigpuinfo *qpiinfoblock, uint globalindex [[thread_position_in_grid]])
__global__ void gpucalcwannierldoslist(const dblcomplex *g0, gpufloat *ldos,const dblcomplex *scat,const qpigpuflaglist *flags,const gpuuint *flagoffsets,const gpuuint *flagentries,dblcomplex *gblbuffer,dblcomplex *gblcldos,gpuuint kpoints,gpuuint n,gpuuint bands,gpuuint oversamp,gpuuint window) {
  gpuuint wanniern=oversamp*n;
  gpuuint n2=(n>>1);
  gpuuint winn=(2*window+1);
  gpuuint globalindex=blockIdx.x * blockDim.x + threadIdx.x;
  gpuuint i=globalindex%n,j=globalindex/n;
  dblcomplex *buffer=gblbuffer+bands*bands*globalindex,
    *cldos=gblcldos+bands*bands*globalindex;
  if(j>=n) return;
  gpuint ipos=(gpuint)i-n2, jpos=(gpuint)j-n2;
  for(gpuuint ii=0;ii<oversamp;ii++)
    for(gpuuint jj=0;jj<oversamp;jj++)
      *(ldos+IDX2C((i*oversamp+ii)%wanniern,(j*oversamp+jj)%wanniern,wanniern))=0.0;
  //R-loop over nearest neighbours
  for(gpuint nni1=-window;nni1<=(gpuint)window;nni1++)
    for(gpuint nnj1=-window;nnj1<=(gpuint)window;nnj1++) {
      //calculate TG0(0,R)
      matrixmultiplication(scat,g0+IDX4C((kpoints+kpoints-(ipos+nni1))%kpoints,(kpoints+kpoints-(jpos+nnj1))%kpoints,0,0,kpoints,bands),buffer,bands);
      //matrixmultiplication(scat,g0[(kpoints+kpoints-(ipos+nni1))%kpoints][(kpoints+kpoints-(jpos+nnj1))%kpoints],buffer,bands);
      //R'-loop over nearest neighbours
      for(gpuint nni2=-window;nni2<=(gpuint)window;nni2++)
	for(gpuint nnj2=-window;nnj2<=(gpuint)window;nnj2++) {
	  //calculate G0(R',0)TG0(0,R)
	  matrixmultiplication(g0+IDX4C((kpoints+ipos+nni2)%kpoints,(kpoints+jpos+nnj2)%kpoints,0,0,kpoints,bands),buffer,cldos,bands);
	  //matrixmultiplication(g0[(kpoints+ipos+nni2)%kpoints][(kpoints+jpos+nnj2)%kpoints],buffer,cldos,bands);
	  //sum over sub-unit cell sampling
	  gpuuint listlen=*(flagentries+IDX4CC(nni1+window,nnj1+window,nni2+window,nnj2+window,winn));
	  //flags[nni1+window][nnj1+window][nni2+window][nnj2+window].size();
	  gpuuint listofs=*(flagoffsets+IDX4CC(nni1+window,nnj1+window,nni2+window,nnj2+window,winn));
	  for(gpuuint ind=0;ind<listlen;ind++) {
	    gpuuint ii=(flags+listofs+ind)->i,
	      jj=(flags+listofs+ind)->j,
	      o1=(flags+listofs+ind)->o1,
	      o2=(flags+listofs+ind)->o2;
	    gpufloat factor=(flags+listofs+ind)->factor;
	    gpufloat val=*(ldos+IDX2C((i*oversamp+ii)%wanniern,(j*oversamp+jj)%wanniern,wanniern));
	    val-=(cldos+IDX2C(o2,o1,bands))->imag*factor;
	    val-=(g0+IDX4C((kpoints-nni1+nni2)%kpoints,(kpoints-nnj1+nnj2)%kpoints,o2,o1,kpoints,bands))->imag*factor;
	    *(ldos+IDX2C((i*oversamp+ii)%wanniern,(j*oversamp+jj)%wanniern,wanniern))=val;
	  }
	}
    }
}

//calculate spectral function
__global__ void gpucalcspf(const dblcomplex *g0, gpufloat *ldos,gpuuint kpoints,gpuuint n,gpuuint bands,gpuuint maxband) {
  gpuuint n2=(n>>1); //this fixes the centre being in the wrong position
  gpuuint globalindex=blockIdx.x * blockDim.x + threadIdx.x;
  gpuuint i=globalindex%n,j=globalindex/n;
  if(j>=n) return;
  gpufloat trace=0.0;
  for(gpuuint k=0;k<maxband;k++)
    //calculate G0(0,R')=G0(0,R')
    trace-=(g0+IDX4C(i,j,k,k,kpoints,bands))->imag;
  *(ldos+IDX2C((n+i-n2)%n,(n+j-n2)%n,n))=trace;
}

//calculate unfolded spectral function
__global__ void gpucalcuspf(const dblcomplex *g0, const gpufloat *pos,gpufloat *ldos,gpuuint kpoints,gpuuint n,gpuuint bands,gpuuint maxband) {
  gpuuint n2=(n>>1); //this fixes the centre being in the wrong position
  gpuuint globalindex=blockIdx.x * blockDim.x + threadIdx.x;
  gpuuint i=globalindex%n,j=globalindex/n;
  if(j>=n) return;
  gpufloat trace=0.0;
  gpufloat kx=((gpufloat)i-n2)/(gpufloat)kpoints,ky=((gpufloat)j-n2)/(gpufloat)kpoints;
  gpuint ki=((i+n2)%n+kpoints)%kpoints,kj=((j+n2)%n+kpoints)%kpoints;
  for(gpuuint k=0;k<maxband;k++)
    for(gpuuint l=0;l<maxband;l++) {
      //calculate G0(0,R')=G0(0,R')
      dblcomplex prefact=cmpexp((dblcomplex){0.0,2.0*CUDART_PI*(kx*(*(pos+IDX2C(0,k,3))-*(pos+IDX2C(0,l,3)))+ky*(*(pos+IDX2C(1,k,3))-*(pos+IDX2C(1,l,3))))});
      gpufloat flpref=(*(pos+IDX2C(2,k,3)))*(*(pos+IDX2C(2,l,3)));
      trace-=(*(g0+IDX4C(ki,kj,l,k,kpoints,bands))*prefact).imag*flpref;
    }
  *(ldos+IDX2C(i,j,n))=trace;
}

//calculate unfolded spectral function
__global__ void gpucalcuspfspin(const dblcomplex *g0, const gpufloat *pos,gpufloat *ldos,gpuuint kpoints,gpuuint n,gpuuint bands,gpuuint maxband) {
  gpuuint n2=(n>>1); //this fixes the centre being in the wrong position
  gpuuint spinbands=maxband>>1;
  gpuuint globalindex=blockIdx.x * blockDim.x + threadIdx.x;
  gpuuint i=globalindex%n,j=globalindex/n;
  if(j>=n) return;
  gpufloat trace=0.0;
  gpufloat kx=((gpufloat)i-n2)/(gpufloat)kpoints,ky=((gpufloat)j-n2)/(gpufloat)kpoints;
  gpuint ki=((i+n2)%n+kpoints)%kpoints,kj=((j+n2)%n+kpoints)%kpoints;
  for(gpuuint k=0;k<spinbands;k++)
    for(gpuuint l=0;l<spinbands;l++) {
      //calculate G0(0,R')=G0(0,R')
      dblcomplex prefact=cmpexp((dblcomplex){0.0,2.0*CUDART_PI*(kx*(*(pos+IDX2C(0,k,3))-*(pos+IDX2C(0,l,3)))+ky*(*(pos+IDX2C(1,k,3))-*(pos+IDX2C(1,l,3))))});
      gpufloat flpref=(*(pos+IDX2C(2,k,3)))*(*(pos+IDX2C(2,l,3)));
      trace-=(*(g0+IDX4C(ki,kj,l,k,kpoints,bands))*prefact).imag*flpref;
      prefact=cmpexp((dblcomplex){0.0,2.0*CUDART_PI*(kx*(*(pos+IDX2C(0,spinbands+k,3))-*(pos+IDX2C(0,spinbands+l,3)))+ky*(*(pos+IDX2C(1,spinbands+k,3))-*(pos+IDX2C(1,spinbands+l,3))))});
      flpref=(*(pos+IDX2C(2,spinbands+k,3)))*(*(pos+IDX2C(2,spinbands+l,3)));
      trace-=(*(g0+IDX4C(ki,kj,spinbands+l,spinbands+k,kpoints,bands))*prefact).imag*flpref;
    }
  *(ldos+IDX2C(i,j,n))=trace;
}

#define gpuErrchk(ostr,ans) { gpuAssert((ostr),(ans), __FILE__, __LINE__); }

inline void gpuAssert(ostream &os,cudaError_t code, const char *file, int line, bool abort=true)
{
  if (code != cudaSuccess) {
    os<<"CPU CUDA error: "<<cudaGetErrorString(code)<<" thrown in "<<file<<", l."<<line<<endl;
    if (abort) exit(code);
  }
}

//note: nkpts needs to be at least (n+1)
void CudaQPI::copygf(vector<vector<gsl_matrix_complex *> > &g0) {
  int n2=nkpts>>1;
#pragma omp parallel for
  for(int i=-n2;i<n2;i++)
    for(int j=-n2;j<n2;j++)
      for(size_t k=0;k<bands;k++)
	for(size_t l=0;l<bands;l++) {
	  gsl_complex c=gsl_matrix_complex_get(g0[(kpoints+i)%kpoints][(kpoints+j)%kpoints],k,l);
	  *(gpug0mem+IDX4C((nkpts+i)%nkpts,(nkpts+j)%nkpts,k,l,nkpts,bands))=(dblcomplex){(gpufloat)GSL_REAL(c),(gpufloat)GSL_IMAG(c)};
	}
}

CudaQPI::CudaQPI(size_t wanniern,size_t kpoints,size_t n,size_t window,size_t oversamp,size_t bands,size_t maxband,bool spin,vector<wannierfunctions> &wf):wanniern(wanniern),kpoints(kpoints),n(n),window(window),oversamp(oversamp),bands(bands),maxband(maxband),spin(spin) {
#ifdef _mpi_version
  if(world_size>1) {
    int count;
    cudaGetDeviceCount(&count);
    cudaSetDevice(world_rank%count);
  }
#endif
  gpumem=0; cpumem=0; nkpts=n+2*window+2;
  scatsize=sizeof(dblcomplex)*bands*bands; g0size=sizeof(dblcomplex)*bands*bands*nkpts*nkpts; ldossize=wanniern*wanniern*sizeof(gpufloat);
  size_t wfsize=maxband*(2*window+1)*oversamp*(2*window+1)*oversamp*sizeof(gpufloat),bufsize=n*n*bands*bands*sizeof(dblcomplex);
  gpuscat=new dblcomplex[bands*bands]; cpumem+=bands*bands*sizeof(dblcomplex);
  gpuErrchk(cerr,cudaMalloc((void **)&d_gpuscat, scatsize)); gpumem+=scatsize;
  gpug0mem=new dblcomplex[bands*bands*nkpts*nkpts]; cpumem+=bands*bands*nkpts*nkpts*sizeof(dblcomplex);
  gpuErrchk(cerr,cudaMalloc((void **)&d_gpug0mem, g0size)); gpumem+=g0size;
  gpufloat *gpuwf=new gpufloat[bands*(2*window+1)*oversamp*(2*window+1)*oversamp];
  gpuErrchk(cerr,cudaMalloc((void **)&d_gpuwf, wfsize)); gpumem+=wfsize;
  gpuErrchk(cerr,cudaMalloc((void **)&d_gpuldosmem, ldossize)); gpumem+=ldossize;
  gpuldosmem=NULL;
  for(size_t i=0;i<maxband;i++)
    for(size_t j=0;j<(2*window+1)*oversamp;j++)
      for(size_t k=0;k<(2*window+1)*oversamp;k++)
	*(gpuwf+IDX3C(i,j,k,(2*window+1)*oversamp))=wf[i].getwave_cached(j,k);
  cudaMemcpy(d_gpuwf, gpuwf, wfsize, cudaMemcpyHostToDevice);
  free(gpuwf);
  gpuErrchk(cerr,cudaMalloc((void **)&d_gpubuf,bufsize)); gpumem+=bufsize;
  gpuErrchk(cerr,cudaMalloc((void **)&d_gpucldos,bufsize)); gpumem+=bufsize;
  //cudaDeviceSetLimit(cudaLimitMallocHeapSize,);
  //setting parameters for GPU  
  size_t total=n*n;
  ldosblocksize=512;
  ldosblocks=total/ldosblocksize;
  if(total%ldosblocksize) ldosblocks++;
  d_gpucontg=NULL; d_mpos=NULL;
}

CudaQPI::CudaQPI(size_t wanniern, size_t kpoints,size_t n,size_t window, size_t oversamp,size_t bands,flaglist &flist):wanniern(wanniern),kpoints(kpoints),n(n),window(window),oversamp(oversamp),bands(bands) {
#ifdef _mpi_version
  if(world_size>1) {
    int count;
    cudaGetDeviceCount(&count);
    cudaSetDevice(world_rank%count);
  }
#endif
  gpumem=0; cpumem=0; nkpts=n+2*window+2;
  size_t winn=2*window+1,nwinn=winn*winn,nwinn2=nwinn*nwinn,tflentries=0;
  scatsize=sizeof(dblcomplex)*bands*bands; g0size=sizeof(dblcomplex)*bands*bands*nkpts*nkpts; ldossize=wanniern*wanniern*sizeof(gpufloat);
  size_t bufsize=n*n*bands*bands*sizeof(dblcomplex);
  gpuscat=new dblcomplex[bands*bands]; cpumem+=bands*bands*sizeof(dblcomplex);
  gpuErrchk(cerr,cudaMalloc((void **)&d_gpuscat, scatsize)); gpumem+=scatsize;
  gpug0mem=new dblcomplex[bands*bands*nkpts*nkpts]; cpumem+=bands*bands*nkpts*nkpts*sizeof(dblcomplex);
  gpuErrchk(cerr,cudaMalloc((void **)&d_gpug0mem, g0size)); gpumem+=g0size;
  d_gpuwf=NULL;
  gpuErrchk(cerr,cudaMalloc((void **)&d_gpuldosmem, ldossize)); gpumem+=ldossize;
  gpuldosmem=NULL;
  gpuuint *flgentr,*flgofs;
  size_t flagarrsize=nwinn2*sizeof(gpuuint);
  qpigpuflaglist *gpuflg;
  flgofs=new gpuuint[nwinn2];
  gpuErrchk(cerr,cudaMalloc((void **)&d_flagofs, flagarrsize)); gpumem+=flagarrsize;
  flgentr=new gpuuint[nwinn2];
  gpuErrchk(cerr,cudaMalloc((void **)&d_flagentries, flagarrsize)); gpumem+=flagarrsize;
  for(size_t i=0;i<winn;i++)
    for(size_t j=0;j<winn;j++)
      for(size_t k=0;k<winn;k++)
	for(size_t l=0;l<winn;l++) {
	  size_t ofs=IDX4CC(i,j,k,l,winn);
	  *(flgofs+ofs)=tflentries;
	  *(flgentr+ofs)=flist[i][j][k][l].size();
	  tflentries+=flist[i][j][k][l].size();
	}
  size_t flsize=tflentries*sizeof(qpigpuflaglist);
  gpuflg=new qpigpuflaglist[tflentries];
  gpuErrchk(cerr,cudaMalloc((void **)&d_flaglist, flsize)); gpumem+=flsize;
  for(size_t i=0;i<winn;i++)
    for(size_t j=0;j<winn;j++)
      for(size_t k=0;k<winn;k++)
	for(size_t l=0;l<winn;l++) {
	  size_t ofs=IDX4CC(i,j,k,l,winn),
	    flpos=*(flgofs+ofs),
	    fllen=*(flgentr+ofs);
	  for(size_t m=0;m<fllen;m++) {
	    (gpuflg+flpos+m)->i=flist[i][j][k][l][m].i;
	    (gpuflg+flpos+m)->j=flist[i][j][k][l][m].j;
	    (gpuflg+flpos+m)->o1=flist[i][j][k][l][m].o1;
	    (gpuflg+flpos+m)->o2=flist[i][j][k][l][m].o2;
	    (gpuflg+flpos+m)->factor=flist[i][j][k][l][m].factor;
	  }
	}
  cudaMemcpy(d_flagofs, flgofs, flagarrsize, cudaMemcpyHostToDevice);
  cudaMemcpy(d_flagentries, flgentr, flagarrsize, cudaMemcpyHostToDevice);
  cudaMemcpy(d_flaglist, gpuflg, flsize, cudaMemcpyHostToDevice);
  free(gpuflg); free(flgofs); free(flgentr);
  gpuErrchk(cerr,cudaMalloc((void **)&d_gpubuf,bufsize)); gpumem+=bufsize;
  gpuErrchk(cerr,cudaMalloc((void **)&d_gpucldos,bufsize)); gpumem+=bufsize;
  //cudaDeviceSetLimit(cudaLimitMallocHeapSize,);
  //setting parameters for GPU  
  size_t total=n*n;
  ldosblocksize=512;
  ldosblocks=total/ldosblocksize;
  if(total%ldosblocksize) ldosblocks++;
  d_gpucontg=NULL; d_mpos=NULL;
}

CudaQPI::CudaQPI(size_t kpoints,size_t n,size_t bands,size_t maxbands,bool spin,vector<vector<double> > &pos,vector<double> &prearr):kpoints(kpoints),n(n),wanniern(n),bands(bands),maxband(maxbands),spin(spin) {
#ifdef _mpi_version
  if(world_size>1) {
    int count;
    cudaGetDeviceCount(&count);
    cudaSetDevice(world_rank%count);
  }
#endif
  gpumem=0; cpumem=0; nkpts=kpoints;
  scatsize=0; g0size=sizeof(dblcomplex)*bands*bands*nkpts*nkpts; ldossize=wanniern*wanniern*sizeof(gpufloat);
  gpug0mem=new dblcomplex[bands*bands*nkpts*nkpts]; cpumem+=bands*bands*nkpts*nkpts*sizeof(dblcomplex);
  gpuErrchk(cerr,cudaMalloc((void **)&d_gpug0mem, g0size)); gpumem+=g0size;
  gpuErrchk(cerr,cudaMalloc((void **)&d_gpuldosmem, ldossize)); gpumem+=ldossize;
  gpuscat=NULL; gpuldosmem=NULL; d_gpuscat=NULL; d_gpubuf=NULL; d_gpucldos=NULL; d_gpucontg=NULL;
  if(pos.size()) {
    size_t possize=pos.size()*3*sizeof(gpufloat);
    gpuErrchk(cerr,cudaMalloc((void **)&d_mpos, possize)); gpumem+=possize;
    gpufloat *gpupos= new gpufloat[3*pos.size()];
    for(size_t i=0;i<pos.size();i++) {
      for(size_t j=0;j<2;j++)
	*(gpupos+IDX2C(j,i,3))=pos[i][j];
      *(gpupos+IDX2C(2,i,3))=prearr[i%prearr.size()];
    }
    cudaMemcpy(d_mpos, gpupos, possize, cudaMemcpyHostToDevice);
    free(gpupos);
  }
  
  //cudaDeviceSetLimit(cudaLimitMallocHeapSize,);
  //setting parameters for GPU  
  size_t total=n*n;
  ldosblocksize=512;
  ldosblocks=total/ldosblocksize;
  if(total%ldosblocksize) ldosblocks++;
  d_gpucontg=NULL;
}

void CudaQPI::printinfo(ostream &os) {
  int numdevices;
  cudaGetDeviceCount(&numdevices);
  if(numdevices==0)
    os<<"Error: no CUDA GPUs found."<<endl;
  else
    os<<"Running continuum QPI on Cuda GPU."<<endl;
  for(int device=0;device<numdevices;device++) {
    struct cudaDeviceProp props;
    cudaGetDeviceProperties(&props, device);
#ifdef _mpi_version
    if(world_size>1) {
      os<<"MPI Task "<<world_rank<<"/"<<world_size<<": "<<endl;
      if((world_rank%numdevices)==device) os<<"CUDA GPU "<<device<<"/"<<numdevices<<": "<<props.name<<endl;
    } else
#endif
      os<<"CUDA GPU "<<device<<"/"<<numdevices<<": "<<props.name<<endl;
    ExecuteCPU0 {
      os<<"Additional memory requirements:"<<endl
	<<"GPU memory:       "<<std::fixed<<std::setprecision(2)<<(double)gpumem/1024.0/1024.0/1024.0<<"GB"<<endl
	<<"CPU memory:       "<<std::fixed<<std::setprecision(2)<<(double)cpumem/1024.0/1024.0/1024.0<<"GB"<<endl
	<<"Blocksize:        "<<ldosblocksize<<endl
	<<"Number of blocks: "<<ldosblocks<<endl;
    }
  }
}

CudaQPI::~CudaQPI() {
  free(gpug0mem); cudaFree(d_gpug0mem); cudaFree(d_gpuldosmem);
  if(gpuldosmem) free(gpuldosmem);
  if(gpuscat) free(gpuscat);
  if(d_gpuscat) cudaFree(d_gpuscat);
  if(d_gpubuf) cudaFree(d_gpubuf);
  if(d_gpucldos) cudaFree(d_gpucldos);
  if(d_gpucontg) cudaFree(d_gpucontg);
  if(d_gpuwf) cudaFree(d_gpuwf);
  if(d_flaglist) cudaFree(d_flaglist);
  if(d_flagentries) cudaFree(d_flagentries);
  if(d_flagofs) cudaFree(d_flagofs);
  if(d_mpos) cudaFree(d_mpos);
}

void CudaQPI::wannierldos(gsl_matrix_complex *scat, vector<vector<gsl_matrix_complex *> > &g0)
{
  //scattering matrix
  for(size_t i=0;i<bands;i++)
    for(size_t j=0;j<bands;j++) {
      gsl_complex c=gsl_matrix_complex_get(scat,i,j);
      *(gpuscat+IDX2C(i,j,bands))=(dblcomplex){(gpufloat)GSL_REAL(c),(gpufloat)GSL_IMAG(c)};
    }
  cudaMemcpy(d_gpuscat, gpuscat, scatsize, cudaMemcpyHostToDevice);
  copygf(g0);
  cudaMemcpy(d_gpug0mem, gpug0mem, g0size, cudaMemcpyHostToDevice);
  // cleartotal=wanniern*wanniern;
  // clearblocksize=512;
  // clearblocks=cleartotal/clearblocksize;
  // if(cleartotal%clearblocksize) clearblocks++;
  //clearldos<<<clearblocks,clearblocksize>>>(d_gpuldosmem,cleartotal);
  //cudaDeviceSynchronize();
  if(spin)
    gpucalcwannierldosspin<<<ldosblocks,ldosblocksize>>>(d_gpug0mem,d_gpuldosmem,d_gpuscat,d_gpuwf,d_gpubuf,d_gpucldos,nkpts,n,bands,oversamp,window,maxband);
  else
    gpucalcwannierldos<<<ldosblocks,ldosblocksize>>>(d_gpug0mem,d_gpuldosmem,d_gpuscat,d_gpuwf,d_gpubuf,d_gpucldos,nkpts,n,bands,oversamp,window,maxband);
}

void CudaQPI::wannierjosephson(gsl_matrix_complex *scat, vector<vector<gsl_matrix_complex *> > &g0,gsl_complex tip)
{
  //scattering matrix
  for(size_t i=0;i<bands;i++)
    for(size_t j=0;j<bands;j++) {
      gsl_complex c=gsl_matrix_complex_get(scat,i,j);
      *(gpuscat+IDX2C(i,j,bands))=(dblcomplex){(gpufloat)GSL_REAL(c),(gpufloat)GSL_IMAG(c)};
    }
  cudaMemcpy(d_gpuscat, gpuscat, scatsize, cudaMemcpyHostToDevice);
  copygf(g0);
  cudaMemcpy(d_gpug0mem, gpug0mem, g0size, cudaMemcpyHostToDevice);
  // cleartotal=wanniern*wanniern;
  // clearblocksize=512;
  // clearblocks=cleartotal/clearblocksize;
  // if(cleartotal%clearblocksize) clearblocks++;
  //clearldos<<<clearblocks,clearblocksize>>>(d_gpuldosmem,cleartotal);
  //cudaDeviceSynchronize();
  dblcomplex ctip=(dblcomplex){(gpufloat)GSL_REAL(tip),(gpufloat)GSL_IMAG(tip)};
  if(spin)
    gpucalcwannierjosephsonspin<<<ldosblocks,ldosblocksize>>>(d_gpug0mem,d_gpuldosmem,d_gpuscat,d_gpuwf,d_gpubuf,d_gpucldos,ctip,nkpts,n,bands,oversamp,window,maxband);
  else
    gpucalcwannierjosephson<<<ldosblocks,ldosblocksize>>>(d_gpug0mem,d_gpuldosmem,d_gpuscat,d_gpuwf,d_gpubuf,d_gpucldos,ctip,nkpts,n,bands,oversamp,window,maxband);
}

void CudaQPI::wannierldoslist(gsl_matrix_complex *scat, vector<vector<gsl_matrix_complex *> > &g0)
{
  //scattering matrix
  for(size_t i=0;i<bands;i++)
    for(size_t j=0;j<bands;j++) {
      gsl_complex c=gsl_matrix_complex_get(scat,i,j);
      *(gpuscat+IDX2C(i,j,bands))=(dblcomplex){(gpufloat)GSL_REAL(c),(gpufloat)GSL_IMAG(c)};
    }
  cudaMemcpy(d_gpuscat, gpuscat, scatsize, cudaMemcpyHostToDevice);
  copygf(g0);
  cudaMemcpy(d_gpug0mem, gpug0mem, g0size, cudaMemcpyHostToDevice);
  gpucalcwannierldoslist<<<ldosblocks,ldosblocksize>>>(d_gpug0mem,d_gpuldosmem,d_gpuscat,d_flaglist,d_flagofs,d_flagentries,d_gpubuf,d_gpucldos,nkpts,n,bands,oversamp,window);
}

void CudaQPI::spf(vector<vector<gsl_matrix_complex *> > &g0)
{
  copygf(g0);
  cudaMemcpy(d_gpug0mem, gpug0mem, g0size, cudaMemcpyHostToDevice);
  gpucalcspf<<<ldosblocks,ldosblocksize>>>(d_gpug0mem,d_gpuldosmem,nkpts,n,bands,maxband);
}

void CudaQPI::uspf(vector<vector<gsl_matrix_complex *> > &g0)
{
  copygf(g0);
  cudaMemcpy(d_gpug0mem, gpug0mem, g0size, cudaMemcpyHostToDevice);
  if(spin)
    gpucalcuspfspin<<<ldosblocks,ldosblocksize>>>(d_gpug0mem,d_mpos,d_gpuldosmem,nkpts,n,bands,maxband);
  else
    gpucalcuspf<<<ldosblocks,ldosblocksize>>>(d_gpug0mem,d_mpos,d_gpuldosmem,nkpts,n,bands,maxband);
}

void CudaQPI::retrieveResult(vector<vector<double> > &ldos)
{
  if(!gpuldosmem) {
    gpuldosmem=new gpufloat[wanniern*wanniern];
    cpumem+=wanniern*wanniern*sizeof(gpufloat);
  }
  cudaDeviceSynchronize();
  cudaMemcpy(gpuldosmem, d_gpuldosmem, ldossize, cudaMemcpyDeviceToHost);
  ldos.resize(wanniern);
#pragma omp parallel
  {
#pragma omp for
    for(size_t i=0;i<wanniern;i++)
      ldos[i].resize(wanniern);
#pragma omp for
    for(size_t i=0;i<wanniern;i++)
      for(size_t j=0;j<wanniern;j++)
	ldos[i][j]=*(gpuldosmem+IDX2C(i,j,wanniern));
  }
}

void CudaQPI::retrieveResult(idl &map,size_t layer)
{
  if(!gpuldosmem) {
    gpuldosmem=new gpufloat[wanniern*wanniern];
    cpumem+=wanniern*wanniern*sizeof(gpufloat);
  }
  cudaDeviceSynchronize();
  cudaMemcpy(gpuldosmem, d_gpuldosmem, ldossize, cudaMemcpyDeviceToHost);
#pragma omp parallel for
  for(size_t i=0;i<wanniern;i++)
    for(size_t j=0;j<wanniern;j++)
      map.set(i,j,layer,*(gpuldosmem+IDX2C(i,j,wanniern))); 
}

void CudaQPI::retrieveResult(double *ldos)
{
  cudaDeviceSynchronize();
  if(typeid(gpufloat)!=typeid(double)) {
    if(!gpuldosmem) {
      gpuldosmem=new gpufloat[wanniern*wanniern];
      cpumem+=wanniern*wanniern*sizeof(gpufloat);
    }
    cudaMemcpy(gpuldosmem, d_gpuldosmem, ldossize, cudaMemcpyDeviceToHost);
#pragma omp parallel for
    for(size_t i=0;i<wanniern*wanniern;i++)
      *(ldos+i)=*(gpuldosmem+i);
  } else
    cudaMemcpy(ldos, d_gpuldosmem, ldossize, cudaMemcpyDeviceToHost);
}
