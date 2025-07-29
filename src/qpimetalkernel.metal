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

#include <metal_stdlib>
#include "qpimetalkernel.h"

using namespace metal;

inline dblcomplex operator+(const dblcomplex a,const dblcomplex b) { return (dblcomplex){a.real+b.real,a.imag+b.imag}; }
inline dblcomplex operator*(const dblcomplex a,const dblcomplex b) { return (dblcomplex){a.real*b.real-a.imag*b.imag,a.real*b.imag+a.imag*b.real}; }

inline dblcomplex operator*(const gpufloat a,const dblcomplex b) { return (dblcomplex){a*b.real,a*b.imag}; }

inline dblcomplex cplxinverse(const dblcomplex x) { gpufloat base=x.real*x.real+x.imag*x.imag; return (dblcomplex){x.real/base,-x.imag/base}; }

inline dblcomplex cmpexp(const dblcomplex x) { gpufloat prod=exp(x.real); return (dblcomplex){prod*cos(x.imag),prod*sin(x.imag)}; }

inline void clearmatrix(device dblcomplex *a, gpuuint n)
{
  for(gpuuint i=0;i<n*n;i++)
    *(a+i)=(dblcomplex){0.0,0.0};
}

inline void vectormultiplication(const thread dblcomplex *alpha,constant const dblcomplex *a,constant const dblcomplex *b,device dblcomplex *c,gpuuint n)
{
  for(gpuuint i=0;i<n;i++)
    for(gpuuint j=0;j<n;j++)
      *(c+IDX2C(i,j,n))=*(c+IDX2C(i,j,n))+(*alpha)*((*(a+i))*(*(b+j)));
}

inline void vectormultiplication(const thread dblcomplex *alpha,constant const dblcomplex *a,device const dblcomplex *b,device dblcomplex *c,gpuuint n)
{
  for(gpuuint i=0;i<n;i++)
    for(gpuuint j=0;j<n;j++)
      *(c+IDX2C(i,j,n))=*(c+IDX2C(i,j,n))+(*alpha)*((*(a+i))*(*(b+j)));
}

inline void vectormultiplication(const thread dblcomplex *alpha,device const dblcomplex *a,device const dblcomplex *b,device dblcomplex *c,gpuuint n)
{
  for(gpuuint i=0;i<n;i++)
    for(gpuuint j=0;j<n;j++)
      *(c+IDX2C(i,j,n))=*(c+IDX2C(i,j,n))+(*alpha)*((*(a+i))*(*(b+j)));
}

inline void matrixmultiplication(const thread dblcomplex *a,constant const dblcomplex *b,device dblcomplex *c,gpuuint n)
{
  for(gpuuint i=0;i<n*n;i++)
    *(c+i)=(dblcomplex){0.0,0.0};
  for(gpuuint i=0;i<n;i++)
    for(gpuuint j=0;j<n;j++)
      for(gpuuint k=0;k<n;k++)
	*(c+IDX2C(i,j,n))=*(c+IDX2C(i,j,n))+((*(a+IDX2C(i,k,n)))*(*(b+IDX2C(k,j,n))));
}

inline void matrixmultiplication(constant const dblcomplex *a,device const dblcomplex *b,device dblcomplex *c,gpuuint n)
{
  for(gpuuint i=0;i<n*n;i++)
    *(c+i)=(dblcomplex){0.0,0.0};
  for(gpuuint i=0;i<n;i++)
    for(gpuuint j=0;j<n;j++)
      for(gpuuint k=0;k<n;k++)
	*(c+IDX2C(i,j,n))=*(c+IDX2C(i,j,n))+((*(a+IDX2C(i,k,n)))*(*(b+IDX2C(k,j,n))));
}

inline void matrixmultiplication(device const dblcomplex *a, device const dblcomplex *b, device dblcomplex *c, gpuuint n)
{
  for(gpuuint i=0;i<n*n;i++)
    *(c+i)=(dblcomplex){0.0,0.0};
  for(gpuuint i=0;i<n;i++)
    for(gpuuint j=0;j<n;j++)
      for(gpuuint k=0;k<n;k++)
	*(c+IDX2C(i,j,n))=*(c+IDX2C(i,j,n))+((*(a+IDX2C(i,k,n)))*(*(b+IDX2C(k,j,n))));
}

kernel void gpucalcgreensfunction(device dblcomplex *g0,constant gpufloat *eval, constant dblcomplex *evect,constant gfgpuinfo *gfinfo,uint globalindex [[thread_position_in_grid]])
{
  gpuuint n=gfinfo->bands;
  if(globalindex>=gfinfo->totalkpoints) return;
  gpuuint gmindex=globalindex*n*n,evindex=globalindex*n;
  clearmatrix(g0+gmindex,n);
  for(gpuuint i=0;i<n;i++) {
    dblcomplex factor=cplxinverse((dblcomplex){gfinfo->omega-*(eval+evindex+i),gfinfo->eta});
    vectormultiplication((thread dblcomplex *)&factor,evect+gmindex+i*n,evect+gmindex+i*n,g0+gmindex,n);
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
kernel void gpucalcwannierldos(device const dblcomplex *g0, device gpufloat *ldos,device const dblcomplex *scat,device const gpufloat *wf,device dblcomplex *gblbuffer,device dblcomplex *gblcldos,constant const qpigpuinfo *qpiinfoblock, uint globalindex [[thread_position_in_grid]]) {
  gpuuint n=qpiinfoblock->n,kpoints=qpiinfoblock->kpoints,bands=qpiinfoblock->bands,oversamp=qpiinfoblock->oversamp;
  gpuint window=qpiinfoblock->window;
  gpuuint maxband=qpiinfoblock->maxband,wanniern=qpiinfoblock->wanniern;
  gpuuint n2=(n>>1);
  gpuuint wfsize=(2*window+1)*oversamp;
  //sum over lattice
  //gpuuint localoffset=bands*bands*globalindex;
  device dblcomplex *buffer=gblbuffer+bands*bands*globalindex,
    *cldos=gblcldos+bands*bands*globalindex;
  gpuuint i=globalindex%n,j=globalindex/n;
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
      for(gpuint nni2=-window;nni2<=(gpuint)window;nni2++)
	for(gpuint nnj2=-window;nnj2<=(gpuint)window;nnj2++) {
	  //calculate G0(R',0)TG0(0,R)    
	  matrixmultiplication(g0+IDX4C((kpoints+ipos+nni2)%kpoints,(kpoints+jpos+nnj2)%kpoints,0,0,kpoints,bands),buffer,cldos,bands);
	  //sum over sub-unit cell sampling
	  for(gpuuint ii=0;ii<oversamp;ii++)
	    for(gpuuint jj=0;jj<oversamp;jj++) {
	      gpuuint x=ii+window*oversamp,y=jj+window*oversamp;
	      gpufloat val=*(ldos+IDX2C((i*oversamp+ii)%wanniern,(j*oversamp+jj)%wanniern,wanniern));
	      for(gpuuint o1=0;o1<maxband;o1++) {
		//if(!isnan(*(wf+IDX3C(o1,0,0,wfsize))))
		for(gpuuint o2=0;o2<maxband;o2++) {
		  //float c=(gblcldos+localoffset+IDX2C(o2,o1,bands))->imag;
		  //*(ldos+IDX2C((i*oversamp+ii)%wanniern,(j*oversamp+jj)%wanniern,wanniern))=c;//-(gblcldos+nindex)->imag;
		  gpufloat factor=*(wf+IDX3C(o1,x-nni1*oversamp,y-nnj1*oversamp,wfsize))*(*(wf+IDX3C(o2,x-nni2*oversamp,y-nnj2*oversamp,wfsize)));
		  val-=(cldos+IDX2C(o2,o1,bands))->imag*factor;
		  val-=(g0+IDX4C((kpoints-nni1+nni2)%kpoints,(kpoints-nnj1+nnj2)%kpoints,o2,o1,kpoints,bands))->imag*factor;
		}
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
//spin-polarized calculation
kernel void gpucalcwannierldosspin(device const dblcomplex *g0, device gpufloat *ldos,device const dblcomplex *scat,device const gpufloat *wf,device dblcomplex *gblbuffer,device dblcomplex *gblcldos,constant const qpigpuinfo *qpiinfoblock, uint globalindex [[thread_position_in_grid]]) {
  gpuuint n=qpiinfoblock->n,kpoints=qpiinfoblock->kpoints,bands=qpiinfoblock->bands,oversamp=qpiinfoblock->oversamp;
  gpuint window=qpiinfoblock->window;
  gpuuint maxband=qpiinfoblock->maxband,wanniern=qpiinfoblock->wanniern;
  gpuuint n2=(n>>1);
  gpuuint wfsize=(2*window+1)*oversamp;
  gpuuint spinbands=maxband>>1;
  //sum over lattice
  //gpuuint localoffset=bands*bands*globalindex;
  device dblcomplex *buffer=gblbuffer+bands*bands*globalindex,
    *cldos=gblcldos+bands*bands*globalindex;
  gpuuint i=globalindex%n,j=globalindex/n;
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
      for(gpuint nni2=-window;nni2<=(gpuint)window;nni2++)
	for(gpuint nnj2=-window;nnj2<=(gpuint)window;nnj2++) {
	  //calculate G0(R',0)TG0(0,R)    
	    matrixmultiplication(g0+IDX4C((kpoints+ipos+nni2)%kpoints,(kpoints+jpos+nnj2)%kpoints,0,0,kpoints,bands),buffer,cldos,bands);
	  //sum over sub-unit cell sampling
	  for(gpuuint ii=0;ii<oversamp;ii++)
	    for(gpuuint jj=0;jj<oversamp;jj++) {
	      gpuuint x=ii+window*oversamp,y=jj+window*oversamp;
	      //gpuuint x=ipos*oversamp+ii,y=jpos*oversamp+jj;
	      gpufloat val=*(ldos+IDX2C((i*oversamp+ii)%wanniern,(j*oversamp+jj)%wanniern,wanniern));
	      for(gpuuint o1=0;o1<spinbands;o1++) {
		for(gpuuint o2=0;o2<spinbands;o2++) {
		    gpufloat factor=*(wf+IDX3C(o1,x-nni1*oversamp,y-nnj1*oversamp,wfsize))*(*(wf+IDX3C(o2,x-nni2*oversamp,y-nnj2*oversamp,wfsize)));
		    val-=(cldos+IDX2C(o2,o1,bands))->imag*factor;
		    val-=(g0+IDX4C((kpoints-nni1+nni2)%kpoints,(kpoints-nnj1+nnj2)%kpoints,o2,o1,kpoints,bands))->imag*factor;
		    factor=*(wf+IDX3C(spinbands+o1,x-nni1*oversamp,y-nnj1*oversamp,wfsize))*(*(wf+IDX3C(spinbands+o2,x-nni2*oversamp,y-nnj2*oversamp,wfsize)));
		    val-=(cldos+IDX2C(spinbands+o2,spinbands+o1,bands))->imag*factor;
		    val-=(g0+IDX4C((kpoints-nni1+nni2)%kpoints,(kpoints-nnj1+nnj2)%kpoints,spinbands+o2,spinbands+o1,kpoints,bands))->imag*factor;
		  }
	      }
	      *(ldos+IDX2C((i*oversamp+ii)%wanniern,(j*oversamp+jj)%wanniern,wanniern))=val;
	    }
	}
    }
}

//calculate Josephson current using Wannier functions - using cached functions
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
kernel void gpucalcwannierjosephson(device const dblcomplex *g0, device gpufloat *ldos,device const dblcomplex *scat,device const gpufloat *wf,device dblcomplex *gblbuffer,device dblcomplex *gblcldos,constant const qpigpuinfo *qpiinfoblock, uint globalindex [[thread_position_in_grid]]) {
  gpuuint n=qpiinfoblock->n,kpoints=qpiinfoblock->kpoints,bands=qpiinfoblock->bands,oversamp=qpiinfoblock->oversamp;
  gpuint window=qpiinfoblock->window;
  gpuuint maxband=qpiinfoblock->maxband,wanniern=qpiinfoblock->wanniern;
  gpuuint n2=(n>>1);
  gpuuint wfsize=(2*window+1)*oversamp;
  //sum over lattice
  //gpuuint localoffset=bands*bands*globalindex;
  device dblcomplex *buffer=gblbuffer+bands*bands*globalindex,
    *cldos=gblcldos+bands*bands*globalindex;
  const dblcomplex tip=qpiinfoblock->tip;
  gpuuint i=globalindex%n,j=globalindex/n;
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
      for(gpuint nni2=-window;nni2<=(gpuint)window;nni2++)
	for(gpuint nnj2=-window;nnj2<=(gpuint)window;nnj2++) {
	  //calculate G0(R',0)TG0(0,R)    
	  matrixmultiplication(g0+IDX4C((kpoints+ipos+nni2)%kpoints,(kpoints+jpos+nnj2)%kpoints,0,0,kpoints,bands),buffer,cldos,bands);
	  //sum over sub-unit cell sampling
	  for(gpuuint ii=0;ii<oversamp;ii++)
	    for(gpuuint jj=0;jj<oversamp;jj++) {
	      gpuuint x=ii+window*oversamp,y=jj+window*oversamp;
	      gpufloat val=*(ldos+IDX2C((i*oversamp+ii)%wanniern,(j*oversamp+jj)%wanniern,wanniern));
	      for(gpuuint o1=0;o1<maxband;o1++) {
		//if(!isnan(*(wf+IDX3C(o1,0,0,wfsize))))
		for(gpuuint o2=0;o2<maxband;o2++) {
		  //float c=(gblcldos+localoffset+IDX2C(o2,o1,bands))->imag;
		  //*(ldos+IDX2C((i*oversamp+ii)%wanniern,(j*oversamp+jj)%wanniern,wanniern))=c;//-(gblcldos+nindex)->imag;
		  gpufloat factor=*(wf+IDX3C(o1,x-nni1*oversamp,y-nnj1*oversamp,wfsize))*(*(wf+IDX3C(o2,x-nni2*oversamp,y-nnj2*oversamp,wfsize)));
		  dblcomplex cval=*(cldos+IDX2C(o2,o1+maxband,bands))+*(g0+IDX4C((kpoints-nni1+nni2)%kpoints,(kpoints-nnj1+nnj2)%kpoints,o2,o1+maxband,kpoints,bands));
		  val-=(cval*tip).imag*factor;
		}
	      }
	      *(ldos+IDX2C((i*oversamp+ii)%wanniern,(j*oversamp+jj)%wanniern,wanniern))=val;
	    }
	}
    }
}

//calculate LDOS using Josephson current functions - using cached functions
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
kernel void gpucalcwannierjosephsonspin(device const dblcomplex *g0, device gpufloat *ldos,device const dblcomplex *scat,device const gpufloat *wf,device dblcomplex *gblbuffer,device dblcomplex *gblcldos,constant const qpigpuinfo *qpiinfoblock, uint globalindex [[thread_position_in_grid]]) {
  gpuuint n=qpiinfoblock->n,kpoints=qpiinfoblock->kpoints,bands=qpiinfoblock->bands,oversamp=qpiinfoblock->oversamp;
  gpuint window=qpiinfoblock->window;
  gpuuint maxband=qpiinfoblock->maxband,wanniern=qpiinfoblock->wanniern;
  gpuuint n2=(n>>1);
  gpuuint wfsize=(2*window+1)*oversamp;
  gpuuint spinbands=maxband>>1;
  //sum over lattice
  //gpuuint localoffset=bands*bands*globalindex;
  device dblcomplex *buffer=gblbuffer+bands*bands*globalindex,
    *cldos=gblcldos+bands*bands*globalindex;
  const dblcomplex tip=qpiinfoblock->tip;
  gpuuint i=globalindex%n,j=globalindex/n;
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
      for(gpuint nni2=-window;nni2<=(gpuint)window;nni2++)
	for(gpuint nnj2=-window;nnj2<=(gpuint)window;nnj2++) {
	  //calculate G0(R',0)TG0(0,R)    
	    matrixmultiplication(g0+IDX4C((kpoints+ipos+nni2)%kpoints,(kpoints+jpos+nnj2)%kpoints,0,0,kpoints,bands),buffer,cldos,bands);
	  //sum over sub-unit cell sampling
	  for(gpuuint ii=0;ii<oversamp;ii++)
	    for(gpuuint jj=0;jj<oversamp;jj++) {
	      gpuuint x=ii+window*oversamp,y=jj+window*oversamp;
	      //gpuuint x=ipos*oversamp+ii,y=jpos*oversamp+jj;
	      gpufloat val=*(ldos+IDX2C((i*oversamp+ii)%wanniern,(j*oversamp+jj)%wanniern,wanniern));
	      for(gpuuint o1=0;o1<spinbands;o1++) {
		for(gpuuint o2=0;o2<spinbands;o2++) {
		    gpufloat factor=*(wf+IDX3C(o1,x-nni1*oversamp,y-nnj1*oversamp,wfsize))*(*(wf+IDX3C(o2,x-nni2*oversamp,y-nnj2*oversamp,wfsize)));
		    dblcomplex cval=*(cldos+IDX2C(o2,o1+maxband,bands))+*(g0+IDX4C((kpoints-nni1+nni2)%kpoints,(kpoints-nnj1+nnj2)%kpoints,o2,o1+maxband,kpoints,bands));
		    val-=(cval*tip).imag*factor;
		    factor=*(wf+IDX3C(spinbands+o1,x-nni1*oversamp,y-nnj1*oversamp,wfsize))*(*(wf+IDX3C(spinbands+o2,x-nni2*oversamp,y-nnj2*oversamp,wfsize)));
		    cval=*(cldos+IDX2C(spinbands+o2,spinbands+o1+maxband,bands))+*(g0+IDX4C((kpoints-nni1+nni2)%kpoints,(kpoints-nnj1+nnj2)%kpoints,spinbands+o2,spinbands+o1+maxband,kpoints,bands));
		    val-=(cval*tip).imag*factor;
		  }
	      }
	      *(ldos+IDX2C((i*oversamp+ii)%wanniern,(j*oversamp+jj)%wanniern,wanniern))=val;
	    }
	}
    }
}

//calculate spectral function
kernel void gpucalcspf(device const dblcomplex *g0, device gpufloat *ldos,constant const qpigpuinfo *qpiinfoblock,uint globalindex [[thread_position_in_grid]]) {
  gpuuint n=qpiinfoblock->n,kpoints=qpiinfoblock->kpoints,bands=qpiinfoblock->bands;
  gpuuint n2=(n>>1); //this fixes the centre being in the wrong position
  gpuuint maxband=qpiinfoblock->maxband;
  gpuuint i=globalindex%n,j=globalindex/n;
  if(j>=n) return;
  gpufloat trace=0.0;
  for(gpuuint k=0;k<maxband;k++)
    //calculate G0(0,R')=G0(0,R')
    trace-=(g0+IDX4C(i,j,k,k,kpoints,bands))->imag;
  *(ldos+IDX2C((n+i-n2)%n,(n+j-n2)%n,n))=trace;
}

//calculate unfolded spectral function
kernel void gpucalcuspf(device const dblcomplex *g0, device const gpufloat *pos,device gpufloat *ldos,constant const qpigpuinfo *qpiinfoblock,uint globalindex [[thread_position_in_grid]]) {
  gpuuint n=qpiinfoblock->n,kpoints=qpiinfoblock->kpoints,bands=qpiinfoblock->bands;
  gpuuint n2=(n>>1); //this fixes the centre being in the wrong position
  gpuuint maxband=qpiinfoblock->maxband;
  gpuuint i=globalindex%n,j=globalindex/n;
  if(j>=n) return;
  gpufloat trace=0.0;
  gpufloat kx=((gpufloat)i-n2)/(gpufloat)kpoints,ky=((gpufloat)j-n2)/(gpufloat)kpoints;
  gpuint ki=((i+n2)%n+kpoints)%kpoints,kj=((j+n2)%n+kpoints)%kpoints;
  for(gpuuint k=0;k<maxband;k++)
    for(gpuuint l=0;l<maxband;l++) {
      //calculate G0(0,R')=G0(0,R')
      dblcomplex prefact=cmpexp((dblcomplex){0.0,2.0*M_PI_F*(kx*(*(pos+IDX2C(0,k,3))-*(pos+IDX2C(0,l,3)))+ky*(*(pos+IDX2C(1,k,3))-*(pos+IDX2C(1,l,3))))});
      gpufloat flpref=(*(pos+IDX2C(2,k,3)))*(*(pos+IDX2C(2,l,3)));
      trace-=(*(g0+IDX4C(ki,kj,l,k,kpoints,bands))*prefact).imag*flpref;
    }
  *(ldos+IDX2C(i,j,n))=trace;
}

//calculate unfolded spectral function
kernel void gpucalcuspfspin(device const dblcomplex *g0, device const gpufloat *pos,device gpufloat *ldos,constant const qpigpuinfo *qpiinfoblock,uint globalindex [[thread_position_in_grid]]) {
  gpuuint n=qpiinfoblock->n,kpoints=qpiinfoblock->kpoints,bands=qpiinfoblock->bands;
  gpuuint n2=(n>>1); //this fixes the centre being in the wrong position
  gpuuint maxband=qpiinfoblock->maxband;
  gpuuint spinbands=maxband>>1;
  gpuuint i=globalindex%n,j=globalindex/n;
  if(j>=n) return;
  gpufloat trace=0.0;
  gpufloat kx=((gpufloat)i-n2)/(gpufloat)kpoints,ky=((gpufloat)j-n2)/(gpufloat)kpoints;
  gpuint ki=((i+n2)%n+kpoints)%kpoints,kj=((j+n2)%n+kpoints)%kpoints;
  for(gpuuint k=0;k<spinbands;k++)
    for(gpuuint l=0;l<spinbands;l++) {
      //calculate G0(0,R')=G0(0,R')
      dblcomplex prefact=cmpexp((dblcomplex){0.0,2.0*M_PI_F*(kx*(*(pos+IDX2C(0,k,3))-*(pos+IDX2C(0,l,3)))+ky*(*(pos+IDX2C(1,k,3))-*(pos+IDX2C(1,l,3))))});
      gpufloat flpref=(*(pos+IDX2C(2,k,3)))*(*(pos+IDX2C(2,l,3)));
      trace-=(*(g0+IDX4C(ki,kj,l,k,kpoints,bands))*prefact).imag*flpref;
      prefact=cmpexp((dblcomplex){0.0,2.0*M_PI_F*(kx*(*(pos+IDX2C(0,spinbands+k,3))-*(pos+IDX2C(0,spinbands+l,3)))+ky*(*(pos+IDX2C(1,spinbands+k,3))-*(pos+IDX2C(1,spinbands+l,3))))});
      flpref=(*(pos+IDX2C(2,spinbands+k,3)))*(*(pos+IDX2C(2,spinbands+l,3)));
      trace-=(*(g0+IDX4C(ki,kj,spinbands+l,spinbands+k,kpoints,bands))*prefact).imag*flpref;
    }
  *(ldos+IDX2C(i,j,n))=trace;
}

//calculate LDOS using Wannier functions - using cached functions
//kernel void gpucalcwannierldos(device const dblcomplex *g0, device gpufloat *ldos,device const dblcomplex *scat,device const gpufloat *wf,device dblcomplex *gblbuffer,device dblcomplex *gblcldos,constant const qpigpuinfo *qpiinfoblock, uint globalindex [[thread_position_in_grid]])
kernel void gpucalcwannierldoslist(device const dblcomplex *g0, device gpufloat *ldos,device const dblcomplex *scat,device const qpigpuflaglist *flags,device const gpuuint *flagoffsets,device const gpuuint *flagentries,device dblcomplex *gblbuffer,device dblcomplex *gblcldos,constant const qpigpuinfo *qpiinfoblock, uint globalindex [[thread_position_in_grid]]) {
  gpuuint n=qpiinfoblock->n,kpoints=qpiinfoblock->kpoints,bands=qpiinfoblock->bands,oversamp=qpiinfoblock->oversamp;
  gpuint window=qpiinfoblock->window,winn=2*window+1;
  gpuuint wanniern=qpiinfoblock->wanniern;
  gpuuint n2=(n>>1);
  device dblcomplex *buffer=gblbuffer+bands*bands*globalindex,
    *cldos=gblcldos+bands*bands*globalindex;
  gpuuint i=globalindex%n,j=globalindex/n;
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
