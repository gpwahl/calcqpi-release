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

#include "tightbinding.h"
#include "wannierfunctions.h"
#include "qpi.h"

#include <gsl/gsl_permutation.h>
#include <gsl/gsl_linalg.h>
#include <gsl/gsl_blas.h>

#ifdef _mpi_version
#include <mpi.h>
#endif

#include <omp.h>

#ifdef _GPU
#include "gpuqpi.h"

#ifdef _CUDA
#include <cuda_runtime_api.h>
#include <cuda.h>
#include "qpicudakernel.h"
#elif _METAL
#include "qpimetal.h"
#elif _HIP
#include <hip/hip_runtime.h>
#include "qpihipkernel.h"
#endif

#endif

#if (GSL_MAJOR_VERSION<2) || (GSL_MINOR_VERSION<7)
void gsl_matrix_complex_conjtrans_memcpy(gsl_matrix_complex *b,gsl_matrix_complex *a) {
  for(size_t i=0;i<b->size1;i++)
    for(size_t j=0;j<b->size2;j++)
      gsl_matrix_complex_set(b,i,j,gsl_complex_conjugate(gsl_matrix_complex_get(a,j,i)));
}
#endif

double matnorm(gsl_matrix_complex *a) {
  double val=0.0;
  for(size_t i=0;i<a->size1;i++)
    for(size_t j=0;j<a->size2;j++)
      val+=gsl_complex_abs2(gsl_matrix_complex_get(a,i,j));
  return sqrt(val);
}

size_t mkthresholdmap(flagarray &flags, vector<wannierfunctions> &wf,size_t window,size_t oversamp,double threshold,size_t maxband,bool spin)
{
  double maxval=0.0;
  size_t orbitals=wf.size(),xs,wins,count=0;
  if(!maxband) maxband=orbitals;
  size_t spinbands=maxband>>1;
  wins=2*window+1;
  xs=oversamp*wins;
  for(size_t l=0;l<orbitals;l++)
    for(size_t i=0;i<xs;i++)
      for(size_t j=0;j<xs;j++) {
	double sqrval=wf[l].getwave_cached(i,j)*wf[l].getwave_cached(i,j);
	if(sqrval>maxval) maxval=sqrval;
      }
  flags.resize(wins);
  for(size_t xn1=0;xn1<wins;xn1++) {
    flags[xn1].resize(wins);
    for(size_t yn1=0;yn1<wins;yn1++) {
      flags[xn1][yn1].resize(wins);
      for(size_t xn2=0;xn2<wins;xn2++) {
	flags[xn1][yn1][xn2].resize(wins);
	for(size_t yn2=0;yn2<wins;yn2++) {
	  flags[xn1][yn1][xn2][yn2].resize(orbitals);
	  for(size_t o1=0;o1<orbitals;o1++) {
	    flags[xn1][yn1][xn2][yn2][o1].resize(orbitals);
	    for(size_t o2=0;o2<orbitals;o2++) {
	      flags[xn1][yn1][xn2][yn2][o1][o2].resize(xs);
	      for(size_t i=0;i<xs;i++)
		flags[xn1][yn1][xn2][yn2][o1][o2][i].assign(xs,false);
	    }
	  }
	}
      }
    }
  }
  if(spin) {
    for(size_t xn1=0;xn1<wins;xn1++) 
      for(size_t yn1=0;yn1<wins;yn1++)
	for(size_t xn2=0;xn2<wins;xn2++) 
	  for(size_t yn2=0;yn2<wins;yn2++)
	    for(size_t o1=0;o1<spinbands;o1++)
	      for(size_t o2=0;o2<spinbands;o2++) 
		for(size_t i=0;i<oversamp;i++)
		  for(size_t j=0;j<oversamp;j++) {
		    size_t x=i+(wins-1)*oversamp,y=j+(wins-1)*oversamp;		  
		    double factor=wf[o1].getwave_cached(x-xn1*oversamp,y-yn1*oversamp)*wf[o2].getwave_cached(x-xn2*oversamp,y-yn2*oversamp);
		    if(fabs(factor)>threshold*maxval) {
		      flags[xn1][yn1][xn2][yn2][o1][o2][i][j]=true;
		      count++;
		    }
		    factor=wf[spinbands+o1].getwave_cached(x-xn1*oversamp,y-yn1*oversamp)*wf[spinbands+o2].getwave_cached(x-xn2*oversamp,y-yn2*oversamp);
		    if(fabs(factor)>threshold*maxval) {
		      flags[xn1][yn1][xn2][yn2][spinbands+o1][spinbands+o2][i][j]=true;
		      count++;
		    }
		  }
  } else {
    for(size_t xn1=0;xn1<wins;xn1++) 
      for(size_t yn1=0;yn1<wins;yn1++)
	for(size_t xn2=0;xn2<wins;xn2++) 
	  for(size_t yn2=0;yn2<wins;yn2++)
	    for(size_t o1=0;o1<maxband;o1++)
	      for(size_t o2=0;o2<maxband;o2++) 
		for(size_t i=0;i<oversamp;i++)
		  for(size_t j=0;j<oversamp;j++) {
		    size_t x=i+(wins-1)*oversamp,y=j+(wins-1)*oversamp;	
		    double factor=wf[o1].getwave_cached(x-xn1*oversamp,y-yn1*oversamp)*wf[o2].getwave_cached(x-xn2*oversamp,y-yn2*oversamp);
		    if(fabs(factor)>threshold*maxval) {
		      flags[xn1][yn1][xn2][yn2][o1][o2][i][j]=true;
		      count++;
		    }
		  }
  }
  return count;
}

size_t mkthresholdlist(flaglist &flags, vector<wannierfunctions> &wf,size_t window,size_t oversamp,double threshold,size_t maxband,bool spin)
{
  double maxval=0.0;
  size_t orbitals=wf.size(),xs,wins,count=0;
  if(!maxband) maxband=orbitals;
  size_t spinbands=maxband>>1;
  wins=2*window+1;
  xs=oversamp*wins;
  for(size_t l=0;l<orbitals;l++)
    for(size_t i=0;i<xs;i++)
      for(size_t j=0;j<xs;j++) {
	double sqrval=wf[l].getwave_cached(i,j)*wf[l].getwave_cached(i,j);
	if(sqrval>maxval) maxval=sqrval;
      }
  if(spin) {
    flags.resize(wins);
    for(size_t xn1=0;xn1<wins;xn1++) {
      flags[xn1].resize(wins);
      for(size_t yn1=0;yn1<wins;yn1++) {
	flags[xn1][yn1].resize(wins);
	for(size_t xn2=0;xn2<wins;xn2++) {
	  flags[xn1][yn1][xn2].resize(wins);
	  for(size_t yn2=0;yn2<wins;yn2++) {
	    for(size_t o1=0;o1<spinbands;o1++)
	      for(size_t o2=0;o2<spinbands;o2++) 
		for(size_t i=0;i<oversamp;i++)
		  for(size_t j=0;j<oversamp;j++) {
		    size_t x=i+(wins-1)*oversamp,y=j+(wins-1)*oversamp;		  
		    double factor=wf[o1].getwave_cached(x-xn1*oversamp,y-yn1*oversamp)*wf[o2].getwave_cached(x-xn2*oversamp,y-yn2*oversamp);
		    if(fabs(factor)>threshold*maxval) {
		      struct flaglistentry fle={i,j,o1,o2,factor};
		      flags[xn1][yn1][xn2][yn2].push_back(fle);
		      count++;
		    }
		    factor=wf[spinbands+o1].getwave_cached(x-xn1*oversamp,y-yn1*oversamp)*wf[spinbands+o2].getwave_cached(x-xn2*oversamp,y-yn2*oversamp);
		    if(fabs(factor)>threshold*maxval) {
		      struct flaglistentry fle={i,j,o1,o2,factor};
		      flags[xn1][yn1][xn2][yn2].push_back(fle);
		      count++;
		    }
		  }
	  }
	}
      }
    }
  } else {
    flags.resize(wins);
    for(size_t xn1=0;xn1<wins;xn1++) {
      flags[xn1].resize(wins);
      for(size_t yn1=0;yn1<wins;yn1++) {
	flags[xn1][yn1].resize(wins);
	for(size_t xn2=0;xn2<wins;xn2++) {
	  flags[xn1][yn1][xn2].resize(wins);
	  for(size_t yn2=0;yn2<wins;yn2++) {
	    for(size_t o1=0;o1<maxband;o1++)
	      for(size_t o2=0;o2<maxband;o2++) 
		for(size_t i=0;i<oversamp;i++)
		  for(size_t j=0;j<oversamp;j++) {
		    size_t x=i+(wins-1)*oversamp,y=j+(wins-1)*oversamp;
		    double factor=wf[o1].getwave_cached(x-xn1*oversamp,y-yn1*oversamp)*wf[o2].getwave_cached(x-xn2*oversamp,y-yn2*oversamp);
		    if(fabs(factor)>threshold*maxval) {
		      struct flaglistentry fle={i,j,o1,o2,factor};
		      flags[xn1][yn1][xn2][yn2].push_back(fle);
		      count++;
		    }
		  }
	  }
	}
      }
    }
  }
  return count;
}

size_t mkthresholdlist(flaglist &flags, vector<wannierfunctions> &wf,size_t window,size_t oversamp,double threshold,vector<int> &spinarr)
{
  double maxval=0.0;
  size_t orbitals=wf.size(),xs,wins,count=0;
  spinarr.resize(orbitals,0);
  wins=2*window+1;
  xs=oversamp*wins;
  for(size_t l=0;l<orbitals;l++)
    for(size_t i=0;i<xs;i++)
      for(size_t j=0;j<xs;j++) {
	double sqrval=wf[l].getwave_cached(i,j)*wf[l].getwave_cached(i,j);
	if(sqrval>maxval) maxval=sqrval;
      }
  flags.resize(wins);
  for(size_t xn1=0;xn1<wins;xn1++) {
    flags[xn1].resize(wins);
    for(size_t yn1=0;yn1<wins;yn1++) {
      flags[xn1][yn1].resize(wins);
      for(size_t xn2=0;xn2<wins;xn2++) {
	flags[xn1][yn1][xn2].resize(wins);
	for(size_t yn2=0;yn2<wins;yn2++) {
	  for(size_t o1=0;o1<orbitals;o1++)
	    if(spinarr[o1])
	      for(size_t o2=0;o2<orbitals;o2++)
		if(spinarr[o1]==spinarr[o2])
		  for(size_t i=0;i<oversamp;i++)
		    for(size_t j=0;j<oversamp;j++) {
		      size_t x=i+(wins-1)*oversamp,y=j+(wins-1)*oversamp;
		      double factor=wf[o1].getwave_cached(x-xn1*oversamp,y-yn1*oversamp)*wf[o2].getwave_cached(x-xn2*oversamp,y-yn2*oversamp);
		      if(fabs(factor)>threshold*maxval) {
			struct flaglistentry fle={i,j,o1,o2,factor};
			flags[xn1][yn1][xn2][yn2].push_back(fle);
			count++;
		      }
		    }
	}
      }
    }
  }
  return count;
}

void  tmatrix::alloc() {
  g0.resize(kpoints);
#pragma omp parallel for
  for(size_t i=0;i<kpoints;i++) {
    g0[i].resize(kpoints);
    for(size_t j=0;j<kpoints;j++)
      g0[i][j]=gsl_matrix_complex_alloc(bands,bands);
  }
}

void tmatrix::alloc_hamiltonian() {
  evect.resize(kpoints);
#pragma omp parallel for
  for(size_t i=0;i<kpoints;i++) {
    evect[i].resize(kpoints);
    for(size_t j=0;j<kpoints;j++)
      evect[i][j]=gsl_matrix_complex_alloc(bands,bands);
  }
  eval.resize(kpoints);
#pragma omp parallel for
  for(size_t i=0;i<kpoints;i++) {
    eval[i].resize(kpoints);
    for(size_t j=0;j<kpoints;j++)
      eval[i][j]=gsl_vector_alloc(bands);
  }
}

void tmatrix::allocldos(size_t wanniern) {
  ldos.resize(wanniern);
#pragma omp parallel for
  for(size_t i=0;i<wanniern;i++)
    ldos[i].assign(wanniern,0.0);
}

#ifdef _GPU
void tmatrix::allocgpumem(size_t wanniern,size_t window,size_t oversamp,size_t maxband,vector<wannierfunctions> &wf) {
  #ifdef _CUDA
  gpuqpi=(GPUQPI *)new CudaQPI(wanniern,kpoints,n,window,oversamp,bands,maxband,spin,wf);
  #elif _METAL
  gpuqpi=(GPUQPI *)new MetalQPI(wanniern,kpoints,n,window,oversamp,bands,maxband,spin,wf);
  #elif _HIP
  gpuqpi=(GPUQPI *)new HipQPI(wanniern,kpoints,n,window,oversamp,bands,maxband,spin,wf);
  #endif
  gpuqpi->printinfo(cout);
}

void tmatrix::allocgpumem(size_t wanniern,size_t window,size_t oversamp,flaglist &flags) {
  #ifdef _CUDA
  gpuqpi=(GPUQPI *)new CudaQPI(wanniern,kpoints,n,window,oversamp,bands,flags);
  #elif _METAL
  gpuqpi=(GPUQPI *)new MetalQPI(wanniern,kpoints,n,window,oversamp,bands,flags);
#elif _HIP
  gpuqpi=(GPUQPI *)new HipQPI(wanniern,kpoints,n,window,oversamp,bands,flags);
#endif
  gpuqpi->printinfo(cout);
}

void tmatrix::allocgpumem(size_t maxbands,vector<vector<double> > &pos,vector<double> &prearr) {
  #ifdef _CUDA
  gpuqpi=(GPUQPI *)new CudaQPI(kpoints,n,bands,maxbands,spin,pos,prearr);
  #elif _METAL
  gpuqpi=(GPUQPI *)new MetalQPI(kpoints,n,bands,maxbands,spin,pos,prearr);
  #elif _HIP
  gpuqpi=(GPUQPI *)new HipQPI(kpoints,n,bands,maxbands,spin,pos,prearr);
  #endif
  gpuqpi->printinfo(cout);
}

void tmatrix::freegpumem() {
  delete gpuqpi;
  gpuqpi=NULL;
}
#endif

void tmatrix::allocgfft(size_t wanniern) {
  if(!gfftdata) {
    gnormconst=wanniern*wanniern;
    gfftsize=wanniern*wanniern;
    gfftdata=new complex<double>[gfftsize];
    mygplan=fftw_plan_dft_2d(wanniern,wanniern,reinterpret_cast<fftw_complex*>(gfftdata),reinterpret_cast<fftw_complex*>(gfftdata),FFTW_FORWARD,0);
  }
}

void tmatrix::alloccontg(size_t wanniern) {
  contg.resize(wanniern);
#pragma omp parallel for
  for(size_t i=0;i<wanniern;i++)
    contg[i].assign(wanniern,GSL_COMPLEX_ZERO);
}

void tmatrix::fftalloc() {
  if(fftdata!=NULL) delete fftdata;
  fftsize=kpoints*kpoints;
  fftdata=new complex<double>[fftsize];
}

void tmatrix::printscattering(ostream &os) {
  if(scat) {
    for(size_t j=0;j<bands;j++) {
      os<<j<<".: ";
      for(size_t i=0;i<bands;i++) {
	os<<"("<<GSL_REAL(gsl_matrix_complex_get(scat,i,j))<<","<<GSL_IMAG(gsl_matrix_complex_get(scat,i,j))<<")    ";
      }
      os<<endl;
    }
  } else {
    for(size_t i=0;i<bands;i++)
      os<<i<<".: "<<vscat[i]<<endl;
  }   
}

#ifdef _mpi_version
#ifndef _GPU
//indexing for 4D array
#define IDX4C(i,j,k,l,ld0,ld1) ((((j)*(ld0))+(i))*(ld1)*(ld1)+(((l)*(ld1))+(k)))
#endif
//indexing for 3D array
#define IDX3CEV(i,j,k,ld0,ld1) ((((j)*(ld0))+(i))*(ld1)+(k))

void tmatrix::calchamiltonian() {
  size_t cut=(kpoints>>1);
  if(world_size==1) {
    alloc_hamiltonian();
#pragma omp parallel
    {
      gsl_eigen_hermv_workspace *w=gsl_eigen_hermv_alloc(bands);
      gsl_matrix_complex *hopping=gsl_matrix_complex_alloc(bands,bands);
      vector<double> kvector(3,0.0);
#pragma omp for
      for(size_t i=0;i<kpoints;i++)
        for(size_t j=0;j<kpoints;j++) {
          double x=(i>cut)?((double)i-kpoints)/kpoints:((double)i/kpoints),
            y=(j>cut)?((double)j-kpoints)/kpoints:((double)j/kpoints); 
          kvector[0]=x;
          kvector[1]=y;
          tb->setmatrix(kvector,hopping);
          gsl_eigen_hermv(hopping, eval[i][j], evect[i][j], w);
        }
      gsl_matrix_complex_free(hopping);
      gsl_eigen_hermv_free(w);
    }
  } else {
    size_t kpointsperprocess=kpoints/world_size;
    if(kpointsperprocess*world_size<kpoints)
      kpointsperprocess++;
    gsl_complex *evectdata=new gsl_complex[bands*bands*kpoints*kpointsperprocess*world_size];
    double *evaldata=new double[bands*kpoints*kpointsperprocess*world_size];
    size_t startkpoint=kpointsperprocess*world_rank,endkpoint=startkpoint+kpointsperprocess;
    if(endkpoint>kpoints) endkpoint=kpoints;
#pragma omp parallel
    {
      gsl_eigen_hermv_workspace *w=gsl_eigen_hermv_alloc(bands);
      gsl_matrix_complex *hopping=gsl_matrix_complex_alloc(bands,bands),
        *levect=gsl_matrix_complex_alloc(bands,bands);
      gsl_vector *leval=gsl_vector_alloc(bands);
      vector<double> kvector(3,0.0);
#pragma omp for
      for(size_t i=0;i<kpoints;i++)
	for(size_t j=startkpoint;j<endkpoint;j++) {
          double x=(i>cut)?((double)i-kpoints)/kpoints:((double)i/kpoints),
            y=(j>cut)?((double)j-kpoints)/kpoints:((double)j/kpoints);
          kvector[0]=x;
          kvector[1]=y;
          tb->setmatrix(kvector,hopping);
          gsl_eigen_hermv(hopping, leval, levect, w);
          for(size_t k=0;k<bands;k++) {
            *(evaldata+IDX3CEV(i,j,k,kpoints,bands))=gsl_vector_get(leval,k);
            for(size_t l=0;l<bands;l++)
              *(evectdata+IDX4C(i,j,k,l,kpoints,bands))=gsl_matrix_complex_get(levect,k,l);
          }
        }
      gsl_matrix_complex_free(hopping);
      gsl_matrix_complex_free(levect);
      gsl_vector_free(leval);
      gsl_eigen_hermv_free(w);
    }
    //MPI_Allgather(evectdata+kpointsperprocess*kpoints*bands*bands*startkpoint,kpoints*bands*bands*kpointsperprocess,MPI_C_DOUBLE_COMPLEX,evectdata,kpoints*bands*bands*kpointsperprocess,MPI_C_DOUBLE_COMPLEX,MPI_COMM_WORLD);
    MPI_Datatype type; //workaround for large data sets
    MPI_Type_contiguous( kpoints*bands*bands, MPI_C_DOUBLE_COMPLEX, &type );
    MPI_Type_commit(&type);
    MPI_Allgather(MPI_IN_PLACE,0,MPI_DATATYPE_NULL,evectdata,kpointsperprocess,type,MPI_COMM_WORLD);
    MPI_Type_free(&type);
    //MPI_Allgather(evaldata+kpointsperprocess*kpoints*bands*startkpoint,kpoints*bands*kpointsperprocess,MPI_DOUBLE,evaldata,kpoints*bands*kpointsperprocess,MPI_DOUBLE,MPI_COMM_WORLD);
    MPI_Allgather(MPI_IN_PLACE,0,MPI_DATATYPE_NULL,evaldata,kpoints*bands*kpointsperprocess,MPI_DOUBLE,MPI_COMM_WORLD);
    alloc_hamiltonian();
#pragma omp parallel for
    for(size_t i=0;i<kpoints;i++)
      for(size_t j=0;j<kpoints;j++)
        for(size_t k=0;k<bands;k++) {
          gsl_vector_set(eval[i][j],k,*(evaldata+IDX3CEV(i,j,k,kpoints,bands)));
          for(size_t l=0;l<bands;l++)
            gsl_matrix_complex_set(evect[i][j],k,l,*(evectdata+IDX4C(i,j,k,l,kpoints,bands)));
        }
    //copy data back to eval, evect
    delete[] evaldata;
    delete[] evectdata;
  }
}
#else
void tmatrix::calchamiltonian() {
  alloc_hamiltonian();
  size_t cut=(kpoints>>1);
#pragma omp parallel
  {
    gsl_eigen_hermv_workspace *w=gsl_eigen_hermv_alloc(bands);
    gsl_matrix_complex *hopping=gsl_matrix_complex_alloc(bands,bands);
    vector<double> kvector(3,0.0);
#pragma omp for
    for(size_t i=0;i<kpoints;i++)
      for(size_t j=0;j<kpoints;j++) {
	double x=(i>cut)?((double)i-kpoints)/kpoints:((double)i/kpoints),
	  y=(j>cut)?((double)j-kpoints)/kpoints:((double)j/kpoints); 
	kvector[0]=x;
	kvector[1]=y;
	tb->setmatrix(kvector,hopping);
	gsl_eigen_hermv(hopping, eval[i][j], evect[i][j], w);
      }
    gsl_matrix_complex_free(hopping);
    gsl_eigen_hermv_free(w);
  }
}
#endif

void tmatrix::setgreensfunction(double omega) {
  if(evect.size()==0)
    calchamiltonian();
  if(!g0.size()) alloc(); 
#pragma omp parallel for
  for(size_t i=0;i<kpoints;i++)
    for(size_t j=0;j<kpoints;j++)
      tb->calcgreensfunction(g0[i][j],evect[i][j],eval[i][j],omega,eta);
}

void tmatrix::calcsurfacegreensfunction(double omega,double epserr,bool surface,tightbind *smodel) {
  size_t cut=(kpoints>>1);
  if(!g0.size()) alloc(); 
#pragma omp parallel
  {
    tightbindingcontext *tbc=tb->getnewcontext(),*tbctemp=tb->getnewcontext();
    vector<double> kvector(3,0.0);
    gsl_matrix_complex *alpha=gsl_matrix_complex_alloc(bands,bands),
      *beta=gsl_matrix_complex_alloc(bands,bands),
      *epsilon_s=gsl_matrix_complex_alloc(bands,bands),
      *epsilon_b=gsl_matrix_complex_alloc(bands,bands),
      *buffer=gsl_matrix_complex_alloc(bands,bands),
      *buffer2=gsl_matrix_complex_alloc(bands,bands),
      *g0b=gsl_matrix_complex_alloc(bands,bands);
#pragma omp for
    for(size_t i=0;i<kpoints;i++)
      for(size_t j=0;j<kpoints;j++) {
	double x=(i>cut)?((double)i-kpoints)/kpoints:((double)i/kpoints),
	  y=(j>cut)?((double)j-kpoints)/kpoints:((double)j/kpoints); 
	kvector[0]=x;
	kvector[1]=y;
	if(smodel)
	  smodel->setmatrix(kvector,tbc->hopping,kzdepinplane);
	else
	  tb->setmatrix(kvector,tbc->hopping,kzdepinplane);
	gsl_matrix_complex_memcpy(epsilon_s,tbc->hopping);
	if(smodel)
	  tb->setmatrix(kvector,tbc->hopping,kzdepinplane);
	gsl_matrix_complex_memcpy(epsilon_b,tbc->hopping);
	tb->setmatrix(kvector,tbctemp->hopping,kzdepoutofplane);
	gsl_matrix_complex_memcpy(alpha,tbctemp->hopping);
	gsl_matrix_complex_conjtrans_memcpy(beta,alpha);
	do {
	  gsl_matrix_complex_memcpy(tbc->hopping,epsilon_b);
	  //tbc->solveeigensystem();
	  tbc->calcgreensfunctioninv(g0b,omega,eta);
	  //epsilon^s_i=epsilon^s_i-1+alpha_i-1(omega-epsilon_i-1)^-1beta_i-1
	  //epsilon^b_i=epsilon^b_i-1+alpha_i-1(omega-epsilon_i-1)^-1beta_i-1
	  gsl_blas_zgemm(CblasNoTrans,CblasNoTrans,GSL_COMPLEX_ONE,g0b,beta,GSL_COMPLEX_ZERO,buffer);
	  gsl_blas_zgemm(CblasNoTrans,CblasNoTrans,GSL_COMPLEX_ONE,alpha,buffer,GSL_COMPLEX_ZERO,buffer2);
	  gsl_matrix_complex_add(epsilon_s,buffer2);
	  gsl_matrix_complex_add(epsilon_b,buffer2);
	  //epsilon^b_i=epsilon^b_i-1+beta_i-1(omega-epsilon_i-1)^-1alpha_i-1
	  gsl_blas_zgemm(CblasNoTrans,CblasNoTrans,GSL_COMPLEX_ONE,g0b,alpha,GSL_COMPLEX_ZERO,buffer);
	  gsl_blas_zgemm(CblasNoTrans,CblasNoTrans,GSL_COMPLEX_ONE,beta,buffer,GSL_COMPLEX_ONE,epsilon_b);
	  //alpha_i=alpha_i-1(omega-epsilon_i-1)^-1alpha_i-1
	  gsl_blas_zgemm(CblasNoTrans,CblasNoTrans,GSL_COMPLEX_ONE,g0b,alpha,GSL_COMPLEX_ZERO,buffer);
	  gsl_blas_zgemm(CblasNoTrans,CblasNoTrans,GSL_COMPLEX_ONE,alpha,buffer,GSL_COMPLEX_ZERO,buffer2);
	  gsl_matrix_complex_memcpy(alpha,buffer2);
	  //beta_i=beta_i-1(omega-epsilon_i-1)^-1beta_i-1
	  gsl_blas_zgemm(CblasNoTrans,CblasNoTrans,GSL_COMPLEX_ONE,g0b,beta,GSL_COMPLEX_ZERO,buffer);
	  gsl_blas_zgemm(CblasNoTrans,CblasNoTrans,GSL_COMPLEX_ONE,beta,buffer,GSL_COMPLEX_ZERO,buffer2);
	  gsl_matrix_complex_memcpy(beta,buffer2);
	} while((matnorm(alpha)>epserr) || (matnorm(beta)>epserr));
	if(surface)
	  gsl_matrix_complex_memcpy(tbc->hopping,epsilon_s);
	else
	  gsl_matrix_complex_memcpy(tbc->hopping,epsilon_b);
	//tbc->solveeigensystem();
	tbc->calcgreensfunctioninv(g0[i][j],omega,eta);
      }
    delete tbc; delete tbctemp; gsl_matrix_complex_free(alpha); gsl_matrix_complex_free(beta);
    gsl_matrix_complex_free(epsilon_s); gsl_matrix_complex_free(epsilon_b);
    gsl_matrix_complex_free(buffer); gsl_matrix_complex_free(buffer2);
    gsl_matrix_complex_free(g0b);
  }
}
 
//calculates G0(0,r,omega)
void tmatrix::calcrealspacegreensfunction() {
  double normconst=kpoints*kpoints; 
  for(size_t k=0;k<bands;k++)
    for(size_t l=0;l<bands;l++) {
#pragma omp parallel for
      for(size_t i=0;i<fftsize;i++)  {
	gsl_complex a=gsl_matrix_complex_get(g0[i%kpoints][i/kpoints],k,l);
	fftdata[i]=complex<double>(GSL_REAL(a),GSL_IMAG(a))/normconst;
      }
      // for(size_t i=0;i<kpoints;i++)
      // 	for(size_t j=0;j<kpoints;j++) {
      // 	gsl_complex a=gsl_matrix_complex_get(g0[i][j],k,l);
      // 	fftdata[i+j*kpoints]=complex<double>(GSL_REAL(a),GSL_IMAG(a))/normconst;
      // }
      fftw_execute(myplan);
#pragma omp parallel for
      for(size_t i=0;i<kpoints;i++)
	for(size_t j=0;j<kpoints;j++)
	  gsl_matrix_complex_set(g0[i][j],k,l,gsl_complex_rect(fftdata[i+j*kpoints].real(),fftdata[i+j*kpoints].imag()));
      //gsl_matrix_complex_set(g0[i][j],k,l,gsl_complex_rect(fftdata[i*kpoints+j].real(),fftdata[i*kpoints+j].imag()));
    }
}

void tmatrix::writeidl(const char *name) {
  idl map(kpoints,kpoints,bands*bands*2,2.0*limit,2.0*limit,1.0,0.0,0.0,-limit,-limit);
  for(size_t k=0;k<bands;k++)
    for(size_t l=0;l<bands;l++)
#pragma omp parallel for
      for(size_t i=0;i<kpoints;i++)
	for(size_t j=0;j<kpoints;j++) {
	  map.set(i,j,2*(k*bands+l),GSL_REAL(gsl_matrix_complex_get(g0[i][j],k,l)));
	  map.set(i,j,2*(k*bands+l)+1,GSL_IMAG(gsl_matrix_complex_get(g0[i][j],k,l)));
	}
  map>>name;
}

void tmatrix::writeldos(const char *name) {
  size_t ldosn=ldos.size();
  idl map(ldosn,ldosn,1,2.0*limit,2.0*limit,1.0,0.0,0.0,-limit,-limit);
#pragma omp parallel for
  for(size_t i=0;i<ldosn;i++)
    for(size_t j=0;j<ldosn;j++) {
      map.set(i,j,0,ldos[i][j]);
    }
  map>>name;
}

void tmatrix::ldos2idl(idl &map,size_t layer) {
#ifdef _GPU
  gpuqpi->retrieveResult(map,layer);
#else
  size_t ldosn=ldos.size(),xs,ys;
  map.dimensions(xs,ys);
  if(xs!=ldosn) {
    ExecuteCPU0 cerr<<"Map and ldos array do not have the same dimensions."<<endl;
    return;
  }
#pragma omp parallel for
  for(size_t i=0;i<ldosn;i++)
    for(size_t j=0;j<ldosn;j++) {
      map.set(i,j,layer,ldos[i][j]);
    }
#endif
}

void tmatrix::spf2idl(idl &map, size_t layer,size_t maxband,bool shift) {
  size_t xs,ys,xshift=0,yshift=0;
  map.dimensions(xs,ys);
  if(!maxband) maxband=bands;
  if((xs!=kpoints)||(ys!=kpoints)) {
    ExecuteCPU0 cerr<<"Map and greens function do not have the same dimensions."<<endl;
    return;
  }
  if(shift) {
    xshift=xs>>1;
    yshift=ys>>1;
  }
#pragma omp parallel for
  for(size_t i=0;i<kpoints;i++)
    for(size_t j=0;j<kpoints;j++) {
      double trace=0.0;
      for(size_t k=0;k<maxband;k++)
	trace-=GSL_IMAG(gsl_matrix_complex_get(g0[i][j],k,k));
      map.set((kpoints+i-xshift)%kpoints,(kpoints+j-yshift)%kpoints,layer,trace);
    }
}

void tmatrix::spf2array(double *map,size_t maxband,bool shift) {
  size_t kshift=0;
  if(shift)
    kshift=kpoints>>1;
  if(!maxband) maxband=bands;
#pragma omp parallel for
  for(size_t i=0;i<kpoints;i++)
    for(size_t j=0;j<kpoints;j++) {
      double trace=0.0;
      for(size_t k=0;k<maxband;k++)
	trace-=GSL_IMAG(gsl_matrix_complex_get(g0[i][j],k,k));
      map[((kpoints+i-kshift)%kpoints)+((kpoints+j-kshift)%kpoints)*kpoints]=trace;
    }
}

void tmatrix::calcspf(size_t maxband,bool shift) {
#ifdef _GPU
  if(!gpuqpi) {
    vector<vector<double> > posarr;
    vector<double> prearr;
    allocgpumem(maxband,posarr,prearr);
  }
  gpuqpi->spf(g0);
  //gpuqpi->retrieveResult(ldos);  
#else
  size_t kshift=0;
  if(n!=kpoints) return;
  allocldos(n);
  if(shift)
    kshift=kpoints>>1;
  if(!maxband) maxband=bands;
#pragma omp parallel for
  for(size_t i=0;i<kpoints;i++)
    for(size_t j=0;j<kpoints;j++) {
      double trace=0.0;
      for(size_t k=0;k<maxband;k++)
	trace-=GSL_IMAG(gsl_matrix_complex_get(g0[i][j],k,k));
      ldos[((kpoints+i-kshift)%kpoints)][((kpoints+j-kshift)%kpoints)]=trace;
    }
#endif
}

void tmatrix::spf2idl(vector<vector<double> > &pos,vector<double> &prearr,idl &map, size_t layer,size_t maxband) {
  size_t xs,ys,xshift=0,yshift=0;
  map.dimensions(xs,ys);
  if(!maxband) maxband=bands;
  xshift=xs>>1;
  yshift=ys>>1;
#pragma omp parallel for
  for(size_t i=0;i<xs;i++)
    for(size_t j=0;j<ys;j++) {
      double kx=((double)i-xshift)/(double)kpoints,
	ky=((double)j-yshift)/(double)kpoints;
      int ki=((i+xshift)%xs+kpoints)%kpoints,kj=((j+yshift)%ys+kpoints)%kpoints;
      double trace=0.0;
      for(size_t k=0;k<maxband;k++)
	for(size_t l=0;l<maxband;l++) {
	  //gsl_complex prefact;
	  //gsl_vector_complex_view evectl=gsl_matrix_complex_column(evect[ki][kj],l);
	  //gsl_vector_complex_view evectk=gsl_matrix_complex_column(evect[ki][kj],k);
	  //gsl_blas_zdotc(&evectl,&evectk,&prefact);
	  gsl_complex prefact=gsl_complex_exp(gsl_complex_rect(0.0,-2.0*M_PI*(kx*(pos[k][0]-pos[l][0])+ky*(pos[k][1]-pos[l][1]))));
	  trace-=prearr[k]*prearr[l]*GSL_IMAG(gsl_complex_mul(gsl_matrix_complex_get(g0[ki][kj],l,k),prefact));
	}
      //map.set((xs+i-xshift)%xs,(ys+j-yshift)%ys,layer,trace);
      map.set(i,j,layer,trace);
    }
}

void tmatrix::spf2array(vector<vector<double> > &pos,vector<double> &prearr,double *map,size_t maxband) {
  size_t xs=n,ys=n,xshift=0,yshift=0;
  if(!maxband) maxband=bands;
  xshift=xs>>1;
  yshift=ys>>1;
#pragma omp parallel for
  for(size_t i=0;i<xs;i++)
    for(size_t j=0;j<ys;j++) {
      double kx=((double)i-xshift)/(double)kpoints,
	ky=((double)j-yshift)/(double)kpoints;
      int ki=((i+xshift)%xs+kpoints)%kpoints,kj=((j+yshift)%ys+kpoints)%kpoints;
      double trace=0.0;
      for(size_t k=0;k<maxband;k++)
	for(size_t l=0;l<maxband;l++) {
	  gsl_complex prefact=gsl_complex_exp(gsl_complex_rect(0.0,2.0*M_PI*(kx*(pos[k][0]-pos[l][0])+ky*(pos[k][1]-pos[l][1]))));
	  trace-=prearr[k]*prearr[l]*GSL_IMAG(gsl_complex_mul(gsl_matrix_complex_get(g0[ki][kj],l,k),prefact));
	  //trace-=GSL_IMAG(gsl_complex_mul(gsl_matrix_complex_get(g0[ki][kj],k,l),prefact));
	}
      map[i+j*xs]=trace;
    }
}

void tmatrix::calcuspf(size_t maxband,vector<vector<double> > &pos,vector<double> &prearr) {
#ifdef _GPU
  if(!gpuqpi)
    allocgpumem(maxband,pos,prearr);
  gpuqpi->uspf(g0);
  //gpuqpi->retrieveResult(ldos);  
#else
  size_t xs=n,ys=n,xshift=0,yshift=0;
  if(!maxband) maxband=bands;
  size_t spinbands=maxband>>1;
  allocldos(n);
  xshift=xs>>1;
  yshift=ys>>1;
#pragma omp parallel for
  for(size_t i=0;i<xs;i++)
    for(size_t j=0;j<ys;j++) {
      double kx=((double)i-xshift)/(double)kpoints,
	ky=((double)j-yshift)/(double)kpoints;
      int ki=((i+xshift)%xs+kpoints)%kpoints,kj=((j+yshift)%ys+kpoints)%kpoints;
      double trace=0.0;
      if(!spin) {
	for(size_t k=0;k<maxband;k++)
	  for(size_t l=0;l<maxband;l++) {
	    gsl_complex prefact=gsl_complex_exp(gsl_complex_rect(0.0,2.0*M_PI*(kx*(pos[k][0]-pos[l][0])+ky*(pos[k][1]-pos[l][1]))));
	    trace-=prearr[k]*prearr[l]*GSL_IMAG(gsl_complex_mul(gsl_matrix_complex_get(g0[ki][kj],l,k),prefact));
	    //trace-=GSL_IMAG(gsl_complex_mul(gsl_matrix_complex_get(g0[ki][kj],k,l),prefact));
	  }
      } else {
	for(size_t k=0;k<spinbands;k++)
	  for(size_t l=0;l<spinbands;l++) {
	    gsl_complex prefact=gsl_complex_exp(gsl_complex_rect(0.0,2.0*M_PI*(kx*(pos[k][0]-pos[l][0])+ky*(pos[k][1]-pos[l][1]))));
	    trace-=prearr[k]*prearr[l]*GSL_IMAG(gsl_complex_mul(gsl_matrix_complex_get(g0[ki][kj],l,k),prefact));
	    prefact=gsl_complex_exp(gsl_complex_rect(0.0,2.0*M_PI*(kx*(pos[spinbands+k][0]-pos[spinbands+l][0])+ky*(pos[spinbands+k][1]-pos[spinbands+l][1]))));
	    trace-=prearr[spinbands+k]*prearr[spinbands+l]*GSL_IMAG(gsl_complex_mul(gsl_matrix_complex_get(g0[ki][kj],spinbands+l,spinbands+k),prefact));
	    //trace-=GSL_IMAG(gsl_complex_mul(gsl_matrix_complex_get(g0[ki][kj],k,l),prefact));
	  }
      }
      ldos[i][j]=trace;
    }
#endif
}

// void tmatrix::spf2array(vector<vector<double> > &pos,double *map,size_t maxband,bool shift) {
//   size_t xs=n,ys=n,xshift=0,yshift=0;
//   size_t cut=(kpoints>>1);
//   if(!maxband) maxband=bands;
//   if(shift) {
//     xshift=xs>>1;
//     yshift=ys>>1;
//   }
// #pragma omp parallel for
//   for(size_t i=0;i<xs;i++)
//     for(size_t j=0;j<ys;j++) {
//       double kx=(i>xshift)?((double)i-xs)/kpoints:((double)i/kpoints),
// 	ky=(j>yshift)?((double)j-ys)/kpoints:((double)j/kpoints);
//       size_t ki=(kpoints+kpoints+i-xshift)%kpoints,kj=(kpoints+kpoints+j-yshift)%kpoints;
//       double trace=0.0;
//       for(size_t k=0;k<maxband;k++) {
// 	//for(size_t l=0;l<maxband;l++) {
// 	gsl_complex prefact=gsl_complex_exp(gsl_complex_rect(0.0,-2.0*M_PI*(kx*(pos[k][0]-pos[0][0])+ky*(pos[k][1]-pos[0][1]))));
// 	trace-=GSL_IMAG(gsl_complex_mul(gsl_matrix_complex_get(g0[ki][kj],k,k),prefact));
//       }
//       //trace-=GSL_IMAG(gsl_complex_mul(gsl_matrix_complex_get(g0[ki][kj],k,l),prefact));
//       map[i+j*xs]=trace;
//     }
// }

void tmatrix::ldos2vector(vector<vector <double> > &map) {
#ifdef _GPU
  gpuqpi->retrieveResult(map);
#else
  size_t ldosn=ldos.size();
  map.resize(ldosn);
#pragma omp parallel
  {
#pragma omp for
    for(size_t i=0;i<ldosn;i++)
      map[i].resize(ldosn);
#pragma omp for
    for(size_t i=0;i<ldosn;i++)
      for(size_t j=0;j<ldosn;j++) 
	map[i][j]=ldos[i][j];
  }
#endif
}

void tmatrix::ldos2array(double *map) {
#ifdef _GPU
  gpuqpi->retrieveResult(map);
#else
  size_t ldosn=ldos.size();
#pragma omp parallel for
  for(size_t i=0;i<ldosn;i++)
    for(size_t j=0;j<ldosn;j++) 
      map[i+j*ldosn]=ldos[i][j];
#endif
}

void tmatrix::setscatteringmatrix()
{
  gsl_matrix_complex *subg=gsl_matrix_complex_alloc(bands,bands),
    *vmatrix=gsl_matrix_complex_calloc(bands,bands);
  if(!scat) scat=gsl_matrix_complex_alloc(bands,bands);
  gsl_matrix_complex_set_identity(scat);
  //set diagonal elements of V to scattering terms
  for(size_t i=0;i<bands;i++)
    gsl_matrix_complex_set(vmatrix,i,i,gsl_complex_rect(vscat[i].real(),vscat[i].imag()));
  //1-V*G_0
  gsl_blas_zgemm(CblasNoTrans,CblasNoTrans,gsl_complex_rect(-1.0,0.0),vmatrix,g0[0][0],gsl_complex_rect(1.0,0.0),scat);
  gsl_permutation *p=gsl_permutation_alloc(bands);
  int signum;
  //invert (1-V*G_0)
  gsl_linalg_complex_LU_decomp(scat,p,&signum);
  gsl_linalg_complex_LU_invert(scat,p,subg);
  gsl_permutation_free(p);
  //T=(1-V*G0)^(-1)*V
  gsl_blas_zgemm(CblasNoTrans,CblasNoTrans,gsl_complex_rect(1.0,0.0),subg,vmatrix,gsl_complex_rect(0.0,0.0),scat);
  gsl_matrix_complex_free(vmatrix);
  gsl_matrix_complex_free(subg);
}

double tmatrix::calcdos(size_t maxband) {
  double result=0.0;
  if(!maxband) maxband=bands;
  for(size_t i=0;i<maxband;i++)
    result-=GSL_IMAG(gsl_matrix_complex_get(g0[0][0],i,i));
  return result;
}

void matrixmultiplication(gsl_matrix_complex *a, gsl_matrix_complex *b, gsl_matrix_complex *c, size_t n)
{
  for(size_t i=0;i<n;i++)
    for(size_t j=0;j<n;j++)
      gsl_matrix_complex_set(c,i,j,GSL_COMPLEX_ZERO);
  for(size_t i=0;i<n;i++)
    for(size_t j=0;j<n;j++)
      for(size_t k=0;k<n;k++) 
	gsl_matrix_complex_set(c,i,j,gsl_complex_add(gsl_matrix_complex_get(c,i,j),gsl_complex_mul(gsl_matrix_complex_get(a,k,j),gsl_matrix_complex_get(b,i,k))));
}

//calculate LDOS using Wannier functions - using cached functions
void tmatrix::calcwannierldos(size_t oversamp,size_t window,vector<wannierfunctions> &wf,size_t maxband) {
  size_t wanniern=oversamp*n;
  if(!maxband) maxband=bands;
  setscatteringmatrix();
#ifdef _GPU
  if(!gpuqpi) allocgpumem(wanniern,window,oversamp,maxband,wf);
  gpuqpi->wannierldos(scat,g0);
  //gpuqpi->retrieveResult(ldos);  
#else
  allocldos(wanniern);
  size_t n2=(n>>1);
  size_t spinbands=maxband>>1;
  //sum over lattice
#pragma omp parallel
  {
    gsl_matrix_complex *buffer=gsl_matrix_complex_alloc(bands,bands),
      *cldos=gsl_matrix_complex_alloc(bands,bands);
#pragma omp for
    for(size_t i=0;i<n;i++)
      for(size_t j=0;j<n;j++) {
	int ipos=(int)i-n2, jpos=(int)j-n2;
	//R-loop over nearest neighbours
	for(int nni1=-window;nni1<=(int)window;nni1++)
	  for(int nnj1=-window;nnj1<=(int)window;nnj1++) {
	    //calculate TG0(0,R)
	    gsl_blas_zgemm(CblasNoTrans,CblasNoTrans,gsl_complex_rect(1.0,0.0),scat,g0[(kpoints+kpoints-(ipos+nni1))%kpoints][(kpoints+kpoints-(jpos+nnj1))%kpoints],gsl_complex_rect(0.0,0.0),buffer);
	    //matrixmultiplication(scat,g0[(kpoints+kpoints-(ipos+nni1))%kpoints][(kpoints+kpoints-(jpos+nnj1))%kpoints],buffer,bands);
	    //R'-loop over nearest neighbours
	    for(int nni2=-window;nni2<=(int)window;nni2++)
	      for(int nnj2=-window;nnj2<=(int)window;nnj2++) {
		//calculate G0(R',0)TG0(0,R)    
		  gsl_blas_zgemm(CblasNoTrans,CblasNoTrans,gsl_complex_rect(1.0,0.0),g0[(kpoints+ipos+nni2)%kpoints][(kpoints+jpos+nnj2)%kpoints],buffer,gsl_complex_rect(0.0,0.0),cldos);
		  //matrixmultiplication(g0[(kpoints+ipos+nni2)%kpoints][(kpoints+jpos+nnj2)%kpoints],buffer,cldos,bands);
		//sum over sub-unit cell sampling
		for(size_t ii=0;ii<oversamp;ii++)
		  for(size_t jj=0;jj<oversamp;jj++) {
		    size_t x=ii+window*oversamp,y=jj+window*oversamp;
		    //size_t x=ipos*oversamp+ii,y=jpos*oversamp+jj;
		    if(!spin) {
		      for(size_t o1=0;o1<maxband;o1++) {
			if(!wf[o1].iszero())
			  for(size_t o2=0;o2<maxband;o2++)
			    if(!wf[o2].iszero()) {
			      double factor=wf[o1].getwave_cached(x-nni1*oversamp,y-nnj1*oversamp)*wf[o2].getwave_cached(x-nni2*oversamp,y-nnj2*oversamp);
			      ldos[(i*oversamp+ii)%wanniern][(j*oversamp+jj)%wanniern]-=GSL_IMAG(gsl_matrix_complex_get(cldos,o2,o1))*factor;
			      ldos[(i*oversamp+ii)%wanniern][(j*oversamp+jj)%wanniern]-=GSL_IMAG(gsl_matrix_complex_get(g0[(kpoints-nni1+nni2)%kpoints][(kpoints-nnj1+nnj2)%kpoints],o2,o1))*factor;
			    }
		      }
		    } else {
		      for(size_t o1=0;o1<spinbands;o1++) {
			if(!wf[o1].iszero())
			  for(size_t o2=0;o2<spinbands;o2++)
			    if(!wf[o2].iszero()) {
			      double factor=wf[o1].getwave_cached(x-nni1*oversamp,y-nnj1*oversamp)*wf[o2].getwave_cached(x-nni2*oversamp,y-nnj2*oversamp);
			      ldos[(i*oversamp+ii)%wanniern][(j*oversamp+jj)%wanniern]-=GSL_IMAG(gsl_matrix_complex_get(cldos,o2,o1))*factor;
			      ldos[(i*oversamp+ii)%wanniern][(j*oversamp+jj)%wanniern]-=GSL_IMAG(gsl_matrix_complex_get(g0[(kpoints-nni1+nni2)%kpoints][(kpoints-nnj1+nnj2)%kpoints],o2,o1))*factor;
			    }
			if(!wf[spinbands+o1].iszero())
			  for(size_t o2=0;o2<spinbands;o2++)
			    if(!wf[spinbands+o2].iszero()) {
			      double factor=wf[spinbands+o1].getwave_cached(x-nni1*oversamp,y-nnj1*oversamp)*wf[spinbands+o2].getwave_cached(x-nni2*oversamp,y-nnj2*oversamp);
			      ldos[(i*oversamp+ii)%wanniern][(j*oversamp+jj)%wanniern]-=GSL_IMAG(gsl_matrix_complex_get(cldos,spinbands+o2,spinbands+o1))*factor;
			      ldos[(i*oversamp+ii)%wanniern][(j*oversamp+jj)%wanniern]-=GSL_IMAG(gsl_matrix_complex_get(g0[(kpoints-nni1+nni2)%kpoints][(kpoints-nnj1+nnj2)%kpoints],spinbands+o2,spinbands+o1))*factor;
			    }
		      }
		    }
		  }
	      }
	  }
      }
    gsl_matrix_complex_free(buffer);
    gsl_matrix_complex_free(cldos);
  }
#endif
}

//calculate Josephson tunneling using Wannier functions - using cached functions
void tmatrix::calcwannierjosephson(size_t oversamp,size_t window,vector<wannierfunctions> &wf, gsl_complex tip) {
  size_t wanniern=oversamp*n;
  size_t maxband=(bands>>1); //only for superconducting TB models
  setscatteringmatrix();
#ifdef _GPU
  if(!gpuqpi) allocgpumem(wanniern,window,oversamp,maxband,wf);
  gpuqpi->wannierjosephson(scat,g0,tip);
  //gpuqpi->retrieveResult(ldos);  
#else
  allocldos(wanniern);
  size_t n2=(n>>1);
  size_t spinbands=maxband>>1;
  //sum over lattice
#pragma omp parallel
  {
    gsl_matrix_complex *buffer=gsl_matrix_complex_alloc(bands,bands),
      *cldos=gsl_matrix_complex_alloc(bands,bands);
#pragma omp for
    for(size_t i=0;i<n;i++)
      for(size_t j=0;j<n;j++) {
	int ipos=(int)i-n2, jpos=(int)j-n2;
	//R-loop over nearest neighbours
	for(int nni1=-window;nni1<=(int)window;nni1++)
	  for(int nnj1=-window;nnj1<=(int)window;nnj1++) {
	    //calculate TG0(0,R)
	    gsl_blas_zgemm(CblasNoTrans,CblasNoTrans,gsl_complex_rect(1.0,0.0),scat,g0[(kpoints+kpoints-(ipos+nni1))%kpoints][(kpoints+kpoints-(jpos+nnj1))%kpoints],gsl_complex_rect(0.0,0.0),buffer);
	    //matrixmultiplication(scat,g0[(kpoints+kpoints-(ipos+nni1))%kpoints][(kpoints+kpoints-(jpos+nnj1))%kpoints],buffer,bands);
	    //R'-loop over nearest neighbours
	    for(int nni2=-window;nni2<=(int)window;nni2++)
	      for(int nnj2=-window;nnj2<=(int)window;nnj2++) {
		//calculate G0(R',0)TG0(0,R)    
		  gsl_blas_zgemm(CblasNoTrans,CblasNoTrans,gsl_complex_rect(1.0,0.0),g0[(kpoints+ipos+nni2)%kpoints][(kpoints+jpos+nnj2)%kpoints],buffer,gsl_complex_rect(0.0,0.0),cldos);
		  //matrixmultiplication(g0[(kpoints+ipos+nni2)%kpoints][(kpoints+jpos+nnj2)%kpoints],buffer,cldos,bands);
		//sum over sub-unit cell sampling
		for(size_t ii=0;ii<oversamp;ii++)
		  for(size_t jj=0;jj<oversamp;jj++) {
		    size_t x=ii+window*oversamp,y=jj+window*oversamp;
		    //size_t x=ipos*oversamp+ii,y=jpos*oversamp+jj;
		    if(!spin) {
		      for(size_t o1=0;o1<maxband;o1++) {
			if(!wf[o1].iszero())
			  for(size_t o2=0;o2<maxband;o2++)
			    if(!wf[o2].iszero()) {
			      double factor=wf[o1].getwave_cached(x-nni1*oversamp,y-nnj1*oversamp)*wf[o2].getwave_cached(x-nni2*oversamp,y-nnj2*oversamp);
			      gsl_complex val=gsl_complex_add(gsl_matrix_complex_get(cldos,o2,o1+maxband),
							      gsl_matrix_complex_get(g0[(kpoints-nni1+nni2)%kpoints][(kpoints-nnj1+nnj2)%kpoints],o2,o1+maxband));
			      val=gsl_complex_mul(val,tip);
			      ldos[(i*oversamp+ii)%wanniern][(j*oversamp+jj)%wanniern]-=GSL_IMAG(val)*factor;
			    }
		      }
		    } else {
		      for(size_t o1=0;o1<spinbands;o1++) {
			if(!wf[o1].iszero())
			  for(size_t o2=0;o2<spinbands;o2++)
			    if(!wf[o2].iszero()) {
			      double factor=wf[o1].getwave_cached(x-nni1*oversamp,y-nnj1*oversamp)*wf[o2].getwave_cached(x-nni2*oversamp,y-nnj2*oversamp);
			      gsl_complex val=gsl_complex_add(gsl_matrix_complex_get(cldos,o2,o1+maxband),
							      gsl_matrix_complex_get(g0[(kpoints-nni1+nni2)%kpoints][(kpoints-nnj1+nnj2)%kpoints],o2,o1+maxband));
			      val=gsl_complex_mul(val,tip);
			      ldos[(i*oversamp+ii)%wanniern][(j*oversamp+jj)%wanniern]-=GSL_IMAG(val)*factor;
			    }
			if(!wf[spinbands+o1].iszero())
			  for(size_t o2=0;o2<spinbands;o2++)
			    if(!wf[spinbands+o2].iszero()) {
			      double factor=wf[spinbands+o1].getwave_cached(x-nni1*oversamp,y-nnj1*oversamp)*wf[spinbands+o2].getwave_cached(x-nni2*oversamp,y-nnj2*oversamp);
			      gsl_complex val=gsl_complex_add(gsl_matrix_complex_get(cldos,spinbands+o2,spinbands+o1+maxband),
							      gsl_matrix_complex_get(g0[(kpoints-nni1+nni2)%kpoints][(kpoints-nnj1+nnj2)%kpoints],spinbands+o2,spinbands+o1+maxband));
			      val=gsl_complex_mul(val,tip);
			      ldos[(i*oversamp+ii)%wanniern][(j*oversamp+jj)%wanniern]-=GSL_IMAG(val)*factor;
			    }
		      }
		    }
		  }
	      }
	  }
      }
    gsl_matrix_complex_free(buffer);
    gsl_matrix_complex_free(cldos);
  }
#endif
}

//calculate LDOS using Wannier functions - using cached functions
void tmatrix::calcwannierldos(size_t oversamp,size_t window,vector<wannierfunctions> &wf,flagarray &flags) {
  size_t wanniern=oversamp*n;
  setscatteringmatrix();
  allocldos(wanniern);
  size_t n2=(n>>1);
  //sum over lattice
#pragma omp parallel
  {
    gsl_matrix_complex *buffer=gsl_matrix_complex_alloc(bands,bands),
      *cldos=gsl_matrix_complex_alloc(bands,bands);
#pragma omp for
    for(size_t i=0;i<n;i++)
      for(size_t j=0;j<n;j++) {
	int ipos=(int)i-n2, jpos=(int)j-n2;
	//R-loop over nearest neighbours
	for(int nni1=-window;nni1<=(int)window;nni1++)
	  for(int nnj1=-window;nnj1<=(int)window;nnj1++) {
	    //calculate TG0(0,R)
	    gsl_blas_zgemm(CblasNoTrans,CblasNoTrans,gsl_complex_rect(1.0,0.0),scat,g0[(kpoints+kpoints-(ipos+nni1))%kpoints][(kpoints+kpoints-(jpos+nnj1))%kpoints],gsl_complex_rect(0.0,0.0),buffer);
	    //matrixmultiplication(scat,g0[(kpoints+kpoints-(ipos+nni1))%kpoints][(kpoints+kpoints-(jpos+nnj1))%kpoints],buffer,bands);
	    //R'-loop over nearest neighbours
	    for(int nni2=-window;nni2<=(int)window;nni2++)
	      for(int nnj2=-window;nnj2<=(int)window;nnj2++) {
		//calculate G0(R',0)TG0(0,R)    
		gsl_blas_zgemm(CblasNoTrans,CblasNoTrans,gsl_complex_rect(1.0,0.0),g0[(kpoints+ipos+nni2)%kpoints][(kpoints+jpos+nnj2)%kpoints],buffer,gsl_complex_rect(0.0,0.0),cldos);
		//matrixmultiplication(g0[(kpoints+ipos+nni2)%kpoints][(kpoints+jpos+nnj2)%kpoints],buffer,cldos,bands);
		//sum over sub-unit cell sampling
		for(size_t ii=0;ii<oversamp;ii++)
		  for(size_t jj=0;jj<oversamp;jj++) {
		    size_t x=ii+window*oversamp,y=jj+window*oversamp;
		    //size_t x=ipos*oversamp+ii,y=jpos*oversamp+jj;
		    for(size_t o1=0;o1<bands;o1++) {
		      for(size_t o2=0;o2<bands;o2++)
			if(flags[window+nni1][window+nnj1][window+nni2][window+nnj2][o1][o2][ii][jj]) {
			  double factor=wf[o1].getwave_cached(x-nni1*oversamp,y-nnj1*oversamp)*wf[o2].getwave_cached(x-nni2*oversamp,y-nnj2*oversamp);
			  ldos[(i*oversamp+ii)%wanniern][(j*oversamp+jj)%wanniern]-=GSL_IMAG(gsl_matrix_complex_get(cldos,o2,o1))*factor;
			  ldos[(i*oversamp+ii)%wanniern][(j*oversamp+jj)%wanniern]-=GSL_IMAG(gsl_matrix_complex_get(g0[(kpoints-nni1+nni2)%kpoints][(kpoints-nnj1+nnj2)%kpoints],o2,o1))*factor;
			}
		    }
		  }
	      }
	  }
      }
    gsl_matrix_complex_free(buffer);
    gsl_matrix_complex_free(cldos);
  }
}

//calculate LDOS using Wannier functions - using cached functions
void tmatrix::calcwannierldos(size_t oversamp,size_t window,flaglist &flags) {
  size_t wanniern=oversamp*n;
  setscatteringmatrix();
#ifdef _GPU
  if(!gpuqpi) allocgpumem(wanniern,window,oversamp,flags);
  gpuqpi->wannierldoslist(scat,g0);
  //gpuqpi->retrieveResult(ldos);  
#else
  allocldos(wanniern);
  size_t n2=(n>>1);
  //sum over lattice
#pragma omp parallel
  {
    gsl_matrix_complex *buffer=gsl_matrix_complex_alloc(bands,bands),
      *cldos=gsl_matrix_complex_alloc(bands,bands);
#pragma omp for
    for(size_t i=0;i<n;i++)
      for(size_t j=0;j<n;j++) {
	int ipos=(int)i-n2, jpos=(int)j-n2;
	//R-loop over nearest neighbours
	for(int nni1=-window;nni1<=(int)window;nni1++)
	  for(int nnj1=-window;nnj1<=(int)window;nnj1++) {
	    //calculate TG0(0,R)
	    gsl_blas_zgemm(CblasNoTrans,CblasNoTrans,gsl_complex_rect(1.0,0.0),scat,g0[(kpoints+kpoints-(ipos+nni1))%kpoints][(kpoints+kpoints-(jpos+nnj1))%kpoints],gsl_complex_rect(0.0,0.0),buffer);
	    //matrixmultiplication(scat,g0[(kpoints+kpoints-(ipos+nni1))%kpoints][(kpoints+kpoints-(jpos+nnj1))%kpoints],buffer,bands);
	    //R'-loop over nearest neighbours
	    for(int nni2=-window;nni2<=(int)window;nni2++)
	      for(int nnj2=-window;nnj2<=(int)window;nnj2++) {
		//calculate G0(R',0)TG0(0,R)    
		gsl_blas_zgemm(CblasNoTrans,CblasNoTrans,gsl_complex_rect(1.0,0.0),g0[(kpoints+ipos+nni2)%kpoints][(kpoints+jpos+nnj2)%kpoints],buffer,gsl_complex_rect(0.0,0.0),cldos);
		//matrixmultiplication(g0[(kpoints+ipos+nni2)%kpoints][(kpoints+jpos+nnj2)%kpoints],buffer,cldos,bands);
		//sum over sub-unit cell sampling
		size_t listlen=flags[nni1+window][nnj1+window][nni2+window][nnj2+window].size();
		for(size_t ind=0;ind<listlen;ind++) {
		  size_t ii=flags[nni1+window][nnj1+window][nni2+window][nnj2+window][ind].i,
		    jj=flags[nni1+window][nnj1+window][nni2+window][nnj2+window][ind].j,
		    o1=flags[nni1+window][nnj1+window][nni2+window][nnj2+window][ind].o1,
		    o2=flags[nni1+window][nnj1+window][nni2+window][nnj2+window][ind].o2;
		  double factor=flags[nni1+window][nnj1+window][nni2+window][nnj2+window][ind].factor;
		  ldos[(i*oversamp+ii)%wanniern][(j*oversamp+jj)%wanniern]-=GSL_IMAG(gsl_matrix_complex_get(cldos,o2,o1))*factor;
		  ldos[(i*oversamp+ii)%wanniern][(j*oversamp+jj)%wanniern]-=GSL_IMAG(gsl_matrix_complex_get(g0[(kpoints-nni1+nni2)%kpoints][(kpoints-nnj1+nnj2)%kpoints],o2,o1))*factor;
		}
	      }
	  }
      }
    gsl_matrix_complex_free(buffer);
    gsl_matrix_complex_free(cldos);
  }
  #endif
}

double tmatrix::calcwannierldospoint_int(size_t xp,size_t yp,size_t oversamp,size_t window,vector<wannierfunctions> &wf,size_t maxband) {
  gsl_matrix_complex *buffer=gsl_matrix_complex_alloc(bands,bands),
    *cldos=gsl_matrix_complex_calloc(bands,bands);
  size_t n2=(n>>1);
  double result=0.0;
  //size_t x=ii+window*oversamp,y=jj+window*oversamp;
  //sum over lattice
  //int ipos=(int)i-n2, jpos=(int)j-n2;
  int ipos=xp/oversamp-n2,jpos=yp/oversamp-n2;
  size_t x=(int)xp-(ipos+n2)*oversamp+window*oversamp, y=(int)yp-(jpos+n2)*oversamp+window*oversamp;
  //size_t i=(n+ipos)%n,j=(n+jpos)%n;
  if(!maxband) maxband=bands;
  size_t spinbands=maxband>>1;
  //R-loop over nearest neighbours
  for(int nni1=-window;nni1<=(int)window;nni1++)
    for(int nnj1=-window;nnj1<=(int)window;nnj1++) {
      //calculate TG0(0,R)
      gsl_blas_zgemm(CblasNoTrans,CblasNoTrans,gsl_complex_rect(1.0,0.0),scat,g0[(kpoints+kpoints-(ipos+nni1))%kpoints][(kpoints+kpoints-(jpos+nnj1))%kpoints],gsl_complex_rect(0.0,0.0),buffer);
      //R'-loop over nearest neighbours
      for(int nni2=-window;nni2<=(int)window;nni2++)
	for(int nnj2=-window;nnj2<=(int)window;nnj2++) {
	  //calculate G0(R',0)TG0(0,R)    
	  gsl_blas_zgemm(CblasNoTrans,CblasNoTrans,gsl_complex_rect(1.0,0.0),g0[(kpoints+ipos+nni2)%kpoints][(kpoints+jpos+nnj2)%kpoints],buffer,gsl_complex_rect(0.0,0.0),cldos);	      
	  //size_t x=ipos*oversamp+ii,y=jpos*oversamp+jj;
	  if(!spin) {
	    for(size_t o1=0;o1<maxband;o1++) {
	      if(!wf[o1].iszero())
		for(size_t o2=0;o2<maxband;o2++)
		  if(!wf[o2].iszero()) {
		    double factor=wf[o1].getwave_cached(x-nni1*oversamp,y-nnj1*oversamp)*wf[o2].getwave_cached(x-nni2*oversamp,y-nnj2*oversamp);
		    result-=GSL_IMAG(gsl_matrix_complex_get(cldos,o2,o1))*factor;
		    result-=GSL_IMAG(gsl_matrix_complex_get(g0[(kpoints-nni1+nni2)%kpoints][(kpoints-nnj1+nnj2)%kpoints],o2,o1))*factor;
		  }
	    }
	  } else {
	    for(size_t o1=0;o1<spinbands;o1++) {
	      if(!wf[o1].iszero())
		for(size_t o2=0;o2<spinbands;o2++)
		  if(!wf[o2].iszero()) {
		    double factor=wf[o1].getwave_cached(x-nni1*oversamp,y-nnj1*oversamp)*wf[o2].getwave_cached(x-nni2*oversamp,y-nnj2*oversamp);
		    result-=GSL_IMAG(gsl_matrix_complex_get(cldos,o2,o1))*factor;
		    result-=GSL_IMAG(gsl_matrix_complex_get(g0[(kpoints-nni1+nni2)%kpoints][(kpoints-nnj1+nnj2)%kpoints],o2,o1))*factor;
		  }
	      if(!wf[o1+spinbands].iszero())
		for(size_t o2=0;o2<spinbands;o2++)
		  if(!wf[o2+spinbands].iszero()) {
		    double factor=wf[spinbands+o1].getwave_cached(x-nni1*oversamp,y-nnj1*oversamp)*wf[spinbands+o2].getwave_cached(x-nni2*oversamp,y-nnj2*oversamp);
		    result-=GSL_IMAG(gsl_matrix_complex_get(cldos,spinbands+o2,spinbands+o1))*factor;
		    result-=GSL_IMAG(gsl_matrix_complex_get(g0[(kpoints-nni1+nni2)%kpoints][(kpoints-nnj1+nnj2)%kpoints],spinbands+o2,spinbands+o1))*factor;  
		  }
	    }
	  }
	}
    }
  gsl_matrix_complex_free(buffer);
  gsl_matrix_complex_free(cldos);
  return result;
}

double tmatrix::calcwannierldospoint(size_t xp,size_t yp,size_t oversamp,size_t window,vector<wannierfunctions> &wf,size_t maxband) {
  setscatteringmatrix();
  return calcwannierldospoint_int(xp,yp,oversamp,window,wf,maxband);
}

