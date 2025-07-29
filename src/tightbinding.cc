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

#include "tightbinding.h"
#include "idl.h"

#include "mpidefs.h"

#include <iostream>
#include <iomanip>
#include <fstream>
#include <math.h>
#include <algorithm> 

#include <gsl/gsl_math.h>
#include <gsl/gsl_complex_math.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_eigen.h>
#include <gsl/gsl_permutation.h>
#include <gsl/gsl_linalg.h>
#include <gsl/gsl_blas.h>

#ifdef _mpi_version
#include <mpi.h>
#endif

#include <omp.h>

#include <vector>
#include <string>
#include <string.h>


using namespace std;

const double kb=8.61734e-05;

double fermi(double x,double t)
{
  double kt=kb*t;
  return 0.5*(1.0-tanh(x/kt/2.0));
}

gsl_complex kvector(vector<double> &k,vector<int> &nnvector) {
    gsl_complex x=gsl_complex_rect(1.0,0.0);
    for(int i=0;i<3;i++) x=gsl_complex_mul(x,gsl_complex_exp(gsl_complex_rect(0.0,2.0*M_PI*k[i]*(double)nnvector[i])));
    return x;
}

gsl_complex kvector(vector<double> &k,vector<double> &nnvector) {
    gsl_complex x=gsl_complex_rect(1.0,0.0);
    for(int i=0;i<3;i++) x=gsl_complex_mul(x,gsl_complex_exp(gsl_complex_rect(0.0,2.0*M_PI*k[i]*nnvector[i])));
    return x;
}

bool operator!(tightbinding &tb) {return tb.error;}


void tightbind::readfile(istream &is) {
  string buf;
  size_t lineno=0;
  if(is) {
    getline(is,buf); lineno++;
    date=buf.substr(0,buf.find_first_of("\n\r"));
    getline(is,buf); lineno++; istringstream(buf)>>n;
    scmaxbands=(scmodel)?(n>>1):n;
    getline(is,buf); lineno++; istringstream(buf)>>neighbourno;
    //read degeneracies
    degeneracies.resize(neighbourno);
    size_t i=0;
    size_t pos=0,npos;
    //ExecuteCPU0 cout<<"Reading degeneracies ..."<<endl;
    do {
      if(!pos) {
	getline(is,buf); lineno++;
	pos=0;
	pos=buf.find_first_of("0123456789",pos);
      }
      npos=buf.find_first_of(" \t",pos);
      size_t len;
      if(npos!=string::npos)
	len=npos-pos;
      else len=buf.length()-pos;
      npos=buf.find_first_of("0123456789",npos);
      istringstream iss(buf.substr(pos,len));
      iss>>degeneracies[i];
      if(iss.fail()) {
	ExecuteCPU0 cerr<<"Error in input file in line "<<lineno<<": "<<buf<<endl;
	error=true;
	return;
      }
      if(npos!=string::npos)
	pos=npos;
      else pos=0;
      i++;
    } while(i<neighbourno);
    ExecuteCPU0 {
      cerr<<"Loading tb model from "<<date<<" with "<<n<<" orbitals and "<<neighbourno<<" hopping terms."<<endl;
    }
    size_t k=0;
    int ox,oy,oz;
    do {
      getline(is,buf); lineno++;
      if(buf.size()>0) {
	istringstream iss(buf);
	int x,y,z;
	size_t u,v; //x,y,z of NN vector, u, v orbitals
	double hr,hc;
	iss>>x>>y>>z>>u>>v>>hr>>hc;
	if(k) {
	  if((x!=ox) || (y!=oy) || (z!=oz))
	    {
	      k++;
	      ox=x; oy=y; oz=z;
	    }
	} else { ox=x; oy=y; oz=z; k++; }
	if(iss.fail()) {
	  ExecuteCPU0 cerr<<"Error in input file in line "<<lineno<<": "<<buf<<endl;
	  error=true;
	  continue;
	}
	ExecuteCPU0 if(verbose) cerr<<"Loading hopping for ("<<x<<","<<y<<","<<z<<"), H_{"<<u<<","<<v<<"}="<<hr<<"+i"<<hc<<endl;
	hr=hr/(double)degeneracies[k-1];
	hc=hc/(double)degeneracies[k-1];
	if((u>0) && (v>0))
	  hoppings.push_back(hoppingterm(x,y,z,u-1,v-1,gsl_complex_rect(hr,hc),((u-1)>=scmaxbands)&&((v-1)>=scmaxbands)));
	else {
	  ExecuteCPU0 cerr<<"Invalid hopping for ("<<x<<","<<y<<","<<z<<"), H_{"<<u<<","<<v<<"}="<<hr<<"+i"<<hc<<" because of invalid orbital index - exiting..."<<endl;
	  exit(1);
	}
      }
    } while(!is.eof());
    ExecuteCPU0 cerr<<"Reading hoppings completed - read "<<lineno<<" lines and found "<<k<<" hopping terms."<<endl;
  } else {
    ExecuteCPU0 cerr<<"Cannot read or open input file."<<endl;
    error=true;
  }
}

tightbind::tightbind(const string &name, double ifermi, bool scmodel):verbose(false),efermi(ifermi),scmodel(scmodel) {
  error=false; ctbc=NULL;
  ifstream is(name.c_str());
  if(!is)
    error=true;
  else {
    readfile(is);
    ctbc=new tightbindingcontext(n);
  }
}

tightbind::tightbind(istream &is,double ifermi,bool scmodel):verbose(false),efermi(ifermi),scmodel(scmodel) {
  error=false;
  ctbc=NULL;
  if(!is)
    error=true;
  else {
    readfile(is);
    ctbc=new tightbindingcontext(n);
  }
}

void tightbind::debugout(ostream &os)
{
  for(size_t i=0;i<n;i++) {
    for(size_t j=0;j<n;j++) {
      gsl_complex c=gsl_matrix_complex_get(ctbc->hopping,i,j);
      os<<std::fixed<<std::setprecision(3)<<GSL_REAL(c)<<"+i"<<GSL_IMAG(c)<<" ";
    }
    os<<endl;
  }
}

void tightbind::setmatrix(vector<double> &k,gsl_matrix_complex *lhopping,kzdep kzdependence) {
  gsl_matrix_complex_set_zero(lhopping);
  if(kzdependence!=kzdepoutofplane)
    for(size_t i=0;i<n;i++) {
      if(i<scmaxbands)
	gsl_matrix_complex_set(lhopping,i,i,gsl_complex_rect(-efermi,0.0));
      else
	gsl_matrix_complex_set(lhopping,i,i,gsl_complex_rect(efermi,0.0));
    }
  for(size_t i=0;i<hoppings.size();i++) {
    switch(kzdependence) {
    case kzdepall:
      hoppings[i].addhop(lhopping,k);
      break;
    case kzdepinplane:
      if(hoppings[i].rz==0.0)
	hoppings[i].addhop(lhopping,k);
      break;
    case kzdepoutofplane:
      if(hoppings[i].rz<0.0)
	hoppings[i].addhop(lhopping,k);
      break;
    }
  }
}
 
void tightbind::setmatrix(vector<double> &k,gsl_matrix_complex *lhopping) {
  gsl_matrix_complex_set_zero(lhopping);
  for(size_t i=0;i<n;i++)
    if(i<scmaxbands)
      gsl_matrix_complex_set(lhopping,i,i,gsl_complex_rect(-efermi,0.0));
    else
      gsl_matrix_complex_set(lhopping,i,i,gsl_complex_rect(efermi,0.0));
  for(size_t i=0;i<hoppings.size();i++)
    hoppings[i].addhop(lhopping,k);
}

void tightbind::setmatrix(vector<double> &k) {
  setmatrix(k,ctbc->hopping);
}

void tightbind::setmatrixd(vector<double> &k,gsl_matrix_complex *lhopping,size_t dim)
{
  gsl_matrix_complex_set_zero(lhopping);
  for(size_t i=0;i<hoppings.size();i++)
    hoppings[i].addhopd(lhopping,k,dim);
}

void tightbind::inthoppings() {
    gsl_matrix_complex_set_zero(ctbc->hopping);  
    for(size_t i=0;i<hoppings.size();i++)
	hoppings[i].addabshop(ctbc->hopping);
}

double tightbind::getmaxhopping() {
  double x=0.0;
  for(size_t i=0;i<hoppings.size();i++) {
	double y=abs(hoppings[i].t);
	if(y>x) x=y;
    }
  return x;
}

void tightbind::showresults(ostream &o,tightbindingcontext *tbc)
{
    for(size_t i=0;i<n;i++)
	o<<gsl_vector_get(tbc->eval,i)<<" ";
    o<<endl;
}

void tightbind::showresults(ostream &o)
{
  showresults(o,ctbc);
}

void tightbind::calcenergies(vector<double> &k,tightbindingcontext *tbc,bool sort)
{
  this->setmatrix(k,tbc->hopping);
  tbc->solveeigensystem();
  if(sort) gsl_eigen_hermv_sort(tbc->eval, tbc->evec, GSL_EIGEN_SORT_VAL_ASC);
}

void tightbind::calcenergies(vector<double> &k,bool sort)
{
  calcenergies(k,ctbc,sort);
}

void tightbind::calcgreensfunction(gsl_matrix_complex *g0,vector<double> &k,double omega, double eta,tightbindingcontext *tbc)
{
  calcenergies(k,tbc);
  tbc->calcgreensfunction(g0,omega,eta);
}

void tightbind::calcgreensfunction(gsl_matrix_complex *g0,vector<double> &k,double omega, double eta)
{
  calcgreensfunction(g0,k,omega,eta,ctbc);
}

void tightbind::calcgreensfunction(gsl_matrix_complex *g0,gsl_matrix_complex *evec,gsl_vector *eval,double omega, double eta) {
  gsl_matrix_complex_set_zero(g0);
  for(size_t i=0;i<n;i++) {
    gsl_complex factor=gsl_complex_inverse(gsl_complex_rect(omega-gsl_vector_get(eval,i),eta));
    gsl_vector_complex_view evect=gsl_matrix_complex_column(evec,i);
    gsl_blas_zgerc(factor,(gsl_vector_complex *)&evect,(gsl_vector_complex *)&evect,g0);
  } 
}

void tightbind::calckpointlist(vector<vector<double> > &kvects, vector<double> &energies,vector<vector<double> > &oweights,vector<vector<double> > &inkvects,bool sort)
{
  size_t nkpts=inkvects.size();
  kvects.resize(nkpts*getbands());
  energies.resize(nkpts*getbands());
  oweights.resize(nkpts*getbands());
#pragma omp parallel
  {
    tightbindingcontext *tbc=this->getnewcontext();
#pragma omp for
    for(size_t j=0;j<nkpts;j++) {
      for(size_t l=0;l<getbands();l++) {
      	kvects[j*getbands()+l].resize(3);
      	oweights[j*getbands()+l].resize(getbands());
      }
      for(size_t i=0;i<3;i++) {
	for(size_t l=0;l<getbands();l++)
	  kvects[j*getbands()+l][i]=inkvects[j][i];
      }
      calcenergies(inkvects[j],tbc,sort);
      for(size_t i=0;i<getbands();i++) {
	energies[j*getbands()+i]=gsl_vector_get(tbc->eval,i);
	for(size_t l=0;l<getbands();l++)
	  oweights[j*getbands()+i][l]=gsl_complex_abs2(gsl_matrix_complex_get(tbc->evec,l,i));
      }
    }
    delete tbc;
  }
}

void tightbind::calckpointlist(vector<vector<double> > &kvects, vector<double> &energies,vector<vector<gsl_complex> > &oweights,vector<vector<double> > &inkvects,bool sort)
{
  size_t nkpts=inkvects.size();
  kvects.resize(nkpts*getbands());
  energies.resize(nkpts*getbands());
  oweights.resize(nkpts*getbands());
#pragma omp parallel
  {
    tightbindingcontext *tbc=this->getnewcontext();
#pragma omp for
    for(size_t j=0;j<nkpts;j++) {
      for(size_t l=0;l<getbands();l++) {
      	kvects[j*getbands()+l].resize(3);
      	oweights[j*getbands()+l].resize(getbands());
      }
      for(size_t i=0;i<3;i++) {
	for(size_t l=0;l<getbands();l++)
	  kvects[j*getbands()+l][i]=inkvects[j][i];
      }
      calcenergies(inkvects[j],tbc,sort);
      for(size_t i=0;i<getbands();i++) {
	energies[j*getbands()+i]=gsl_vector_get(tbc->eval,i);
	for(size_t l=0;l<getbands();l++)
	  oweights[j*getbands()+i][l]=gsl_matrix_complex_get(tbc->evec,l,i);
      }
    }
    delete tbc;
  }
}

void tightbind::calckpointlist(vector<vector<double> > &kvects, vector<double> &delta,vector<double> &berryphase,double sign)
{
  size_t nkpts=kvects.size();
  delta.resize(nkpts);
  berryphase.resize(nkpts,0.0);
#pragma omp parallel
  {
    tightbindingcontext *tbc=this->getnewcontext();
    gsl_matrix_complex *hdot=gsl_matrix_complex_alloc(n,n);
    gsl_vector_complex *nikx=gsl_vector_complex_alloc(n),
      *niky=gsl_vector_complex_alloc(n),
      *res=gsl_vector_complex_alloc(n);
#pragma omp for
    for(size_t j=0;j<nkpts;j++) {
      size_t ind=0;
      double energy;
      calcenergies(kvects[j],tbc);
      energy=gsl_vector_get(tbc->eval,0);
      for(size_t i=1;i<getbands();i++) {
	double nenergy=gsl_vector_get(tbc->eval,i);
	if(abs(nenergy)<abs(energy) && ((sign*nenergy<0.0) || isnan(sign))) {
	  ind=i;
	  energy=nenergy;
	}
      }
      gsl_complex val;
      //d/dkx
      setmatrixd(kvects[j],hdot,0);
      gsl_vector_complex_set_zero(nikx);
      for(size_t i=0;i<n;i++)
	if(gsl_vector_get(tbc->eval,i)!=energy) {
	  gsl_vector_complex_view evi=gsl_matrix_complex_column(tbc->evec,i),
	    evindex=gsl_matrix_complex_column(tbc->evec,ind);
	  gsl_blas_zgemv(CblasNoTrans,GSL_COMPLEX_ONE,hdot,&evi.vector,GSL_COMPLEX_ZERO,res);
	  gsl_blas_zdotc(&evindex.vector,res,&val);
	  gsl_complex_mul_real(val,1.0/(energy-gsl_vector_get(tbc->eval,i)));
	  gsl_vector_complex_memcpy(res,&evi.vector);
	  gsl_vector_complex_scale(res,val);
	  gsl_vector_complex_add(nikx,res);
	}
      //d/dky
      setmatrixd(kvects[j],hdot,1);
      gsl_vector_complex_set_zero(niky);
      for(size_t i=0;i<n;i++)
	if(gsl_vector_get(tbc->eval,i)!=energy) {
	  gsl_vector_complex_view evi=gsl_matrix_complex_column(tbc->evec,i),
	    evindex=gsl_matrix_complex_column(tbc->evec,ind);
	  gsl_blas_zgemv(CblasNoTrans,GSL_COMPLEX_ONE,hdot,&evi.vector,GSL_COMPLEX_ZERO,res);
	  gsl_blas_zdotc(&evindex.vector,res,&val);
	  gsl_complex_mul_real(val,1.0/(energy-gsl_vector_get(tbc->eval,i)));
	  gsl_vector_complex_memcpy(res,&evi.vector);
	  gsl_vector_complex_scale(res,val);
	  gsl_vector_complex_add(niky,res);
	}
      gsl_blas_zdotc(nikx,niky,&val);
      berryphase[j]+=(-2.0*GSL_IMAG(val));
      delta[j]=energy;
    }
    delete tbc;
    gsl_matrix_complex_free(hdot);
    gsl_vector_complex_free(nikx);
    gsl_vector_complex_free(niky);
    gsl_vector_complex_free(res);
  }
}

void tightbind::calckpointlistop(vector<vector<double> > &kvects, gsl_matrix_complex *op,vector<complex<double> > &result,bool spin,double sign)
{
  const double tolerance=1.0e-10;
  size_t nkpts=kvects.size();
  result.resize(nkpts,complex<double>(0.0,0.0));
#pragma omp parallel
  {
    tightbindingcontext *tbc=this->getnewcontext();
    vector<double> kvector(3,0.0);
    gsl_vector_complex *buf=gsl_vector_complex_alloc(n),*ref=gsl_vector_complex_alloc(n);
    gsl_vector_complex_set_zero(ref);
    size_t n4=(spin)?(n>>2):(n>>1);
    for(size_t i=0;i<n4;i++)
      gsl_vector_complex_set(ref,i,GSL_COMPLEX_ONE);
#pragma omp for
    for(size_t i=0;i<nkpts;i++) {
      vector <size_t> indices;
      calcenergies(kvects[i],tbc);
      double energy=gsl_vector_get(tbc->eval,0);
      size_t ind=0;
      indices.resize(1,0);
      for(size_t j=1;j<getbands();j++) {
	double nenergy=gsl_vector_get(tbc->eval,j);
	if(abs(nenergy)<abs(energy) && ((sign*nenergy<0.0) || isnan(sign))) {
	  ind=j;
	  energy=nenergy;
	  indices.resize(1);
	  indices[0]=ind;
	} else if(abs(nenergy-energy)<tolerance)
	  indices.push_back(j);
      }
      if(indices.size()>1) {
	double maxval=0.0;
	for(size_t i=0;i<indices.size();i++) {
	  gsl_complex sprod;
	  gsl_vector_complex_view evect=gsl_matrix_complex_column(tbc->evec,indices[i]);
	  gsl_blas_zdotc((gsl_vector_complex *)&evect, ref, &sprod);
	  if(gsl_complex_abs(sprod)>maxval) {
	    maxval=gsl_complex_abs(sprod);
	    ind=indices[i];
	  }
	}
      }
      gsl_complex val;//,norm;
      gsl_vector_complex_view evect=gsl_matrix_complex_column(tbc->evec,ind);
      gsl_blas_zgemv(CblasNoTrans, GSL_COMPLEX_ONE, op, (gsl_vector_complex *)&evect, GSL_COMPLEX_ZERO, buf);
      gsl_blas_zdotc((gsl_vector_complex *)&evect, buf, &val);
      result[i]=complex<double>(GSL_REAL(val),GSL_IMAG(val));
    }
    delete tbc;
    gsl_vector_complex_free(buf); gsl_vector_complex_free(ref);
  }
}

void tightbind::calckpointlistop(vector<vector<double> > &kvects, vector<vector<complex<double> > > &result,gsl_matrix_complex **op,double temperature,size_t ops)
{
  size_t nkpts=kvects.size();
  result.resize(nkpts);
  for(size_t i=0;i<nkpts;i++)
    result[i].resize(ops,complex<double>(0.0,0.0));
#pragma omp parallel
  {
    tightbindingcontext *tbc=this->getnewcontext();
    vector<double> kvector(3,0.0);
    gsl_vector_complex *buf=gsl_vector_complex_alloc(n);
#pragma omp for
    for(size_t i=0;i<nkpts;i++) {
      calcenergies(kvects[i],tbc);
      for(size_t l=0;l<n;l++) {
	double ev=gsl_vector_get(tbc->eval,l);
	gsl_complex cres;
	gsl_vector_complex_view evect=gsl_matrix_complex_column(tbc->evec,l);
	double fact=fermi(ev,temperature);
	for(size_t o=0;o<ops;o++) {
	  gsl_blas_zgemv(CblasNoTrans, GSL_COMPLEX_ONE, op[o], (gsl_vector_complex *)&evect, GSL_COMPLEX_ZERO, buf);
	  gsl_blas_zdotc((gsl_vector_complex *)&evect, buf, &cres);
	  result[i][o]=fact*complex<double>(GSL_REAL(cres),GSL_IMAG(cres));
	}
      }
    }
    delete tbc;
    gsl_vector_complex_free(buf);
  }
}

void tightbind::calcdir(vector<vector<double> > &kvects, vector<double> &energies,vector<vector<double> > &oweights,vector<double> &k1,vector<double> &dir,int n,size_t maxbands)
{
  if(!maxbands) maxbands=getbands();
  kvects.resize(n*maxbands);
  energies.resize(n*maxbands);
  oweights.resize(n*getbands());
#pragma omp parallel
  {
    tightbindingcontext *tbc=this->getnewcontext();
    vector<double> k(3,0.0);
#pragma omp for
    for(int j=0;j<n;j++) {
      for(size_t l=0;l<maxbands;l++) {
	kvects[j*maxbands+l].resize(3);
	oweights[j*maxbands+l].resize(getbands());
      }
      for(size_t i=0;i<3;i++) {
	k[i]=k1[i]+dir[i]*j;
	for(size_t l=0;l<maxbands;l++)
	  kvects[j*maxbands+l][i]=k[i];
      }
      calcenergies(k,tbc);
      for(size_t i=0;i<maxbands;i++) {
	energies[j*maxbands+i]=gsl_vector_get(tbc->eval,i);
	for(size_t l=0;l<getbands();l++)
	  oweights[j*maxbands+i][l]=gsl_complex_abs2(gsl_matrix_complex_get(tbc->evec,l,i));
      }
    }
    delete tbc;
  }
}

void tightbind::calcline(vector<vector<double> > &kvects, vector<double> &energies,vector<vector<double> > &oweights,vector<double> &k1,vector<double> &k2,int n,size_t maxbands)
{
    vector<double> dir(3,0.0);
    for(int i=0;i<3;i++) dir[i]=(k2[i]-k1[i])/(n-1);
    calcdir(kvects,energies,oweights,k1,dir,n,maxbands);
}

void tightbind::calcdir(vector<vector<double> > &kvects, vector<double> &energies,vector<vector<complex<double> > > &oweights,vector<double> &k1,vector<double> &dir,int n,size_t maxbands)
{
  if(!maxbands) maxbands=getbands();
  kvects.resize(n*maxbands);
  energies.resize(n*maxbands);
  oweights.resize(n*getbands());
#pragma omp parallel
  {
    tightbindingcontext *tbc=this->getnewcontext();
    vector<double> k(3,0.0);
#pragma omp for
    for(int j=0;j<n;j++) {
      for(size_t l=0;l<maxbands;l++) {
	kvects[j*maxbands+l].resize(3);
	oweights[j*maxbands+l].resize(getbands());
      }
      for(size_t i=0;i<3;i++) {
	k[i]=k1[i]+dir[i]*j;
	for(size_t l=0;l<maxbands;l++)
	  kvects[j*maxbands+l][i]=k[i];
      }
      calcenergies(k,tbc);
      for(size_t i=0;i<maxbands;i++) {
	energies[j*maxbands+i]=gsl_vector_get(tbc->eval,i);
	for(size_t l=0;l<getbands();l++) {
	  gsl_complex val=gsl_matrix_complex_get(tbc->evec,l,i);
	    oweights[j*maxbands+i][l]=complex<double>(GSL_REAL(val),GSL_IMAG(val));
	}
      }
    }
    delete tbc;
  }
}

void tightbind::calcline(vector<vector<double> > &kvects, vector<double> &energies,vector<vector<complex<double> > > &oweights,vector<double> &k1,vector<double> &k2,int n,size_t maxbands)
{
    vector<double> dir(3,0.0);
    for(int i=0;i<3;i++) dir[i]=(k2[i]-k1[i])/(n-1);
    calcdir(kvects,energies,oweights,k1,dir,n,maxbands);
}

#ifdef _mpi_version
#ifndef _GPU
//indexing for 4D array
#define IDX4C(i,j,k,l,ld0,ld1) ((((j)*(ld0))+(i))*(ld1)*(ld1)+(((l)*(ld1))+(k)))
#endif
//indexing for 3D array
#define IDX3CEV(i,j,k,ld0,ld1) ((((j)*(ld0))+(i))*(ld1)+(k))
//indexing for 2D array
#define IDX2C(i,j,ld) (((j)*(ld))+(i))

void tightbind::calcdos3d(vector<vector<double> > &hist,double a,double b,size_t bins,bool orb_res,double kstep)
{
  double weight=kstep*kstep*kstep;
  size_t columns=(orb_res)?n:1;
  ExecuteCPU0 {
    hist.resize(bins);
    for(size_t i=0;i<bins;i++) {
      hist[i].resize(columns+1,0.0);
      hist[i][0]=(b-a)/bins*((double)i+0.5)+a;
    }
  }
  size_t res=(size_t)(1.0/kstep);
  if(world_size==1) {
#pragma omp parallel
    {
      tightbindingcontext *tbc=this->getnewcontext();
      vector<double> kvector(3,0.0);
#pragma omp for
      for(size_t i=0;i<res;i++)
	for(size_t j=0;j<res;j++)
	  for(size_t k=0;k<res;k++) {
	    kvector[0]=-0.5+(double)i*kstep;
	    kvector[1]=-0.5+(double)j*kstep;
	    kvector[2]=-0.5+(double)k*kstep;
	    calcenergies(kvector,tbc);
	    for(size_t l=0;l<n;l++) {
	      double ev=gsl_vector_get(tbc->eval,l);
	      int ind=(int)((ev-a)*bins/(b-a));
	      if((size_t)ind<bins) {
		if(orb_res) hist[ind][l+1]+=weight;
		else hist[ind][1]+=weight;
	      }
	    }
	  }
      delete tbc;
    }
  } else {
    size_t kpointsperprocess=res/world_size;
    if(kpointsperprocess*world_size<res)
      kpointsperprocess++;
    size_t startkpoint=kpointsperprocess*world_rank,endkpoint=startkpoint+kpointsperprocess;
    if(endkpoint>res) endkpoint=res;
    size_t localarraysize=bins*columns;
    size_t arraysize=localarraysize*((world_rank)?1:world_size);
    double *histdata=new double[arraysize];
#pragma omp parallel
    {
      tightbindingcontext *tbc=this->getnewcontext();
      vector<double> kvector(3,0.0);
#pragma omp for
      for(size_t i=0;i<localarraysize;i++)
	*(histdata+i)=0.0;
#pragma omp for
      for(size_t i=0;i<res;i++)
	for(size_t j=startkpoint;j<endkpoint;j++)
	  for(size_t k=0;k<res;k++) {
	    kvector[0]=-0.5+(double)i*kstep;
	    kvector[1]=-0.5+(double)j*kstep;
	    kvector[2]=-0.5+(double)k*kstep;
	    calcenergies(kvector,tbc);
	    for(size_t l=0;l<n;l++) {
	      double ev=gsl_vector_get(tbc->eval,l);
	      int ind=(int)((ev-a)*bins/(b-a));
	      if((ind>=0) && ((size_t)ind<bins)) {
		if(orb_res) *(histdata+IDX2C(ind,l,columns))+=weight;
		else *(histdata+IDX2C(ind,0,columns))+=weight;
	      }
	    }
	  }
      delete tbc;
    }
    MPI_Gather(histdata,localarraysize,MPI_DOUBLE,histdata,localarraysize,MPI_DOUBLE,0,MPI_COMM_WORLD);
    ExecuteCPU0 {
#pragma omp parallel for
      for(size_t j=0;j<bins;j++)
	for(size_t i=0;i<(size_t)world_size;i++)
	  for(size_t k=0;k<columns;k++)
	    hist[j][k+1]+=*(histdata+i*localarraysize+IDX2C(j,k,columns));
      }	
    delete[] histdata;
  }
}
#else
void tightbind::calcdos3d(vector<vector<double> > &hist,double a,double b,size_t bins,bool orb_res,double kstep)
{
  double weight=kstep*kstep*kstep;
  size_t columns=(orb_res)?n:1;
  hist.resize(bins);
  for(size_t i=0;i<bins;i++) {
    hist[i].resize(columns+1,0.0);
    hist[i][0]=(b-a)/bins*((double)i+0.5)+a;
  }
  size_t res=(size_t)(1.0/kstep);
  #pragma omp parallel
  {
    tightbindingcontext *tbc=this->getnewcontext();
    vector<double> kvector(3,0.0);
    #pragma omp for
    for(size_t i=0;i<res;i++)
      for(size_t j=0;j<res;j++)
	for(size_t k=0;k<res;k++) {
	  kvector[0]=-0.5+(double)i*kstep;
	  kvector[1]=-0.5+(double)j*kstep;
	  kvector[2]=-0.5+(double)k*kstep;
	  calcenergies(kvector,tbc);
	  for(size_t l=0;l<n;l++) {
	    double ev=gsl_vector_get(tbc->eval,l);
	    int ind=(int)((ev-a)*bins/(b-a));
	    if((ind>=0) && ((size_t)ind<bins)) {
	      if(orb_res) hist[ind][l+1]+=weight;
	      else hist[ind][1]+=weight;
	    }
	  }
	}
    delete tbc;
  }
}
#endif

#ifdef _mpi_version
void tightbind::calcdos2d(vector<vector<double> > &hist,double a,double b,size_t bins,bool orb_res,double kstep)
{
  double weight=kstep*kstep;
  size_t columns=(orb_res)?n:1;
  ExecuteCPU0 {
    hist.resize(bins);
    for(size_t i=0;i<bins;i++) {
      hist[i].resize(columns+1,0.0);
      hist[i][0]=(b-a)/bins*((double)i+0.5)+a;
    }
  }
  size_t res=(size_t)(1.0/kstep);
  if(world_size==1) {
#pragma omp parallel
    {
      tightbindingcontext *tbc=this->getnewcontext();
      vector<double> kvector(3,0.0);
#pragma omp for
      for(size_t i=0;i<res;i++)
	for(size_t j=0;j<res;j++) {
	  kvector[0]=-0.5+(double)i*kstep;
	  kvector[1]=-0.5+(double)j*kstep;
	  calcenergies(kvector,tbc);
	  for(size_t l=0;l<n;l++) {
	    double ev=gsl_vector_get(tbc->eval,l);
	    int ind=(int)((ev-a)*bins/(b-a));
	    if((size_t)ind<bins) {
	      if(orb_res) hist[ind][l+1]+=weight;
	      else hist[ind][1]+=weight;
	    }
	  }
	}
      delete tbc;
    }
  } else {
    size_t kpointsperprocess=res/world_size;
    if(kpointsperprocess*world_size<res)
      kpointsperprocess++;
    size_t startkpoint=kpointsperprocess*world_rank,endkpoint=startkpoint+kpointsperprocess;
    if(endkpoint>res) endkpoint=res;
    size_t localarraysize=bins*columns;
    size_t arraysize=localarraysize*((world_rank)?1:world_size);
    double *histdata=new double[arraysize];
#pragma omp parallel
    {
      tightbindingcontext *tbc=this->getnewcontext();
      vector<double> kvector(3,0.0);
#pragma omp for
      for(size_t i=0;i<localarraysize;i++)
	*(histdata+i)=0.0;
#pragma omp for
      for(size_t i=0;i<res;i++)
	for(size_t j=startkpoint;j<endkpoint;j++) {
	  kvector[0]=-0.5+(double)i*kstep;
	  kvector[1]=-0.5+(double)j*kstep;
	  calcenergies(kvector,tbc);
	  for(size_t l=0;l<n;l++) {
	    double ev=gsl_vector_get(tbc->eval,l);
	    int ind=(int)((ev-a)*bins/(b-a));
	    if((ind>=0) && ((size_t)ind<bins)) {
	      if(orb_res) *(histdata+IDX2C(ind,l,columns))+=weight;
	      else *(histdata+IDX2C(ind,0,columns))+=weight;
	    }
	  }
	}
      delete tbc;
    }
    MPI_Gather(histdata,localarraysize,MPI_DOUBLE,histdata,localarraysize,MPI_DOUBLE,0,MPI_COMM_WORLD);
    ExecuteCPU0 {
#pragma omp parallel for
      for(size_t j=0;j<bins;j++)
	for(size_t i=0;i<(size_t)world_size;i++)
	  for(size_t k=0;k<columns;k++)
	    hist[j][k+1]+=*(histdata+i*localarraysize+IDX2C(j,k,columns));
      }	
    delete[] histdata;
  }
}
#else
void tightbind::calcdos2d(vector<vector<double> > &hist,double a,double b,size_t bins,bool orb_res,double kstep)
{
  double weight=kstep*kstep;
  size_t columns=(orb_res)?n:1;
  hist.resize(bins);
  for(size_t i=0;i<bins;i++) {
    hist[i].resize(columns+1,0.0);
    hist[i][0]=(b-a)/bins*((double)i+0.5)+a;
  }
  size_t res=(size_t)(1.0/kstep);
#pragma omp parallel
  {
    tightbindingcontext *tbc=this->getnewcontext();
    vector<double> kvector(3,0.0);
#pragma omp for
    for(size_t i=0;i<res;i++)
      for(size_t j=0;j<res;j++) {
	kvector[0]=-0.5+(double)i*kstep;
	kvector[1]=-0.5+(double)j*kstep;
	calcenergies(kvector,tbc);
	for(size_t l=0;l<n;l++) {
	  double ev=gsl_vector_get(tbc->eval,l);
	  int ind=(int)((ev-a)*bins/(b-a));
	  if((ind>=0) && ((size_t)ind<bins)) {
	    if(orb_res) hist[ind][l+1]+=weight;
	    else hist[ind][1]+=weight;
	  }
	}
      }
    delete tbc;
  }
}
#endif

void tightbind::projop2d(vector<complex<double> > &result,gsl_matrix_complex **op,double temperature,double kstep,size_t ops)
{
  double weight=kstep*kstep;
  result.resize(ops,complex<double>(0.0,0.0));
  size_t res=(size_t)(1.0/kstep);
#pragma omp parallel
  {
    tightbindingcontext *tbc=this->getnewcontext();
    vector<double> kvector(3,0.0);
    gsl_vector_complex *buf=gsl_vector_complex_alloc(n);
#pragma omp for
    for(size_t i=0;i<res;i++)
      for(size_t j=0;j<res;j++) {
	kvector[0]=-0.5+(double)i*kstep;
	kvector[1]=-0.5+(double)j*kstep;
	calcenergies(kvector,tbc);
	for(size_t l=0;l<n;l++) {
	  double ev=gsl_vector_get(tbc->eval,l);
	  gsl_complex cres;
	  gsl_vector_complex_view evect=gsl_matrix_complex_column(tbc->evec,l);
	  double fact=weight*fermi(ev,temperature);
	  for(size_t o=0;o<ops;o++) {
	    gsl_blas_zgemv(CblasNoTrans, GSL_COMPLEX_ONE, op[o], (gsl_vector_complex *)&evect, GSL_COMPLEX_ZERO, buf);
	    gsl_blas_zdotc((gsl_vector_complex *)&evect, buf, &cres);
	    result[o]+=fact*complex<double>(GSL_REAL(cres),GSL_IMAG(cres));
	  }
	}
      }
    delete tbc;
    gsl_vector_complex_free(buf);
  }
}

void tightbind::occupations3d(vector<vector<double> > &hist,double a,double b,size_t bins,bool orb_res,double kstep)
{
  double weight=kstep*kstep*kstep;
  size_t columns=(orb_res)?n:1;
    hist.resize(bins);
    for(size_t i=0;i<bins;i++) {
      hist[i].resize(columns+1,0);
      hist[i][0]=a+(b-a)/bins*((double)i+0.5);
    }
    size_t res=(size_t)(1.0/kstep);
#pragma omp parallel
    {
      tightbindingcontext *tbc=this->getnewcontext();
      vector<double> kvector(3,0.0);
#pragma omp for
      for(size_t i=0;i<res;i++)
	for(size_t j=0;j<res;j++)
	  for(size_t k=0;k<res;k++) {
	    kvector[0]=-0.5+(double)i*kstep;
	    kvector[1]=-0.5+(double)j*kstep;
	    kvector[2]=-0.5+(double)k*kstep;
	    calcenergies(kvector,tbc);
	    for(size_t l=0;l<n;l++) {
	      double ev=gsl_vector_get(tbc->eval,l);
	      for(size_t m=0;m<bins;m++)
		if(ev<hist[m][0]) {
		  if(orb_res)
		    hist[m][l+1]+=weight;
		  else
		    hist[m][1]+=weight;
		}
	    }
	  }
      }
}

#ifdef _mpi_version
void tightbind::occupations2d(vector<vector<double> > &hist,double a,double b,size_t bins,bool orb_res,double kstep)
{
  double weight=kstep*kstep;
  size_t columns=(orb_res)?n:1;
  ExecuteCPU0 {
    hist.resize(bins);
    for(size_t i=0;i<bins;i++) {
      hist[i].resize(columns+1,0.0);
      hist[i][0]=(b-a)/bins*((double)i+0.5)+a;
    }
  }
  size_t res=(size_t)(1.0/kstep);
  if(world_size==1) {
#pragma omp parallel
    {
      tightbindingcontext *tbc=this->getnewcontext();
      vector<double> kvector(3,0.0);
#pragma omp for
      for(size_t i=0;i<res;i++)
	for(size_t j=0;j<res;j++) {
	  kvector[0]=-0.5+(double)i*kstep;
	  kvector[1]=-0.5+(double)j*kstep;
	  calcenergies(kvector,tbc);
	  for(size_t l=0;l<n;l++) {
	    double ev=gsl_vector_get(tbc->eval,l);
	    for(size_t m=0;m<bins;m++)
	      if(ev<hist[m][0])  {
		if(orb_res) hist[m][l+1]+=weight;
		else hist[m][1]+=weight;
	      }
	  }
	}
      delete tbc;
    }
  } else {
    size_t kpointsperprocess=res/world_size;
    if(kpointsperprocess*world_size<res)
      kpointsperprocess++;
    size_t startkpoint=kpointsperprocess*world_rank,endkpoint=startkpoint+kpointsperprocess;
    if(endkpoint>res) endkpoint=res;
    size_t localarraysize=bins*columns;
    size_t arraysize=localarraysize*((world_rank)?1:world_size);
    double *histdata=new double[arraysize];
#pragma omp parallel
    {
      tightbindingcontext *tbc=this->getnewcontext();
      vector<double> kvector(3,0.0);
#pragma omp for
      for(size_t i=0;i<localarraysize;i++)
	*(histdata+i)=0.0;
#pragma omp for
      for(size_t i=0;i<res;i++)
	for(size_t j=startkpoint;j<endkpoint;j++) {
	  kvector[0]=-0.5+(double)i*kstep;
	  kvector[1]=-0.5+(double)j*kstep;
	  calcenergies(kvector,tbc);
	  for(size_t l=0;l<n;l++) {
	    double ev=gsl_vector_get(tbc->eval,l);
	    for(size_t m=0;m<bins;m++)
	      if(ev<hist[m][0])  {
	      if(orb_res) *(histdata+IDX2C(m,l,columns))+=weight;
	      else *(histdata+IDX2C(m,0,columns))+=weight;
	    }
	  }
	}
      delete tbc;
    }
    MPI_Gather(histdata,localarraysize,MPI_DOUBLE,histdata,localarraysize,MPI_DOUBLE,0,MPI_COMM_WORLD);
    ExecuteCPU0 {
#pragma omp parallel for
      for(size_t j=0;j<bins;j++)
	for(size_t i=0;i<(size_t)world_size;i++)
	  for(size_t k=0;k<columns;k++)
	    hist[j][k+1]+=*(histdata+i*localarraysize+IDX2C(j,k,columns));
      }	
    delete[] histdata;
  }
}
#else
void tightbind::occupations2d(vector<vector<double> > &hist,double a,double b,size_t bins,bool orb_res,double kstep)
{
  double weight=kstep*kstep;
  size_t columns=(orb_res)?n:1;
    hist.resize(bins);
    for(size_t i=0;i<bins;i++) {
      hist[i].resize(columns+1,0.0);
      hist[i][0]=a+(b-a)/bins*((double)i+0.5);
    }
    // size_t total=0;
    size_t res=(size_t)(1.0/kstep);
#pragma omp parallel
    {
      tightbindingcontext *tbc=this->getnewcontext();
      vector<double> kvector(3,0.0);
#pragma omp for
      for(size_t i=0;i<res;i++)
	for(size_t j=0;j<res;j++) {
	  kvector[0]=-0.5+(double)i*kstep;
	  kvector[1]=-0.5+(double)j*kstep;
	  //	  total++;
	  calcenergies(kvector,tbc);
	  for(size_t l=0;l<n;l++) {
	    double ev=gsl_vector_get(tbc->eval,l);
	    for(size_t m=0;m<bins;m++)
	      if(ev<hist[m][0])  {
		if(orb_res)
		  hist[m][l+1]+=weight;
		else
		  hist[m][1]+=weight;
	      }
	  }
	}
    }
}
#endif

double tightbind::getoccupation2d(double fermi,double kstep)
{
  double weight=kstep*kstep;
  vector<double> occupied(n,0.0);
  //size_t total=0;
  size_t res=(size_t)(1.0/kstep);
#pragma omp parallel
  {
    tightbindingcontext *tbc=this->getnewcontext();
    vector<double> kvector(3,0.0);
#pragma omp for
    for(size_t i=0;i<res;i++)
      for(size_t j=0;j<res;j++) {
	kvector[0]=-0.5+(double)i*kstep;
	kvector[1]=-0.5+(double)j*kstep;
	//total++;
	calcenergies(kvector,tbc);
	for(size_t l=0;l<n;l++)
	  if(gsl_vector_get(tbc->eval,l)<fermi)
	    occupied[l]+=weight;
      }
    delete tbc;
  }
  double occupation=0.0;
  for(size_t i=0;i<n;i++) {
    cout<<"Band "<<n<<": "<<occupied[i]<<endl;
    occupation+=(double) occupied[i];
  }
  return occupation;
}

double tightbind::getoccupation3d(double fermi,double kstep)
{
  double weight=kstep*kstep*kstep;
  vector<double> occupied(n,0.0);
  //size_t total=0;
  size_t res=(size_t)(1.0/kstep);
#pragma omp parallel
  {
    tightbindingcontext *tbc=this->getnewcontext();
    vector<double> kvector(3,0.0);
#pragma omp for
    for(size_t i=0;i<res;i++)
      for(size_t j=0;j<res;j++)
	for(size_t k=0;k<res;k++) {
	  kvector[0]=-0.5+(double)i*kstep;
	  kvector[1]=-0.5+(double)j*kstep;
	  kvector[2]=-0.5+(double)k*kstep;
	  //total++;
	  calcenergies(kvector,tbc);
	  for(size_t l=0;l<n;l++)
	    if(gsl_vector_get(tbc->eval,l)<fermi)
	      occupied[l]+=weight;
	}
    delete tbc;
  }
  double occupation=0.0;
  for(size_t i=0;i<n;i++) {
    cout<<"Band "<<n<<": "<<occupied[i]<<endl;
    occupation+=(double) occupied[i];
  }
  return occupation;
}

void tightbind::calc2dbandstructure(ostream &os, double kz,size_t npix)
{
  
  vector<vector<vector<double> > > bs;
  bs.resize(npix);
#pragma omp parallel
  {
    tightbindingcontext *tbc=this->getnewcontext();
    vector<double> kvector(3,0.0);
    kvector[2]=kz;
#pragma omp for
    for(size_t i=0;i<npix;i++) {
      bs[i].resize(npix);
      for(size_t j=0;j<npix;j++) {
	bs[i][j].resize(npix);
	kvector[0]=(double)2.0*i/(npix-1)*M_PI-M_PI;
	kvector[1]=(double)2.0*j/(npix-1)*M_PI-M_PI;
	calcenergies(kvector,tbc);
	for(size_t l=0;l<n;l++)
	  bs[i][j][l]=gsl_vector_get(tbc->eval,l);
      }
    }
    delete tbc;
  }
  for(size_t l=0;l<n;l++) {
    cout<<"#Band "<<l<<endl;
    for(size_t j=0;j<npix;j++) {
      for(size_t i=0;i<npix;i++)
	cout<<bs[i][j][l]<<" ";
      cout<<endl;
    }
  }      
}

void tightbind::writebs(const char *name,size_t pts,bool sort)
{
  idl map(pts,pts,n,1.0,1.0,1.0,0.0,0.0,-0.5,-0.5);
#pragma omp parallel
  {
    tightbindingcontext *tbc=this->getnewcontext();
    vector<double> kvector(3,0.0);  
#pragma omp for
    for(size_t i=0;i<pts;i++)
      for(size_t j=0;j<pts;j++) {
	kvector[0]=map.getx(i); kvector[1]=map.gety(j);
	calcenergies(kvector,tbc,sort);
	  for(size_t k=0;k<n;k++)
	    map.set(i,j,k,gsl_vector_get(tbc->eval,k));
      }
    delete tbc;
  }
  map.setname(date);
  map>>name;
}

void tightbind::writebs3d(const char *name,size_t pts,size_t zpts,bool sort)
{
  size_t nl=n*zpts;
  idl map(pts,pts,nl,1.0,1.0,1.0,0.0,0.0,-0.5,-0.5);
#pragma omp parallel
  {
    tightbindingcontext *tbc=this->getnewcontext();
    vector<double> kvector(3,0.0);  
#pragma omp for
    for(size_t l=0;l<zpts;l++)
      for(size_t i=0;i<pts;i++)
	for(size_t j=0;j<pts;j++) {
	  kvector[0]=map.getx(i); kvector[1]=map.gety(j); kvector[2]=(double)l/(zpts-1)-0.5;
	  calcenergies(kvector,tbc,sort);
	  for(size_t k=0;k<n;k++)
	    map.set(i,j,l*n+k,gsl_vector_get(tbc->eval,k));
	}
    delete tbc;
  }
  string comment;
  ostringstream nameos;
  nameos<<date<<", kgrid=("<<pts<<"x"<<pts<<"x"<<zpts<<")";
  date=nameos.str();
  map.setname(date);
  map>>name;
}

//weighting
void tightbind::calcbst3dhisthr(const char *fname,size_t pix,double le,double ue,size_t layers,size_t oversamp,double ux,double uy,size_t maxbands)
{
  size_t ospix=pix*oversamp;
  idl bst(pix,pix,layers,1.0*ux,1.0*uy,le,ue,0.0,-0.5*ux,-0.5*uy);
  const double binsize=(ue-le)/(layers-1);
  lorentzian weight((ue-le)/layers*2.0);
  if(!maxbands) maxbands=getbands();
#pragma omp parallel
  {
    tightbindingcontext *tbc=this->getnewcontext();
    vector<double> k(3,0.0);
#pragma omp for
    for(size_t io=0;io<ospix;io++)
      for(size_t jo=0;jo<ospix;jo++) {
	size_t i=io/oversamp,j=jo/oversamp;
	k[0]=(1.0/(ospix-1)*io-0.5)*ux; k[1]=(1.0/(ospix-1)*jo-0.5)*uy;
	calcenergies(k,tbc);
	for(size_t l=0;l<maxbands;l++) {
	  double ev=gsl_vector_get(tbc->eval,l);
	  if((ev>le-2.0*binsize) && (ev<ue+2.0*binsize)) {
	    int lay=bst.getbiasindex(ev),nls,nle;
	    nls=lay-1; nle=lay+1;
	    for(int nl=nls;nl<=nle;nl++)
	      if((nl>=0)&&(nl<(int)layers))
		{
		  double val=bst(i,j,nl)+weight(bst.getbias(nl)-ev);
		  bst.set(i,j,nl,val);
		}
	  }
	}
      }
    delete tbc;
  }
  bst.setname(date);
  bst>>fname;
}

//weighting
void tightbind::calcbst3dhisthrsel(const char *fname,size_t pix,double le,double ue,size_t layers,size_t oversamp,vector<size_t> &orbs,double ux,double uy)
{
  size_t ospix=pix*oversamp,orbitals=n;
  idl bst(pix,pix,layers,1.0*ux,1.0*uy,le,ue,0.0,-0.5*ux,-0.5*uy);
  const double binsize=(ue-le)/(layers-1);
  lorentzian weight((ue-le)/layers*2.0);
  if(orbs.size()<n) orbitals=orbs.size();
#pragma omp parallel
  {
    tightbindingcontext *tbc=this->getnewcontext();
    vector<double> k(3,0.0);
#pragma omp for
    for(size_t io=0;io<ospix;io++)
      for(size_t jo=0;jo<ospix;jo++) {
	size_t i=io/oversamp,j=jo/oversamp;
	k[0]=(1.0/(ospix-1)*io-0.5)*ux; k[1]=(1.0/(ospix-1)*jo-0.5)*uy;
	calcenergies(k,tbc);
	for(size_t l=0;l<orbitals;l++) {
	  double ev=gsl_vector_get(tbc->eval,l);
	  if((ev>le) && (ev<ue+binsize)) {
	    size_t lay=bst.getbiasindex(ev),nls,nle;
	    if(lay<layers) {
	      if(lay>0) nls=lay-1; else nls=0;
	      if(lay<layers-1) nle=lay+1; else nle=layers-1;
	      for(size_t nl=nls;nl<=nle;nl++) {
		for(size_t evc=0;evc<n;evc++) {
		  gsl_complex evv=gsl_matrix_complex_get(tbc->evec,evc,l);
		  complex<double> evvc(GSL_REAL(evv), GSL_IMAG(evv));
		  double val=bst(i,j,nl)+weight(bst.getbias(nl)-ev)*abs(evvc)*(double)orbs[l];
		  bst.set(i,j,nl,val);
		}
	      }
	    }
	  }
	}
      }
    delete tbc;
  }
  bst.setname(date);
  bst>>fname;
}

//weighting
void tightbind::calcbst3dhisthr(const char *fname,size_t pix,size_t kint,double le,double ue,size_t layers,size_t oversamp,size_t maxbands)
{
  size_t ospix=pix*oversamp;
  idl bst(pix,pix,layers,1.0,1.0,le,ue,0.0,-0.5,-0.5);
  const double binsize=(ue-le)/(layers-1);
  lorentzian weight((ue-le)/layers*2.0);
  if(!maxbands) maxbands=getbands();
#pragma omp parallel
  {
    tightbindingcontext *tbc=this->getnewcontext();
    vector<double> k(3,0.0);
#pragma omp for
    for(size_t io=0;io<ospix;io++)
      for(size_t jo=0;jo<ospix;jo++)
	for(size_t ko=0;ko<kint;ko++) {
	  size_t i=io/oversamp,j=jo/oversamp;
	  k[0]=1.0/(ospix-1)*io-0.5; k[1]=1.0/(ospix-1)*jo-0.5; k[2]=1.0/(kint-1)*ko-0.5;
	  calcenergies(k,tbc);
	  for(size_t l=0;l<maxbands;l++) {
	    double ev=gsl_vector_get(tbc->eval,l);
	    if((ev>le) && (ev<ue+binsize)) {
	      size_t lay=bst.getbiasindex(ev),nls,nle;
	      if(lay<layers) {
		if(lay>0) nls=lay-1; else nls=0;
		if(lay<layers-1) nle=lay+1; else nle=layers-1;
		for(size_t nl=nls;nl<=nle;nl++) {
		  double val=bst(i,j,nl)+weight(bst.getbias(nl)-ev);
		  bst.set(i,j,nl,val);
		}
	      }
	    }
	  }
	}
    delete tbc;
  }
  bst.setname(date);
  bst>>fname;
}

//weighting
void tightbind::calcbstcechisthr(const char *fname,double e,size_t pix,size_t zpix,size_t oversamp,double width)
{
  size_t ospix=pix*oversamp,zospix=zpix*oversamp;
  idl bst(pix,pix,zpix,1.0,1.0,-0.5,0.5,0.0,-0.5,-0.5);
  lorentzian weight(width);
#pragma omp parallel
  {
    tightbindingcontext *tbc=this->getnewcontext();
    vector<double> kvector(3,0.0);
#pragma omp for
    for(size_t io=0;io<ospix;io++)
      for(size_t jo=0;jo<ospix;jo++)
	for(size_t ko=0;ko<zospix;ko++) {
	  size_t i=io/oversamp,j=jo/oversamp,k=ko/oversamp;
	  kvector[0]=1.0/(ospix-1)*io-0.5; kvector[1]=1.0/(ospix-1)*jo-0.5; kvector[2]=1.0/(zospix-1)*ko-0.5;
	  calcenergies(kvector,tbc);
	  double val=bst(i,j,k);
	  for(size_t l=0;l<n;l++) {
	    double ev=gsl_vector_get(tbc->eval,l);
	    val+=weight(e-ev);
	  }
	  bst.set(i,j,k,val);
	}
    delete tbc;
  }
  bst.setname(date);
  bst>>fname;
}

//complex JDOS calculation
//do here orbitally projected QPI
void tightbind::calcjdoshr(const char *fname,size_t pix,double le,double ue,size_t layers,size_t oversamp)
{
  double div=oversamp*oversamp;
  size_t ospix=pix*oversamp;
  idl bst(pix,pix,layers,1.0,1.0,le,ue,0.0,-0.5,-0.5);
  vector<vector<vector<vector<complex<double> > > > > arrays;
  arrays.resize(pix);
  for(size_t i=0;i<pix;i++) {
    arrays[i].resize(pix);
    for(size_t j=0;j<pix;j++) {
      arrays[i][j].resize(layers);
      for(size_t l=0;l<layers;l++) 
        arrays[i][j][l].assign(n,complex<double>(0.0,0.0));
    }
  }
  div/=bst.getrelx(1)*bst.getrely(1)*fabs(bst.getbias(1)-bst.getbias(0));
  const double binsize=(ue-le)/(layers-1);
  lorentzian weight((ue-le)/layers*2.0);
#pragma omp parallel
  {
    tightbindingcontext *tbc=this->getnewcontext();
    vector<double> k(3,0.0);
#pragma omp for
    for(size_t io=0;io<ospix;io++)
      for(size_t jo=0;jo<ospix;jo++) {
	size_t i=io/oversamp,j=jo/oversamp;
	k[0]=1.0/(ospix-1)*io-0.5; k[1]=1.0/(ospix-1)*jo-0.5;
	calcenergies(k,tbc);
	for(size_t l=0;l<n;l++) {
	  double ev=gsl_vector_get(tbc->eval,l);
	  if((ev>le) && (ev<ue+binsize)) {
	    size_t lay=bst.getbiasindex(ev),nls,nle;
	    if(lay<layers) {
	      if(lay>0) nls=lay-1; else nls=0;
	      if(lay<layers-1) nle=lay+1; else nle=layers-1;
	      for(size_t nl=nls;nl<=nle;nl++)
		for(size_t evc=0;evc<n;evc++) {
		  gsl_complex evv=gsl_matrix_complex_get(tbc->evec,evc,l);
		  complex<double> evvc(GSL_REAL(evv), GSL_IMAG(evv));
		  arrays[i][j][nl][evc]+=weight(bst.getbias(nl)-ev)*abs(evvc)/div;
		}
	    }
	  }
	}
      }
    delete tbc;
  }
  size_t fftsize=pix*pix;
  double normconst=pix*pix;
  complex<double> *fftdata=new complex<double>[fftsize];
  fftw_init_threads();
  fftw_plan_with_nthreads(omp_get_max_threads());
  fftw_plan myplan=fftw_plan_dft_2d(pix,pix,reinterpret_cast<fftw_complex*>(fftdata),reinterpret_cast<fftw_complex*>(fftdata),FFTW_FORWARD,FFTW_ESTIMATE),
    myinvplan=fftw_plan_dft_2d(pix,pix,reinterpret_cast<fftw_complex*>(fftdata),reinterpret_cast<fftw_complex*>(fftdata),FFTW_BACKWARD,FFTW_ESTIMATE);
  for(size_t l=0;l<layers;l++)
    for(size_t e=0;e<n;e++) {
      for(size_t i=0;i<fftsize;i++) {
        size_t xc=i/pix,yc=(i%pix);
        fftdata[i]=arrays[xc][yc][l][e];
      }
      fftw_execute(myplan);
      for(size_t i=0;i<fftsize;i++)
        fftdata[i]=fftdata[i]*conj(fftdata[i]);
      fftw_execute(myinvplan);
      for(size_t i=0;i<fftsize;i++) {
        size_t xc=i/pix,yc=(i%pix);
        arrays[xc][yc][l][e]=fftdata[i]/normconst;
      }
    }
  fftw_destroy_plan(myplan); fftw_destroy_plan(myinvplan);
  delete[] fftdata;
  //write data to target
#pragma omp parallel
  {
#pragma omp for
    for(size_t i=0;i<pix;i++)
      for(size_t j=0;j<pix;j++)
	for(size_t l=0;l<layers;l++) {
	  complex<double> val(0.0,0.0);
	  for(size_t e=0;e<n;e++)
	    val+=arrays[i][j][l][e];
	  bst.set(i,j,l,val.real());
	  //bst.set(i,j,l,abs(val*conj(val)));
	}
  }
  bst.shift(-(int)(pix>>1),-(int)(pix>>1));
  bst.setname(date);
  bst>>fname;
}

size_t tightbind::findindex(int x, int y, int z,size_t o1,size_t o2) {
  size_t searchindex;
  for(searchindex=0;searchindex<neighbourno;searchindex++)
    if((hoppings[searchindex].rx==x) &&
       (hoppings[searchindex].ry==y)&&
       (hoppings[searchindex].rz==z)&&
       (hoppings[searchindex].o1==o1)&&
       (hoppings[searchindex].o2==o2))
      break;
  return searchindex;
}
