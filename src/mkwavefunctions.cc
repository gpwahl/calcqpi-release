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

//calculate wavefunctions for overlap with tip

#include <iostream>
#include <iomanip>
#include <unistd.h>
#include <sstream>
#include <string.h>
#include <fstream>
#include <vector>
#include <stdio.h>

using namespace std;

#define _MPI_MAIN

#include "idl.h"
#include "wannierfunctions.h"

#include "addtimestamp.h"
#include "parser.h"
#include "gitversion.h"

#include <gsl/gsl_interp.h>

idl *interpolate(vector<idl *> &idlorbitals,vector<double> &values, double x,const gsl_interp_type *gsl_intp=gsl_interp_linear) {
  size_t layers=idlorbitals[0]->layers();
  size_t norbs=idlorbitals.size();
  for(size_t i=1;i<norbs;i++)
    if(layers!=idlorbitals[i]->layers()) return NULL;
  gsl_interp *gslintp=gsl_interp_alloc(gsl_intp,norbs);
  cerr<<"Using interpolation type "<<gsl_interp_name(gslintp)<<endl;
  gsl_interp_accel *acc=gsl_interp_accel_alloc ();
  double *xvals=new double[norbs], *yvals=new double[norbs];
  size_t xs,ys;
  idlorbitals[0]->dimensions(xs,ys);
  idl *intorb=new idl(xs,ys,layers);
  ostringstream nameos;
  nameos<<idlorbitals[0]->getname()<<" interpolated at x="<<x<<" using "<<gsl_interp_name(gslintp);
  intorb->setname(nameos.str());
  for(size_t i=0;i<norbs;i++)
    xvals[i]=values[i];
  for(size_t i=0;i<layers;i++) {
    for(size_t l=0;l<xs;l++)
      for(size_t m=0;m<ys;m++) {
	for(size_t j=0;j<norbs;j++)
	  yvals[j]=idlorbitals[j]->operator()(l,m,i);
	gsl_interp_init(gslintp, xvals, yvals, norbs);
	intorb->set(l,m,i,gsl_interp_eval(gslintp, xvals, yvals, x, acc));
      }
  }
  gsl_interp_accel_free(acc);
  gsl_interp_free(gslintp);
  delete[] xvals;
  delete[] yvals;
  return intorb;
}

void help()
{
  cerr<<"Syntax:"<<endl
      <<"mkwavefunctions configfile"<<endl<<endl
      <<"Parameters of the configuration file:"<<endl
      <<"Input and output files:"<<endl
      <<" logfile    : output log to logfile (default: cout)"<<endl
      <<" tbfile     : name of tightbinding model"<<endl
      <<" mkwffile   : output wave functions to the specified file"<<endl
      <<"Parameters for QPI calculation:"<<endl
      <<" oversamp   : n-times oversampling (default: 4)"<<endl
      <<" window     : window of size n (default: 0 (auto), if too small, a warning will be issued)"<<endl
      <<"              if no window specified, the value will be estimated automatically"<<endl
      <<"Parameters for Slater-type orbitals:"<<endl
      <<" radius     : use radius for size of orbital functions (default: 0.5)"<<endl
      <<" radarr     : an array of radii, one for each orbital"<<endl
      <<" angle      : rotate wannier functions by specified angle (default: 0.0)"<<endl
      <<" anglearr   : additional angle by which individual Wannier functions are being rotated (default: none)"<<endl
      <<" phiarr     : out-of-plane angle by which individual Wannier functions are being rotated (default: none)"<<endl
      <<" prearr     : array of prefactors for orbitals (default: all 1)"<<endl
      <<" orbitals   : list of orbitals used for Wannier functions"<<endl
      <<" pos[0], ...: positions of the orbitals (default: 0,0)"<<endl
      <<" zheight    : height of z-layer above top-most atom (when using Wannier functions from DFT, negative height is below bottom atom; used also for atomic-like orbitals)"<<endl
      <<" basisvector[0], [1]: basis vectors for a basis where they are not just (1,0) and (0,1)"<<endl
      <<"Input wave functions from DFT or other sources"<<endl
      <<" orbitalfiles: list of files used for Wannier functions"<<endl
      <<" idlorbitalfile: idl file containing the Wannier functions"<<endl
      <<"Input wave functions from DFT for further modification"<<endl
      <<" modorbitalfiles: list of files for Wannier functions"<<endl
      <<" modpos[0],...           : shift position of input wave function"<<endl
      <<"Interpolation of orbitals:"<<endl
      <<" idlorbitalfiles         : list of idl files containing the Wannier functions for interpolation"<<endl
      <<" values=(x1,x2,...)      : values corresponding to filename1, filename2, ..., is no values are provided, x1 is assumed to be zero and xn 1"<<endl
      <<" interpolate=val         : value at which the model is to be interpolated"<<endl
      <<" interpmethod=\"method\" : interpolation method, one of linear, polynomial, akima, cspline, cspline_periodic, akima_periodic, steffen"<<endl
      <<"Reorganizing wave functions:"<<endl
      <<" double     : double orbitals for spin-polarized calculation"<<endl
      <<" reorder    : reorder - provide list of numbers which contains indices to the original order."<<endl;
}

void checkmax(double val, double &max) {
  if(val>max) max=val;
}

void checkwavefunctions(ostream &os,vector<wannierfunctions> &wfs)
{
  double max=0.0;
  os<<"Checking wave functions:"<<endl;
  for(size_t i=0;i<wfs.size();i++) {
    double wfmaxb=wfs[i].getmaxboundary(),wfmax=wfs[i].getmax();
    os<<" Wave function "<<i<<": Max:"<<wfmax<<", Max at boundary: "<<wfmaxb<<", Ratio: "<<wfmaxb/wfmax<<endl;
    checkmax(wfmaxb/wfmax,max);
  }
  os<<"---------------------------"<<endl
    <<"Maximum rel. value at boundary: "<<max<<endl;
  if(max>0.05) os<<"Warning: boundary too close, use larger window size!"<<endl;
}

int main(int argc, char *argv[])
{
  size_t oversamp=4,window=2;
  double radius=0.5,angle=0.0,zheight=5.0;
  vector<vector<double> > basisvectors;
  vector<wannierfunctions> wfs;
  vector<xcfloader *> modwfs;
  ostream *logstream;
  ofstream logfilestr;
  if(argc>1) {
    streambuf* oldcerrstreambuf = cerr.rdbuf();
    ostringstream errbuf;
    cerr.rdbuf( errbuf.rdbuf() );
    loadconfig lf(argv[1]);
    if(!lf) {
      cerr.rdbuf(oldcerrstreambuf);
      if(errbuf.str().length()) cerr<<errbuf.str()<<endl;
      return -1;
    }
    if(lf.probefield("logfile")) {
      string logfile=lf.getstring("logfile");
      logfilestr.open(logfile.c_str(),ofstream::out|ofstream::app);
      if(logfilestr.is_open()) {
	logstream=(ostream *)&logfilestr;
	std::cerr.rdbuf(logstream->rdbuf());
	std::cout.rdbuf(logstream->rdbuf());
      }
    } else cerr.rdbuf(oldcerrstreambuf);
    AddTimeStamp ats(cerr),ats2(cout);
    cout<<"mkwavefunctions (commit "<<gitversion<<"), setting wave functions from "<<argv[1]<<endl;
    if(errbuf.str().length()) cerr<<"Error while reading configuration file:"<<endl<<errbuf.str();
    if(!lf.probefield("mkwffile")) {
      cerr<<"No output file specified, exiting."<<endl;
      return 0;
    }
    if(lf.probefield("oversamp")) oversamp=lf.getintvalue("oversamp");
    if(lf.probefield("window")) window=lf.getintvalue("window");
    
    cout<<"Parameters of calculation:"<<endl
	      <<"         window               : "<<window<<endl
	      <<"         oversamp             : "<<oversamp<<endl;
    if(lf.probefield("basisvector[0]")) {
      vector<double> avector=lf.getvector("basisvector[0]");
      avector.resize(2,0.0);
      basisvectors.push_back(avector);
      if(!lf.probefield("basisvector[1]")) {
	cerr<<"Error: only one basis vector specified."<<endl;
	return 0;
      }
      avector=lf.getvector("basisvector[1]");
      avector.resize(2,0.0);
      basisvectors.push_back(avector);
      cout<<"Using basis vectors ("<<basisvectors[0][0]<<","<<basisvectors[0][1]<<"), ("<<basisvectors[1][0]<<","<<basisvectors[1][1]<<")."<<endl;
    }
    if(lf.probefield("modorbitalfiles")) {
      cout<<"Using Wannier90/XSF-orbitals for vacuum overlap (mod code)."<<endl;
      vector<string> modorbitalfiles=lf.getstringlist("modorbitalfiles");
      for(size_t i=0;i<modorbitalfiles.size();i++) {
	cout<<"Loading wave function file "<<modorbitalfiles[i]<<"."<<endl;
	modwfs.push_back(new xcfloader(modorbitalfiles[i].c_str()));
	if(!modwfs[i]) {
	  cerr<<"Error loading wave function file "<<modorbitalfiles[i]<<endl;
	  return 0;
	}
	ostringstream modposstr;
	modposstr<<"modpos["<<i<<"]";
	if(lf.probefield(modposstr.str())) {
	  vector<double> posarr=lf.getvector(modposstr.str());
	  modwfs[i]->setshift(-posarr[0],-posarr[1],-posarr[2]);
	}
      }
      vector<double> orbitals=lf.getvector("orbitals");
      if(lf.probefield("angle")) angle=lf.getvalue("angle");
      if(lf.probefield("zheight")) zheight=lf.getvalue("zheight");
      vector<double> anglearr;
      if(lf.probefield("anglearr")) anglearr=lf.getvector("anglearr");
      else anglearr.assign(orbitals.size(),0.0);
      for(size_t i=0;i<orbitals.size();i++) {
	size_t orbnum=(size_t)orbitals[i];
	ostringstream posstr;
	posstr<<"pos["<<i<<"]";
	if(lf.probefield(posstr.str())) {
	  vector<double> pos=lf.getvector(posstr.str());
	  if(pos.size()==2) pos.push_back(0.0);
	  wfs.push_back(wannierfunctions(basisvectors,modwfs[orbnum],window,oversamp,zheight-pos[2],pos[0],pos[1],(angle+anglearr[i])*M_PI/180.0));
	} else 
	  wfs.push_back(wannierfunctions(basisvectors,modwfs[orbnum],window,oversamp,zheight,0.0,0.0,(angle+anglearr[i])*M_PI/180.0));
      }
      cout<<"Parameters for Wannier functions:"<<endl;
      for(size_t i=0;i<wfs.size();i++) {
	double x,y,h;
	wfs[i].getpos(x,y,h);
	cout<<"Orbital "<<i<<": "<<"@("<<x<<","<<y<<"), height "<<h<<", angle theta "<<wfs[i].gettheta()*180.0/M_PI<<"deg"<<endl;
      }
      if(modwfs.size()) {
	for(size_t i=0;i<modwfs.size();i++)
	  if(modwfs[i])
	    delete modwfs[i];
      }
    }
    //Initialization of orbitals for Gausian orbital mode
    else if(lf.probefield("idlorbitalfile")) { //Initialization of orbitals from an IDL file written by calcqpi
      string orbitalfile=lf.getstring("idlorbitalfile");
      cout<<"Using IDL orbital file for vacuum overlap."<<endl
		<<"Loading wave function file "<<orbitalfile<<"."<<endl;
      idl orbitals(orbitalfile.c_str());
      size_t orbs=orbitals.layers();
      vector<double> prearr;
      if(lf.probefield("prearr")) prearr=lf.getvector("prearr");
      prearr.resize(orbs,1.0);
      vector<vector<double> > arr;
      size_t xs,ys;
      orbitals.dimensions(xs,ys);
      if((xs!=ys) || (xs!=(2*window+1)*oversamp)) {
	cerr<<"Orbital functions in "<<orbitalfile<<" have unsuitable dimensions."<<endl;
	return 0;
      }
      arr.resize(xs);
      for(size_t i=0;i<xs;i++)
	arr[i].assign(ys,0.0);
      for(size_t k=0;k<orbs;k++) {
	for(size_t i=0;i<xs;i++)
	  for(size_t j=0;j<ys;j++)
	    arr[i][j]=orbitals(i,j,k);
	wfs.push_back(wannierfunctions(arr,prearr[k]));
      }
    } else if(lf.probefield("orbitals")) {
      vector<string> orbitals=lf.getlist("orbitals");
      vector<double> radarr;
      if(lf.probefield("radius")) radius=lf.getvalue("radius");
      if(lf.probefield("radarr"))
	radarr=lf.getvector("radarr");
      if(lf.probefield("angle")) angle=lf.getvalue("angle");
      if(lf.probefield("zheight")) zheight=lf.getvalue("zheight");
      cout<<"Using gaussian orbitals for vacuum overlap."<<endl
	  <<"    Parameters:"<<endl
	  <<"             Radius:         ";
      if(radarr.size()) {
	for(size_t i=0;i<radarr.size();i++) {
	  if(i)
	    cout<<" "<<radarr[i];
	  else cout<<radarr[i];
	}
      } else
	cout<<radius;
      cout<<endl<<"             Rotation angle: "<<angle<<endl
	  <<"             Height:         "<<zheight<<endl;
      vector<double> anglearr;
      if(lf.probefield("anglearr")) anglearr=lf.getvector("anglearr");
      else anglearr.assign(orbitals.size(),0.0);
      vector<double> phiarr;
      if(lf.probefield("phiarr")) phiarr=lf.getvector("phiarr");
      else phiarr.assign(orbitals.size(),0.0);
      vector<double> prearr;
      if(lf.probefield("prearr")) prearr=lf.getvector("prearr");
      prearr.resize(orbitals.size(),1.0);
      for(size_t i=0;i<orbitals.size();i++) {
	ostringstream posstr;
	posstr<<"pos["<<i<<"]";
	if(radarr.size())
	  radius=radarr[i%radarr.size()];
	if(lf.probefield(posstr.str())) {
	  vector<double> pos=lf.getvector(posstr.str());
	  if(pos.size()==2) pos.push_back(0.0);
	  wfs.push_back(wannierfunctions(basisvectors,orbitals[i],radius,zheight-pos[2],pos[0],pos[1],(angle+anglearr[i])*M_PI/180.0,phiarr[i]*M_PI/180.0,prearr[i]));
	} else 
	  wfs.push_back(wannierfunctions(basisvectors,orbitals[i],radius,zheight,0.0,0.0,(angle+anglearr[i])*M_PI/180.0,phiarr[i]*M_PI/180.0,prearr[i]));
      }
      cout<<"Parameters for Wannier functions:"<<endl;
      for(size_t i=0;i<wfs.size();i++) {
	double x,y,h;
	if(radarr.size()) radius=radarr[i%radarr.size()];
	wfs[i].getpos(x,y,h);
	cout<<"Orbital "<<i<<": "<<wfs[i].getname()<<"@("<<x<<","<<y<<"), height "<<h<<", angle theta "<<wfs[i].gettheta()*180.0/M_PI<<"deg, phi "<<wfs[i].getphi()*180.0/M_PI<<" deg, radius="<<radius<<endl;
      }
      cout<<"Precalculating wave functions."<<endl;
      for(size_t i=0;i<wfs.size();i++)
	wfs[i].precalculate(window,oversamp);
      cout<<"Done."<<endl;
    } else if(lf.probefield("orbitalfiles")) { //Initialization of orbitals from DFT/Wannier90 output
      if(lf.probefield("zheight")) zheight=lf.getvalue("zheight");
      cout<<"Using Wannier90/XSF-orbitals for vacuum overlap."<<endl
		<<"    Parameters:"<<endl
		<<"             zheight:         "<<zheight<<endl;
      vector<string> orbitalfiles=lf.getstringlist("orbitalfiles");
      vector<double> prearr;
      if(lf.probefield("prearr")) prearr=lf.getvector("prearr");
      prearr.resize(orbitalfiles.size(),1.0);
      vector<double> shift(3,0.0);
      if(lf.probefield("shift")) {
	shift=lf.getvector("shift");
	shift.resize(3,0.0);
      } 
      cout<<"Applying global shift of ("<<shift[0]<<","<<shift[1]<<","<<shift[2]<<")"<<endl;
      for(size_t i=0;i<orbitalfiles.size();i++) {
	cout<<"Loading wave function file "<<orbitalfiles[i]<<"."<<endl;
	wfs.push_back(wannierfunctions(orbitalfiles[i],shift,zheight,window,oversamp,prearr[i]));
	if(!wfs[i]) {
	  cerr<<"Error loading wave function file "<<orbitalfiles[i]<<endl;
	  return 0;
	}
      }
    } else if(lf.probefield("idlorbitalfiles")) {
      vector<string> idlorbfiles=lf.getstringlist("idlorbitalfiles");
      const gsl_interp_type *gslintptype=gsl_interp_linear;
      vector<double> values;
      double x=0.5;
      if(lf.probefield("values"))
	values=lf.getvector("values");
      else {
	values.resize(idlorbfiles.size());
	for(size_t i=0;i<values.size();i++)
	  values[i]=(double)i/(values.size()-1);
      }
      if(lf.probefield("interpolate"))
	x=lf.getvalue("interpolate");
      if(lf.probefield("interpmethod")) {
	string imethod=lf.getstring("interpmethod");
	cerr<<"Setting interpolation method to "<<imethod<<endl;
	if(imethod=="linear")
	  gslintptype=gsl_interp_linear;
	else if(imethod=="polynomial")
	  gslintptype=gsl_interp_polynomial;
	else if(imethod=="cspline")
	  gslintptype=gsl_interp_cspline;
	else if(imethod=="cspline_periodic")
	  gslintptype=gsl_interp_cspline_periodic;
	else if(imethod=="akima")
	  gslintptype=gsl_interp_akima;
	else if(imethod=="akima_periodic")
	  gslintptype=gsl_interp_akima_periodic;
	else if(imethod=="steffen")
	  gslintptype=gsl_interp_steffen;
	else cerr<<"Unknown interpolation method '"<<imethod<<"'"<<endl;
      }
      vector<idl *> idlorbitals;
      idlorbitals.resize(idlorbfiles.size());
      for(size_t i=0;i<idlorbfiles.size();i++) {
	cerr<<"Loading orbital file "<<idlorbfiles[i]<<endl;
	idlorbitals[i]=new idl(idlorbfiles[i].c_str());
      }
      cerr<<"Interpolating model at value "<<x<<endl;
      idl *intorb=interpolate(idlorbitals,values,x,gslintptype);
      for(size_t i=0;i<idlorbitals.size();i++)
	delete idlorbitals[i];
      size_t orbs=intorb->layers();
      vector<vector<double> > arr;
      size_t xs,ys;
      intorb->dimensions(xs,ys);
      if((xs!=ys) || (xs!=(2*window+1)*oversamp)) {
	cerr<<"Orbital functions in have unsuitable dimensions."<<endl;
	return 0;
      }
      arr.resize(xs);
      for(size_t i=0;i<xs;i++)
	arr[i].assign(ys,0.0);
      for(size_t k=0;k<orbs;k++) {
	for(size_t i=0;i<xs;i++)
	  for(size_t j=0;j<ys;j++)
	    arr[i][j]=intorb->operator()(i,j,k);
	wfs.push_back(wannierfunctions(arr));
      }
      delete intorb;
    }
    checkwavefunctions(cout,wfs); 
    if(lf.probefield("reorder")) {
      cout<<"Reordering orbitals."<<endl;
      vector<double> reordervals=lf.getvector("reorder");
      vector<size_t> reorderarr;
      reorderarr.resize(reordervals.size());
      for(size_t i=0;i<reordervals.size();i++) 
	reorderarr[i]=(size_t)reordervals[i];
      vector<wannierfunctions> newwfs(wfs);
      wfs.clear();  
      for(size_t i=0;i<reorderarr.size();i++) 
	wfs.push_back(newwfs[reorderarr[i]]);
    }
    if(lf.probefield("double")) {
      cout<<"Duplicating orbitals for spin-polarized calculation."<<endl;
      size_t orbitals=wfs.size();
      for(size_t i=0;i<orbitals;i++)
	wfs.push_back(wfs[i]);
    }
    if(lf.probefield("mkwffile")) {
      string tbfile;
      if(lf.probefield("tbfile")) tbfile=lf.getstring("tbfile");
      size_t xs,ys;
      wfs[0].getcachesize(xs,ys);
      idl wfout(xs,ys,wfs.size());
      for(size_t l=0;l<wfs.size();l++)
	for(size_t i=0;i<xs;i++)
	  for(size_t j=0;j<ys;j++)
	    wfout.set(i,j,l,wfs[l].getwave_cached(i,j));
      ostringstream comment;
      comment<<"Wannier function file for "<<tbfile<<" with window="<<window<<", oversamp="<<oversamp<<", zheight="<<zheight;
      wfout.setname(comment.str());
      cout<<"Writing wave function file to "<<lf.getstring("mkwffile")<<endl;
      wfout>>lf.getstring("mkwffile").c_str();
    }
    return 0;
  }
  else help();
  return 1;
}





