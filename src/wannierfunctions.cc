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

#include "wannierfunctions.h"
#include <math.h>

#include <iostream>
using namespace std;

double wannierfunctions::swave(double x, double y, double z) {
  double r=sqrt(x*x+y*y+z*z);
  return prefact*exp(-r/width)/sqrt(4.0*M_PI);
}

double wannierfunctions::pxwave(double x, double y, double z) {
  double r=sqrt(x*x+y*y+z*z);
  if(r==0.0) return 0.0;
  else return prefact*exp(-r/width)*x/r*sqrt(3.0/4.0/M_PI);
}

double wannierfunctions::pywave(double x, double y, double z) {
  double r=sqrt(x*x+y*y+z*z);
  if(r==0.0) return 0.0;
  else return prefact*exp(-r/width)*y/r*sqrt(3.0/4.0/M_PI);
}

double wannierfunctions::pzwave(double x, double y, double z) {
  double r=sqrt(x*x+y*y+z*z);
  if(r==0.0) return 0.0;
  else return prefact*exp(-r/width)*z/r*sqrt(3.0/4.0/M_PI);
}

double wannierfunctions::dx2wave(double x, double y, double z) {
  double r=sqrt(x*x+y*y+z*z);
  if(r==0.0) return 0.0;
  else return prefact*exp(-r/width)*(x*x-y*y)/2.0/r/r*sqrt(15.0/4.0/M_PI);
}

double wannierfunctions::dxywave(double x, double y, double z) {
  double r=sqrt(x*x+y*y+z*z);
  if(r==0.0) return 0.0;
  else return prefact*exp(-r/width)*x*y/r/r*sqrt(15.0/4.0/M_PI);
}

double wannierfunctions::dxzwave(double x, double y, double z) {
  double r=sqrt(x*x+y*y+z*z);
  if(r==0.0) return 0.0;
  else return prefact*exp(-r/width)*x*z/r/r*sqrt(15.0/4.0/M_PI);
}

double wannierfunctions::dyzwave(double x, double y, double z) {
  double r=sqrt(x*x+y*y+z*z);
  if(r==0.0) return 0.0;
  else return prefact*exp(-r/width)*y*z/r/r*sqrt(15.0/4.0/M_PI);
}

double wannierfunctions::dr2wave(double x, double y, double z) {
  double r=sqrt(x*x+y*y+z*z);
  if(r==0.0) return 0.0;
  else return prefact*exp(-r/width)*(3.0*z*z-r*r)/4.0/r/r*sqrt(5.0/M_PI);
}

double wannierfunctions::fy3x2wave(double x, double y, double z) {
  double r=sqrt(x*x+y*y+z*z);
  if(r==0.0) return 0.0;
  else return prefact*exp(-r/width)*y*(3.0*x*x-y*y)/4.0/r/r/r*sqrt(35.0/2.0/M_PI);
}

double wannierfunctions::fxyzwave(double x, double y, double z) {
  double r=sqrt(x*x+y*y+z*z);
  if(r==0.0) return 0.0;
  else return prefact*exp(-r/width)*x*y*z/r/r/r*sqrt(105.0/4.0/M_PI);
}

double wannierfunctions::fyz2wave(double x, double y, double z) {
  double r=sqrt(x*x+y*y+z*z);
  if(r==0.0) return 0.0;
  else return prefact*exp(-r/width)*y*(5.0*z*z-r*r)/r/r/r/4.0*sqrt(21.0/2.0/M_PI);
}

double wannierfunctions::fz3wave(double x, double y, double z) {
  double r=sqrt(x*x+y*y+z*z);
  if(r==0.0) return 0.0;
  else return prefact*exp(-r/width)*(5.0*z*z*z-3.0*z*r*r)/r/r/r/4.0*sqrt(7.0/M_PI);
}

double wannierfunctions::fxz2wave(double x, double y, double z) {
  double r=sqrt(x*x+y*y+z*z);
  if(r==0.0) return 0.0;
  else return prefact*exp(-r/width)*x*(5.0*z*z-r*r)/4.0/r/r/r*sqrt(21.0/2.0/M_PI);
}

double wannierfunctions::fzx2wave(double x, double y, double z) {
  double r=sqrt(x*x+y*y+z*z);
  if(r==0.0) return 0.0;
  else return prefact*exp(-r/width)*z*(x*x-y*y)/r/r/r/4.0*sqrt(105.0/M_PI);
}

double wannierfunctions::fxx2wave(double x, double y, double z) {
  double r=sqrt(x*x+y*y+z*z);
  if(r==0.0) return 0.0;
  else return prefact*exp(-r/width)*x*(x*x-3.0*y*y)/4.0/r/r/r*sqrt(35.0/2.0/M_PI);
}

double wannierfunctions::nsymmwave(double x, double y, double z) {
  double r=sqrt(x*x+y*y+z*z);
  double scos=cos(symmetry*atan2(x,y)/2.0);
  return prefact*exp(-r/width)*scos*scos*r/width;
}

double wannierfunctions::zerowave(double x, double y, double z) {
  return 0.0;
}

double wannierfunctions::getwave(double x,double y) {
  double z=0.0; //z=height;
  switch(type) {
  case swavetype :
    shift(x,y,z);
    if(basisvectors.size()) transform(x,y);
    return swave(x,y,z);
  case pxwavetype :
    shift(x,y,z);
    if(basisvectors.size()) transform(x,y);
    if((theta!=0.0) || (phi!=0.0)) rotate(x,y,z);    
    return pxwave(x,y,z);
  case pywavetype :
    shift(x,y,z);
    if(basisvectors.size()) transform(x,y);
    if((theta!=0.0) || (phi!=0.0)) rotate(x,y,z);
    return pywave(x,y,z);
  case pzwavetype :
    shift(x,y,z);
    if(basisvectors.size()) transform(x,y);
    if((theta!=0.0) || (phi!=0.0)) rotate(x,y,z);
    return pzwave(x,y,z);
  case dx2wavetype :
    shift(x,y,z);
    if(basisvectors.size()) transform(x,y);
    if((theta!=0.0) || (phi!=0.0)) rotate(x,y,z);
    return dx2wave(x,y,z);
  case dxywavetype :
    shift(x,y,z);
    if(basisvectors.size()) transform(x,y);
    if((theta!=0.0) || (phi!=0.0)) rotate(x,y,z);
    return dxywave(x,y,z);
  case dxzwavetype :
    shift(x,y,z);
    if(basisvectors.size()) transform(x,y);
    if((theta!=0.0) || (phi!=0.0)) rotate(x,y,z);
    return dxzwave(x,y,z);
  case dyzwavetype :
    shift(x,y,z);
    if(basisvectors.size()) transform(x,y);
    if((theta!=0.0) || (phi!=0.0)) rotate(x,y,z);
    return dyzwave(x,y,z);
  case dr2wavetype :
    shift(x,y,z);
    if(basisvectors.size()) transform(x,y);
    if((theta!=0.0) || (phi!=0.0)) rotate(x,y,z);
    return dr2wave(x,y,z);
  case fy3x2wavetype :
    shift(x,y,z);
    if(basisvectors.size()) transform(x,y);
    if((theta!=0.0) || (phi!=0.0)) rotate(x,y,z);
    return fy3x2wave(x,y,z);
  case fxyzwavetype :
    shift(x,y,z);
    if(basisvectors.size()) transform(x,y);
    if((theta!=0.0) || (phi!=0.0)) rotate(x,y,z);
    return fxyzwave(x,y,z);
  case fyz2wavetype :
    shift(x,y,z);
    if(basisvectors.size()) transform(x,y);
    if((theta!=0.0) || (phi!=0.0)) rotate(x,y,z);
    return fyz2wave(x,y,z);
  case fz3wavetype :
    shift(x,y,z);
    if(basisvectors.size()) transform(x,y);
    if((theta!=0.0) || (phi!=0.0)) rotate(x,y,z);
    return fz3wave(x,y,z);
  case fxz2wavetype :
    shift(x,y,z);
    if(basisvectors.size()) transform(x,y);
    if((theta!=0.0) || (phi!=0.0)) rotate(x,y,z);
    return fxz2wave(x,y,z);
  case fzx2wavetype :
    shift(x,y,z);
    if(basisvectors.size()) transform(x,y);
    if((theta!=0.0) || (phi!=0.0)) rotate(x,y,z);
    return fzx2wave(x,y,z);
  case fxx2wavetype :
    shift(x,y,z);
    if(basisvectors.size()) transform(x,y);
    if((theta!=0.0) || (phi!=0.0)) rotate(x,y,z);
    return fxx2wave(x,y,z);
  case nsymmwavetype :
    shift(x,y,z);
    if(basisvectors.size()) transform(x,y);
    if((theta!=0.0) || (phi!=0.0)) rotate(x,y,z);
    return nsymmwave(x,y,z);
  case cubetype:
    //if(wanfile) return wanfile->getvalue(x,y,zheight);
    //else
    return NAN;
  case zerotype:
    return zerowave(x,y,z);
  }
  return NAN;
}

double wannierfunctions::getwave(double x,double y,double z) {
  switch(type) {
  case swavetype :
    shift(x,y,z);
    if(basisvectors.size()) transform(x,y);
    return swave(x,y,z);
  case pxwavetype :
    shift(x,y,z);
    if(basisvectors.size()) transform(x,y);
    if((theta!=0.0) || (phi!=0.0)) rotate(x,y,z);    
    return pxwave(x,y,z);
  case pywavetype :
    shift(x,y,z);
    if(basisvectors.size()) transform(x,y);
    if((theta!=0.0) || (phi!=0.0)) rotate(x,y,z);
    return pywave(x,y,z);
  case pzwavetype :
    shift(x,y,z);
    if(basisvectors.size()) transform(x,y);
    if((theta!=0.0) || (phi!=0.0)) rotate(x,y,z);
    return pzwave(x,y,z);
  case dx2wavetype :
    shift(x,y,z);
    if(basisvectors.size()) transform(x,y);
    if((theta!=0.0) || (phi!=0.0)) rotate(x,y,z);
    return dx2wave(x,y,z);
  case dxywavetype :
    shift(x,y,z);
    if(basisvectors.size()) transform(x,y);
    if((theta!=0.0) || (phi!=0.0)) rotate(x,y,z);
    return dxywave(x,y,z);
  case dxzwavetype :
    shift(x,y,z);
    if(basisvectors.size()) transform(x,y);
    if((theta!=0.0) || (phi!=0.0)) rotate(x,y,z);
    return dxzwave(x,y,z);
  case dyzwavetype :
    shift(x,y,z);
    if(basisvectors.size()) transform(x,y);
    if((theta!=0.0) || (phi!=0.0)) rotate(x,y,z);
    return dyzwave(x,y,z);
  case dr2wavetype :
    shift(x,y,z);
    if(basisvectors.size()) transform(x,y);
    if((theta!=0.0) || (phi!=0.0)) rotate(x,y,z);
    return dr2wave(x,y,z);
  case fy3x2wavetype :
    shift(x,y,z);
    if(basisvectors.size()) transform(x,y);
    if((theta!=0.0) || (phi!=0.0)) rotate(x,y,z);
    return fy3x2wave(x,y,z);
  case fxyzwavetype :
    shift(x,y,z);
    if(basisvectors.size()) transform(x,y);
    if((theta!=0.0) || (phi!=0.0)) rotate(x,y,z);
    return fxyzwave(x,y,z);
  case fyz2wavetype :
    shift(x,y,z);
    if(basisvectors.size()) transform(x,y);
    if((theta!=0.0) || (phi!=0.0)) rotate(x,y,z);
    return fyz2wave(x,y,z);
  case fz3wavetype :
    shift(x,y,z);
    if(basisvectors.size()) transform(x,y);
    if((theta!=0.0) || (phi!=0.0)) rotate(x,y,z);
    return fz3wave(x,y,z);
  case fxz2wavetype :
    shift(x,y,z);
    if(basisvectors.size()) transform(x,y);
    if((theta!=0.0) || (phi!=0.0)) rotate(x,y,z);
    return fxz2wave(x,y,z);
  case fzx2wavetype :
    shift(x,y,z);
    if(basisvectors.size()) transform(x,y);
    if((theta!=0.0) || (phi!=0.0)) rotate(x,y,z);
    return fzx2wave(x,y,z);
  case fxx2wavetype :
    shift(x,y,z);
    if(basisvectors.size()) transform(x,y);
    if((theta!=0.0) || (phi!=0.0)) rotate(x,y,z);
    return fxx2wave(x,y,z);
  case nsymmwavetype :
    shift(x,y,z);
    if(basisvectors.size()) transform(x,y);
    if((theta!=0.0) || (phi!=0.0)) rotate(x,y,z);
    return nsymmwave(x,y,z);
  case cubetype:
    //if(wanfile) return wanfile->getvalue(x,y,zheight);
    //else
    return NAN;
  case zerotype:
    return zerowave(x,y,z);
  }
  return NAN;
}

double wannierfunctions::getmax() {
  if(wfcube.size()!=0) {
    double maxval=0.0;
    for(size_t i=0;i<wfcube.size();i++)
      for(size_t j=0;j<wfcube[i].size();j++)
	if(abs(wfcube[i][j])>maxval)
	  maxval=abs(wfcube[i][j]);
    return maxval;
  } else {
    switch(type) {
    case swavetype : return swave(0.0,0.0,height);
    case pxwavetype : return pxwave(width/sqrt(2.0),0.0,height);
    case pywavetype : return pywave(0.0,width/sqrt(2.0),height);
    case pzwavetype : return pzwave(0.0,0.0,height);
    case dx2wavetype : return dx2wave(width,0.0,height);
    case dxywavetype : return dxywave(width/sqrt(2.0),width/sqrt(2.0),height);
    case dxzwavetype : return dxzwave(width/sqrt(2.0),0.0,height);
    case dyzwavetype : return dyzwave(0.0,width/sqrt(2.0),height);
    case dr2wavetype : return dr2wave(width,0.0,height);
      //need to fix these, currently approximate only
    case fy3x2wavetype : return fy3x2wave(-width,0.0,height);
    case fxyzwavetype : return fxyzwave(width/sqrt(2.0),width/sqrt(2.0),height);
    case fyz2wavetype : return fyz2wave(width/sqrt(2.0),0.0,height);
    case fz3wavetype : return fz3wave(0.0,0.0,height);
    case fxz2wavetype : return fxz2wave(0.0,-width/sqrt(2.0),height);
    case fzx2wavetype : return fzx2wave(0.0,width,height);
    case fxx2wavetype : return fxx2wave(0.0,-width,height);
    case nsymmwavetype : return nsymmwave(width,0.0,height);
    case cubetype : {
      double maxval=0.0;
      for(size_t i=0;i<wfcube.size();i++)
	for(size_t j=0;j<wfcube[i].size();j++)
	  if(abs(wfcube[i][j])>maxval)
	    maxval=abs(wfcube[i][j]);
      return maxval;
    }
    case zerotype : return 0.0;
    }
  }
  return NAN;
}

double wannierfunctions::getmaxboundary() {
  double maxval=0.0;
  if(wfcube.size()==0) return NAN;
  else {
    for(size_t i=0;i<wfcube.size();i++) {
      if(abs(wfcube[i][0])>maxval)
	maxval=abs(wfcube[i][0]);
      if(abs(wfcube[i][wfcube[i].size()-1])>maxval)
	maxval=abs(wfcube[i][wfcube[i].size()-1]);
    }
    for(size_t j=0;j<wfcube[0].size();j++) {
      if(abs(wfcube[0][j])>maxval)
	maxval=abs(wfcube[0][j]);
      if(abs(wfcube[wfcube.size()-1][j])>maxval)
	maxval=abs(wfcube[wfcube.size()-1][j]);
    }
  }
  return maxval;
}

double wannierfunctions::getrange(double p) {
  return width*sqrt(log(1.0/p));
}

string wannierfunctions::getname() {
  switch(type) {
  case swavetype : return "s";
  case pxwavetype : return "px";
  case pywavetype : return "py";
  case pzwavetype : return "pz";
  case dx2wavetype : return "dx2";
  case dxywavetype : return "dxy";
  case dxzwavetype : return "dxz";
  case dyzwavetype : return "dyz";
  case dr2wavetype : return "dr2";
  case fy3x2wavetype : return "fy3x2";
  case fxyzwavetype : return "fxyz";
  case fyz2wavetype : return "fyz2";
  case fz3wavetype : return "fz3";
  case fxz2wavetype : return "fxz2";
  case fzx2wavetype : return "fzx2";
  case fxx2wavetype : return "fxx2";
  case nsymmwavetype : return "n";
  case cubetype : return "cube";
  case zerotype : return "zero";
  }
  return "error";
}

bool wannierfunctions::settype(const string &name)
{
  if(name=="s") { type=swavetype; return true; }
  if(name=="px") { type=pxwavetype; return true; }
  if(name=="py") { type=pywavetype; return true; }
  if(name=="pz") { type=pzwavetype; return true; }
  if(name=="dx2") { type=dx2wavetype; return true; }
  if(name=="dxy") { type=dxywavetype; return true; }
  if(name=="dxz") { type=dxzwavetype; return true; }
  if(name=="dyz") { type=dyzwavetype; return true; }
  if(name=="dr2") { type=dr2wavetype; return true; }
  if(name=="fy3x2") { type=fy3x2wavetype; return true; }
  if(name=="fxyz") { type=fxyzwavetype; return true; }
  if(name=="fyz2") { type=fyz2wavetype; return true; }
  if(name=="fz3") { type=fz3wavetype; return true; }
  if(name=="fxz2") { type=fxz2wavetype; return true; }
  if(name=="fzx2") { type=fzx2wavetype; return true; }
  if(name=="fxx2") { type=fxx2wavetype; return true; }
  if(name=="n") { type=nsymmwavetype; return true; }
  if(name=="z") { type=zerotype; return true; }
  return false;
}

void wannierfunctions::precalculate(size_t window, size_t oversamp) 
{
  size_t n=(2*window+1)*oversamp;
  wfcube.resize(n);
  for(size_t i=0;i<n;i++)
    wfcube[i].assign(n,0.0);
  for(size_t i=0;i<n;i++)
    for(size_t j=0;j<n;j++)
      wfcube[i][j]=getwave((double)i/oversamp-(double)window-0.5,(double)j/oversamp-(double)window-0.5);
}

wannierfunctions::wannierfunctions(const string &filename,double height,size_t window, size_t oversamp,double prefact)
{
  error=true;
  type=cubetype;
  xcfloader *wanfile=new xcfloader(filename.c_str());
  if(!*wanfile)
    return;
  error=false;
  double zheight=wanfile->getzheight(height);
  size_t n=(2*window+1)*oversamp;
  wfcube.resize(n);
  for(size_t i=0;i<n;i++)
    wfcube[i].assign(n,0.0);
  for(size_t i=0;i<n;i++)
    for(size_t j=0;j<n;j++)
      wfcube[i][j]=prefact*wanfile->getvalue((double)i/oversamp-(double)window,(double)j/oversamp-(double)window,zheight);
  delete wanfile;
  wanfile=NULL;
}

wannierfunctions::wannierfunctions(const string &filename,vector<double> &shift,double height,size_t window, size_t oversamp,double prefact,bool centre)
{
  error=true;
  type=cubetype;
  xcfloader *wanfile=new xcfloader(filename.c_str());
  if(!*wanfile)
    return;
  error=false;
  wanfile->setshift(shift);
  double zheight=wanfile->getzheight(height);
  size_t n=(2*window+1)*oversamp;
  wfcube.resize(n);
  for(size_t i=0;i<n;i++)
    wfcube[i].assign(n,0.0);
  double xofs=0.0,yofs=0.0;
  if(centre) {
    xofs=0.5; yofs=0.5;
  }
  for(size_t i=0;i<n;i++)
    for(size_t j=0;j<n;j++)
      wfcube[i][j]=prefact*wanfile->getvalue((double)i/oversamp-(double)window-xofs,(double)j/oversamp-(double)window-yofs,zheight);
  delete wanfile;
  wanfile=NULL;
}

wannierfunctions::wannierfunctions(const vector<vector<double> > &arr,double prefact) {
  type=cubetype;
  size_t n=arr.size();
  wfcube.resize(n);
  for(size_t i=0;i<n;i++)
    wfcube[i].assign(n,0.0);
  for(size_t i=0;i<n;i++)
    for(size_t j=0;j<n;j++)
      wfcube[i][j]=prefact*arr[i][j];
}

wannierfunctions::wannierfunctions(vector<vector<double> > &ucv,xcfloader *wf,size_t window, size_t oversamp,double height,double xpos,double ypos,double theta,double prefact):height(height),xpos(xpos),ypos(ypos),theta(theta),prefact(prefact) {
  error=true;
  type=cubetype;
  if(!*wf)
    return;
  error=false;
  setbasisvectors(ucv); phi=0.0;
  double zpos=wf->getzheight(height); 
  size_t n=(2*window+1)*oversamp;
  wf->setrotation(theta);
  wfcube.resize(n);
  for(size_t i=0;i<n;i++)
    wfcube[i].assign(n,0.0);
  for(size_t i=0;i<n;i++)
    for(size_t j=0;j<n;j++) {
      double x=(double)i/oversamp-(double)window,y=(double)j/oversamp-(double)window,z=zpos;
      shift(x,y);
      if(basisvectors.size()) transform(x,y);
      wfcube[i][j]=prefact*wf->getcartmodvalue(x,y,z);
    }
}

bool operator!(wannierfunctions &wf) {return wf.error;}

double wannierfunctions::overlap(wannierfunctions &wf2,double dx,double dy,double dz,double maxrange,size_t grid) {
  double step=maxrange/grid,sum=0.0;
#pragma omp parallel for reduction (+:sum)
  for(size_t i=0;i<grid;i++)
    for(size_t j=0;j<grid;j++)
      for(size_t k=0;k<grid;k++) {
	double x=(double)i*step-maxrange/2.0,y=(double)j*step-maxrange/2.0,z=(double)k*step-maxrange/2.0;
	sum+=getwave(x-dx/2.0,y-dy/2.0,z-dz/2.0)*wf2.getwave(x+dx/2.0,y+dy/2.0,z+dz/2.0);
      }
  return sum*step*step*step;
}
