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

#ifndef _wannierfunctions_h
#define _wannierfunctions_h

#include <math.h>
#include <string>
#include <vector>
#include "loadxcf.h"

using namespace std;

enum wanniertypes {swavetype, pxwavetype, pywavetype, pzwavetype, dx2wavetype, dxywavetype, dr2wavetype, dxzwavetype, dyzwavetype, fy3x2wavetype, fxyzwavetype, fyz2wavetype, fz3wavetype, fxz2wavetype, fzx2wavetype, fxx2wavetype,nsymmwavetype, cubetype, zerotype};

class wannierfunctions {
  bool error;
  wanniertypes type;
  double width,height,xpos,ypos,theta,phi,prefact,symmetry;
  vector<vector<double> > wfcube;
  vector<vector<double> > basisvectors;
  void rotate(double &x, double &y) { double xc=x; x=xc*cos(theta)+y*sin(theta); y=-xc*sin(theta)+y*cos(theta); }
  void rotate(double &x, double &y,double &z) { double xc=x, yc=y, zc=z;
    x=xc*cos(theta)+yc*sin(theta)*cos(phi)+zc*sin(theta)*sin(phi);
    y=-xc*sin(theta)+yc*cos(theta)*cos(phi)+zc*cos(theta)*sin(phi);
    z=-sin(phi)*yc+cos(phi)*zc;
  }
  void shift(double &x, double &y) { x-=xpos; y-=ypos; }
  void shift(double &x, double &y,double &z) { x-=xpos; y-=ypos; z-=height; }
  void transform(double &x, double &y) {
    double a=x,b=y;
    x=basisvectors[0][0]*a+basisvectors[1][0]*b;
    y=basisvectors[0][1]*a+basisvectors[1][1]*b;
  }
  void setbasisvectors(vector<vector<double> > &ucv) {
    basisvectors.resize(ucv.size());
    for(size_t i=0;i<basisvectors.size();i++) {
      basisvectors[i].resize(ucv[i].size());
      for(size_t j=0;j<basisvectors.size();j++)
	basisvectors[i][j]=ucv[i][j];
    }
  }
public:
  wannierfunctions(wanniertypes type,double width,double height,double xpos=0.0,double ypos=0.0,double theta=0.0,double phi=0.0,double prefact=1.0,double symmetry=4.0): error(false),type(type),width(width),height(height),xpos(xpos), ypos(ypos), theta(theta),phi(phi),prefact(prefact),symmetry(symmetry) {}
  wannierfunctions(const string &name,double width,double height,double xpos=0.0,double ypos=0.0,double theta=0.0,double phi=0.0,double prefact=1.0,double symmetry=4.0): error(false),width(width),height(height),xpos(xpos), ypos(ypos), theta(theta),phi(phi),prefact(prefact),symmetry(symmetry) { settype(name); }
  wannierfunctions(vector<vector<double> > &ucv,wanniertypes type,double width,double height,double xpos=0.0,double ypos=0.0,double theta=0.0,double phi=0.0,double prefact=1.0,double symmetry=4.0): error(false),type(type),width(width),height(height),xpos(xpos), ypos(ypos), theta(theta),phi(phi),prefact(prefact),symmetry(symmetry) {setbasisvectors(ucv);}
  wannierfunctions(vector<vector<double> > &ucv,const string &name,double width,double height,double xpos=0.0,double ypos=0.0,double theta=0.0,double phi=0.0,double prefact=1.0,double symmetry=4.0): error(false),width(width),height(height),xpos(xpos), ypos(ypos), theta(theta),phi(phi),prefact(prefact),symmetry(symmetry) { settype(name); setbasisvectors(ucv); }
  wannierfunctions(vector<vector<double> > &ucv,xcfloader *wf,size_t window,size_t oversamp,double height,double xpos=0.0,double ypos=0.0,double theta=0.0, double prefact=1.0);
  wannierfunctions(const string &filename,double height, size_t window, size_t oversamp, double prefact=1.0);
  wannierfunctions(const string &filename,vector<double> &shift,double height, size_t window, size_t oversamp, double prefact=1.0,bool centre=true);
  wannierfunctions(const vector<vector<double> > &arr,double prefact=1.0);
  void precalculate(size_t window, size_t oversamp);
  double swave(double x, double y, double z);
  double pxwave(double x, double y, double z);
  double pywave(double x, double y, double z);
  double pzwave(double x, double y, double z);
  double dx2wave(double x, double y, double z);
  double dxywave(double x, double y, double z);
  double dxzwave(double x, double y, double z);
  double dyzwave(double x, double y, double z);
  double dr2wave(double x, double y, double z);
  double fy3x2wave(double x, double y, double z);
  double fxyzwave(double x, double y, double z);
  double fyz2wave(double x, double y, double z);
  double fz3wave(double x, double y, double z);
  double fxz2wave(double x, double y, double z);
  double fzx2wave(double x, double y, double z);
  double fxx2wave(double x, double y, double z);
  double nsymmwave(double x, double y, double z);
  double zerowave (double x, double y, double z);
  void settype(wanniertypes stype) {type=stype;}
  bool settype(const string &name);
  string getname();
  double getwidth() {return width;}
  double getrange(double p);
  void getpos(double &x, double &y) {x=xpos; y=ypos;}
  void getpos(double &x, double &y,double &h) {x=xpos; y=ypos; h=height;}
  double gettheta() {return theta;}
  double getphi() {return phi;}
  double getwave(double x,double y);
  double getwave(double x,double y,double z);
  double getwave_cached(size_t i,size_t j) {return wfcube[i][j]; }
  void getcachesize(size_t &xs, size_t &ys) { xs=wfcube.size(); ys=wfcube[0].size(); }
  double getmax();
  double getmaxboundary();
  bool iszero() {return (type==zerotype);}
  double overlap(wannierfunctions &wf2,double dx,double dy,double dz,double maxrange=7.0,size_t grid=100);
  friend bool operator!(wannierfunctions &wf);
};

#endif
