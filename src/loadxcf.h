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

#ifndef _xcfloader_h
#define _xcfloader_h

#include "mpidefs.h"

#include <math.h>
#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <sstream>

#include <gsl/gsl_matrix.h>
#include <gsl/gsl_permutation.h>

using namespace std;

class xcfloader {
  size_t xs,ys,zs;
  vector<string> comments;
  vector<vector<double> > primvec, convvec, atomcoords,cubeaxis;
  vector<double> origin;
  vector<double> shift;
  vector<string> atoms;
  double cosphi,sinphi;
  vector<vector<vector<double> > > cube;
  bool error,crystal;
  gsl_matrix *m;
  gsl_permutation *p;
  int signum;
  void alloc();
  void readfile(istream &is);
  
public:
  xcfloader(const char *name);
  friend bool operator!(xcfloader &lf);
  void showcomments(ostream &os);
  void writeidl(const char *name);
  double operator()(double x, double y, double z);
  double operator*(xcfloader &xcf2);
  double overlap(xcfloader &xcf2,double dx,double dy,double dz);
  double getvalue(double xu, double yu, double zheight);
  double getmodvalue(double x, double y, double z);
  double getcartmodvalue(double x, double y, double z);
  double getzheight(double height);
  void setshift(double x,double y,double z) { shift[0]=x; shift[1]=y; shift[2]=z; }
  void setshift(vector<double> &nshift) { shift=nshift; shift.resize(3,0.0); }
  void setrotation(double phi) { cosphi=cos(phi); sinphi=sin(phi); }
  vector<double> getatomcoords(size_t n) {return atomcoords[n];}
  ~xcfloader() {
    gsl_permutation_free(p);
    gsl_matrix_free(m);
  }
};
  
bool operator!(xcfloader &lf);

#endif
