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

#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <sstream>

#include "idl.h"
#include "loadxcf.h"

#include <gsl/gsl_linalg.h>

using namespace std;

void xcfloader::alloc() {
  cube.resize(xs);
  for(size_t i=0;i<xs;i++) {
    cube[i].resize(ys);
    for(size_t j=0;j<ys;j++)
      cube[i][j].resize(zs);
  }
}

void xcfloader::readfile(istream &is) {
  size_t lineno=0;
  crystal=false;
  while(!is.eof()) {
    string line;     
    getline(is,line);
    if(!is.eof()) {
      lineno++;
      size_t hashtag;
      hashtag=line.find('#');
      if(hashtag!=string::npos) {
	comments.push_back(line.substr(hashtag+1));
	continue;
      }
      if(line.find("CRYSTAL")!=string::npos)
	crystal=true;
      else if(crystal && (line.find("PRIMVEC")!=string::npos)) {
	for(size_t i=0;i<3;i++) {
	  string parline;
	  getline(is,parline); lineno++;
	  istringstream iss(parline);
	  double x,y,z;
	  iss>>x>>y>>z;
	  vector<double> coords;
	  coords.push_back(x);
	  coords.push_back(y);
	  coords.push_back(z);
	  primvec.push_back(coords);
	}	    
      } else if(crystal && (line.find("CONVVEC")!=string::npos)) {
	for(size_t i=0;i<3;i++) {
	  string parline;
	  getline(is,parline); lineno++;
	  istringstream iss(parline);
	  double x,y,z;
	  iss>>x>>y>>z;
	  vector<double> coords;
	  coords.push_back(x);
	  coords.push_back(y);
	  coords.push_back(z);
	  convvec.push_back(coords);
	}
      } else if(crystal && (line.find("PRIMCOORD")!=string::npos)) {
	size_t atomnum, type;
	string parline;
	getline(is,parline); lineno++;
	istringstream iss(parline);
	iss>>atomnum>>type;
	if(type!=1) ExecuteCPU0 cerr<<"Line "<<lineno<<": Invalid value in PRIMCOORD ("<<type<<")"<<endl;
	for(size_t i=0;i<atomnum;i++) {
	  getline(is,parline); lineno++;
	  iss.str(parline); iss.clear();
	  string name;
	  double x,y,z;
	  vector<double>  coords;
	  iss>>name>>x>>y>>z;
	  coords.push_back(x);
	  coords.push_back(y);
	  coords.push_back(z);
	  atomcoords.push_back(coords);
	  atoms.push_back(name);
	}
      } else if(line.find("BEGIN_BLOCK_DATAGRID_3D")!=string::npos) {
	string keyword;
	string parline;
	getline(is,parline); lineno++;
	istringstream iss(parline);
	iss>>keyword;
	if(keyword!="3D_field") {
	  ExecuteCPU0 cerr<<"Line "<<lineno<<": Unknown data type ("<<keyword<<")"<<endl;
	  error=true;
	}
	getline(is,parline); lineno++;
	iss.str(parline); iss.clear();
	iss>>keyword;
	if(keyword!="BEGIN_DATAGRID_3D_UNKNOWN")
	  {
	    ExecuteCPU0 cerr<<"Line "<<lineno<<": Unknown data grid identifier ("<<keyword<<")"<<endl;
	    error=true;
	  }
	getline(is,parline); lineno++;
	iss.str(parline); iss.clear();
	iss>>xs>>ys>>zs;
	alloc();
	double x,y,z;
	getline(is,parline); lineno++;
	iss.str(parline); iss.clear();
	iss>>x>>y>>z;
	origin.push_back(x);
	origin.push_back(y);
	origin.push_back(z);
	for(size_t i=0;i<3;i++) {
	  vector<double> coords;
	  getline(is,parline); lineno++;
	  iss.str(parline); iss.clear();
	  iss>>x>>y>>z;
	  coords.push_back(x);
	  coords.push_back(y);
	  coords.push_back(z);
	  cubeaxis.push_back(coords);
	}
	//size_t total=xs*ys*zs;
	//for(size_t n=0;n<total;n++); {
	//  double x;
	//  is>>xx;
	ExecuteCPU0 cout<<"Loading data cube with "<<xs<<"x"<<ys<<"x"<<zs<<" ..."<<endl;
	getline(is,line); lineno++;
	size_t pos=0,npos=0;
	for(size_t k=0;k<zs;k++)
	  for(size_t j=0;j<ys;j++)
	    for(size_t i=0;i<xs;i++) {
	      bool reload=false;
	      pos=line.find_first_of("+-0123456789",npos);
	      if(pos!=string::npos) {
		npos=line.find_first_of(" \t",pos);
		size_t len;
		if(npos!=string::npos)
		  len=npos-pos;
		else {
		  len=line.length()-pos;
		  reload=true;
		}
		istringstream niss(line.substr(pos,len));
		niss>>cube[i][j][k];
	      } else {
		ExecuteCPU0 cerr<<"Line "<<lineno<<": Cannot parse line ("<<line<<")."<<endl;
		error=true;
		return;
	      }
	      if(reload) {
		getline(is,line); lineno++;
		npos=0; pos=0;
		reload=false;
	      }
	    }
	if(line.find("END_DATAGRID_3D")==string::npos) {
	  ExecuteCPU0 cerr<<"Line "<<lineno<<": Cannot parse line ("<<line<<")."<<endl;
	  error=true;
	}
	getline(is,line); lineno++;
	if(line.find("END_BLOCK_DATAGRID_3D")==string::npos) {
	  ExecuteCPU0 cerr<<"Line "<<lineno<<": Cannot parse line ("<<line<<")."<<endl;
	  error=true;
	}
      }
    }
  }
  ExecuteCPU0 cout<<"Read "<<lineno<<" lines from file."<<endl;
}


xcfloader::xcfloader(const char *name) {
  error=false;
  ifstream is(name);
  if(!is) {
    ExecuteCPU0 cerr<<"Could not open file "<<name<<endl;
    error=true;
  } else
    readfile(is);
  m=gsl_matrix_alloc(3,3);
  p=gsl_permutation_alloc(3);
  for(size_t i=0;i<3;i++)
    for(size_t j=0;j<3;j++)
      gsl_matrix_set(m,j,i,cubeaxis[i][j]);
  gsl_linalg_LU_decomp(m,p,&signum);
  shift.resize(3,0.0);
  cosphi=1.0; sinphi=0.0;
}

void xcfloader::showcomments(ostream &os) {
  for(size_t i=0;i<comments.size();i++)
    os<<"#"<<comments[i]<<endl;
}

void xcfloader::writeidl(const char *name) {
  idl cubefile(xs,ys,zs);
  for(size_t i=0;i<xs;i++)
    for(size_t j=0;j<ys;j++)
      for(size_t k=0;k<zs;k++)
	cubefile.set(i,j,k,cube[i][j][k]);
  cubefile>>name;
}

double xcfloader::operator()(double x, double y, double z) {
  gsl_vector *res=gsl_vector_alloc(3),
    *b=gsl_vector_alloc(3);
  gsl_vector_set(b,0,x-origin[0]-shift[0]);
  gsl_vector_set(b,1,y-origin[1]-shift[1]);
  gsl_vector_set(b,2,z-origin[2]-shift[2]);
  gsl_linalg_LU_solve(m,p,b,res);
  int xi=round(gsl_vector_get(res,0)*xs),
    yi=round(gsl_vector_get(res,1)*ys),
    zi=round(gsl_vector_get(res,2)*zs);
  gsl_vector_free(b);
  gsl_vector_free(res);
  if((xi>=0)&&(yi>=0)&&(zi>=0)
     &&((size_t)xi<xs)&&((size_t)yi<ys)&&((size_t)zi<zs))
    return cube[xi%xs][yi%ys][zi%zs];
  else return 0.0;
}

double xcfloader::getmodvalue(double x, double y, double z) {
  gsl_vector *res=gsl_vector_alloc(3),
    *b=gsl_vector_alloc(3);
  double xu=x,yu=y;
  x=xu*convvec[0][0]+yu*convvec[1][0];
  y=xu*convvec[0][1]+yu*convvec[1][1];
  
  double xb=x;
  x=x*cosphi-y*sinphi;
  y=xb*sinphi+y*cosphi;
  
  double xc=x-origin[0]-shift[0],
    yc=y-origin[1]-shift[1],
    zc=z-origin[2]-shift[2];
  
  gsl_vector_set(b,0,xc);
  gsl_vector_set(b,1,yc);
  gsl_vector_set(b,2,zc);
  gsl_linalg_LU_solve(m,p,b,res);
  int xi=round(gsl_vector_get(res,0)*xs),
    yi=round(gsl_vector_get(res,1)*ys),
    zi=round(gsl_vector_get(res,2)*zs);
  gsl_vector_free(b);
  gsl_vector_free(res);
  if((xi>=0)&&(yi>=0)&&(zi>=0)
     &&((size_t)xi<xs)&&((size_t)yi<ys)&&((size_t)zi<zs))
    return cube[xi][yi][zi];
  else return 0.0;
}

double xcfloader::getcartmodvalue(double x, double y, double z) {
  gsl_vector *res=gsl_vector_alloc(3),
    *b=gsl_vector_alloc(3);
 
  double xb=x;
  x=x*cosphi-y*sinphi;
  y=xb*sinphi+y*cosphi;
  
  double xc=x-origin[0]-shift[0],
    yc=y-origin[1]-shift[1],
    zc=z-origin[2]-shift[2];
  
  gsl_vector_set(b,0,xc);
  gsl_vector_set(b,1,yc);
  gsl_vector_set(b,2,zc);
  gsl_linalg_LU_solve(m,p,b,res);
  int xi=round(gsl_vector_get(res,0)*xs),
    yi=round(gsl_vector_get(res,1)*ys),
    zi=round(gsl_vector_get(res,2)*zs);
  gsl_vector_free(b);
  gsl_vector_free(res);
  if((xi>=0)&&(yi>=0)&&(zi>=0)
     &&((size_t)xi<xs)&&((size_t)yi<ys)&&((size_t)zi<zs))
    return cube[xi][yi][zi];
  else return 0.0;
}

double xcfloader::operator*(xcfloader &xcf2) {
  double val=0.0;
  for(size_t i=0;i<xs;i++)
    for(size_t j=0;j<ys;j++)
      for(size_t k=0;k<zs;k++)
	val+=cube[i][j][k]*xcf2.cube[i][j][k];
  return val;
}

double xcfloader::overlap(xcfloader &xcf2,double dx,double dy,double dz) {
  gsl_vector *res=gsl_vector_alloc(3),
    *b=gsl_vector_alloc(3);
  gsl_vector_set(b,0,dx);
  gsl_vector_set(b,1,dy);
  gsl_vector_set(b,2,dz);
  gsl_linalg_LU_solve(m,p,b,res);
  int xd=round(gsl_vector_get(res,0)*xs),
    yd=round(gsl_vector_get(res,1)*ys),
    zd=round(gsl_vector_get(res,2)*zs);
  gsl_vector_free(b);
  gsl_vector_free(res);
  double val=0.0;
  size_t num=0;
  for(size_t i=0;i<xs;i++)
    for(size_t j=0;j<ys;j++)
      for(size_t k=0;k<zs;k++)
	if(((i+xd)>=0)&&((i+xd)<xs)
	   &&((j+yd)>=0)&&((j+yd)<ys)
	   &&((k+zd)>=0)&&((k+zd)<zs)) {
	  val+=cube[i][j][k]*xcf2.cube[i+xd][j+yd][k+zd];
	  num++;
	}
  return val/(double)num;
}

double xcfloader::getvalue(double xu,double yu,double height) {
  return this->operator()(xu*convvec[0][0]+yu*convvec[1][0],xu*convvec[0][1]+yu*convvec[1][1],height);
}

double xcfloader::getzheight(double height) {
  if(height>0.0) {
    double maxh=-DBL_MAX;
    for(size_t i=0;i<atomcoords.size();i++)
      if(atomcoords[i][2]>maxh)
	maxh=atomcoords[i][2];
    return maxh+height;
  } else if(height<0.0) {
    double minh=DBL_MAX;
    for(size_t i=0;i<atomcoords.size();i++)
      if(atomcoords[i][2]<minh)
	minh=atomcoords[i][2];
    return minh+height;
  } else return NAN;
}

bool operator!(xcfloader &lf) {return lf.error;}
