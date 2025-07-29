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

#ifndef _idl_h
#define _idl_h

#include <math.h>
#include <fftw3.h>
#include <string>
#include <iostream>
#include <vector>
#include <limits>
#include <complex>

using namespace std;

#ifndef DBL_MAX
const long double DBL_MAX=numeric_limits<double>::max();
#endif

#ifndef NAN
#define NAN FP_NAN
#endif

typedef vector<double> dblval;
typedef vector<dblval> dblcol;

//workaround for integration.h ...
#ifndef _integration_h
typedef vector<dblcol> dblarr;
#endif

///Main data description class of the IDL STM routines
class idl {
  bool headeronly,     //!<true: only header is loaded, data array remains empty
    failed;            //!<true: load failed
 protected:
  string name,         //!<Original file name with path or description of current data set
    date;              //!<Date of data creation or modification (in case of processing)
  size_t size[3];      //!<Dimensions of data array
  double scan_range[2],//!<Scan range of data set in AA
    offset[2],         //!<Offset of map (position of center, preferably in AA)
    junction_bias[2],  //!<Bias range (twice the same value for topos)
    setpoint;          //!<Setpoint current
  dblarr data;         //!<data array
  size_t fftsize,      //!<size of array holding FFT (0 as long as no FFT has been calculated)
    fftslice,          //!<layer from which FFT has been calculated (for backtransform)
    fftn;              //!<length of one row of FFT
  complex<double> *fftdata; //!<array holding FFT values
 public:
  void setbiasrange(double lb,double hb);
  void getbiasrange(double &lb,double &hb);
 protected:
  void alloc(dblarr &arr,double init=0.0);
  void fftalloc();
  void cfftalloc();
  void read(istream &is);
  void write(ostream &os);
  ///Sets headeronly to true, meaning that following read operations will only load the header but not the data
  void setheaderonly() { headeronly=true; }
 public:
  idl();
  idl(istream &is, bool h=false);
  idl(const char *name, bool h=false);
  idl(int x,int y,int z,double xd=1.0,double yd=1.0,double lb=0.0,double hb=DBL_MAX,double sp=0.0,double xp=0.0,double yp=0.0);
  void open(const char *name);
  void open(istream &is);
  void fft(size_t slice=0);
  void cfft(size_t slice=0,bool overwrite=false);
  ///get k-vector corresponding to index x
  ///@param[in] x x-index
  ///@return corresponding k-vector of FFT
  ///@todo join with getky
  double getkx(size_t x) { return ((x>(size[0]>>1))?((double)x-size[0])*2.0*M_PI/scan_range[0]:x*2.0*M_PI/scan_range[0]); }
  ///get k-vector corresponding to index y
  ///@param[in] y y-index
  ///@return corresponding k-vector of FFT
  ///@todo join with getkx
  double getky(size_t y) { return (y*2.0*M_PI/scan_range[1]); }
  //    size_t getindkx(double kx) { int index=(int)(kx*scan_range[1]/2.0/M_PI);
  //    return (index>0)?(size_t)index:(size_t)(index+size[1]); }
  //    size_t getindky(double ky) { return (size_t)(ky*scan_range[1]/2.0/M_PI); }
  ///get value of fft at (x,y)
  ///no range checking, use only if FFT has been performed (not checked)
  ///@param[in] x x-index
  ///@param[in] y y-index
  ///@return complex value at (x,y)
  ///@todo range checking and check validity of fftdata
  complex<double> getfft(size_t x,size_t y) { return fftdata[x*fftn+y]; }
  ///set value of fft at (x,y)
  ///no range checking, use only if FFT has been performed (not checked)
  ///@param[in] x x-index
  ///@param[in] y y-index
  ///@param[in] v complex value
  ///@todo range checking and check validity of fftdata
  void setfft(size_t x,size_t y,complex<double> v) { fftdata[x*fftn+y]=v; }
  void ifft();
  void icfft(size_t slice=0,bool inplace=false);
  void correlate(idl &idlfile,size_t slice=0,size_t slice2=0,bool norm=false);
  void autocorrelate(size_t slice=0,bool norm=false);
  ///get number of slice of which the FFT is stored in fftdata
  ///@return index of slice
  size_t getfftslice() {return fftslice;}
  ///check whether fftdata points to valid data
  ///@return true, if fftdata contains an array, false if not
  bool isfft() { return (fftdata!=NULL); }
  ///get dimensions of FFT array
  ///@param[out] kx size in x-direction
  ///@param[out] ky size in y-direction
  void fftdimensions(size_t &kx,size_t &ky) { kx=size[0]; ky=fftn; } 
  void setname(string nname);
  string getname();
  string getdate();
  void fill(double val,size_t k=0);
  void init(double val=0.0);
  void set(size_t i,size_t j,double val);
  void set(size_t i,size_t j,size_t k,double val);
  void set(double x,double y,double val);
  void set(double x,double y,double bias,double val);
  void add(idl &map);
  double getrelx(int i);
  double getrely(int i);
  double getrel(size_t dim, int i);
  double getx(int i);
  double gety(int i);
  double get(size_t dim, int i);
  double getidx(double x);
  double getidy(double y);
  double getid(size_t dim, double x);
  size_t getix(double x);
  size_t getiy(double y);
  size_t geti(size_t dim,double x);
  double getbias(int i);
  int getbiasindex(double bias);
  double max(size_t &x,size_t &y);
  double min(size_t &x,size_t &y);
  double min(size_t slice=0);
  double max(size_t slice=0);
  double max_pos(size_t x,size_t y,bool bias=false);
  double min_pos(size_t x,size_t y,bool bias=false);
  double average(size_t k=0);
  double averagesqr(size_t k=0);
  double median(size_t k=0);
  double mode(size_t k,size_t bins=10000);
  double average(size_t x,size_t y);
  void dimensions(size_t &xs,size_t &ys);
  void dimensions(size_t sa[3]);
  ///return number of layers (2nd dimension) of IDL object
  ///@return number of layers
  ///@todo use generalized coordinates
  size_t layers() { return size[2]; }
  int cut(size_t x0,size_t y0,size_t xs,size_t ys);
  int cut(size_t x0,size_t y0,size_t z0,size_t xs,size_t ys,size_t zs);
  int extend(size_t x0,size_t y0,size_t xns,size_t yns);
  int extend(size_t x0,size_t y0,size_t z0,size_t xns,size_t yns,size_t zns);
  bool rangecheck(size_t dim,int &l);
  void shift(int sx,int sy);
  void plane_compensation(size_t x0=0,size_t y0=0,size_t xs=0,size_t ys=0,size_t k=0);
  void removeconstant();
  void removeconstant(size_t slice);
  void lfilter2d(dblcol &filter,bool ignorenan=false,bool normalize=false);
  void lfilter2d(dblcol &filter,size_t dim0, size_t dim1,bool ignorenan=false,bool normalize=false);
  void lfilter2dslice(dblcol &filter,size_t slice,bool ignorenan=false,bool normalize=false);
  void lfilter(dblarr &filter,bool ignorenan=false,bool normalize=false);
  void median2d(int n);
  void median(int nx,int ny,int nz);
  void destripe(size_t slice);
  void differentiate(size_t dimension=2);
  void integrate(size_t dimension=2);
  void getdata(double *arr);
  void setdata(double *arr);
  ///obtain pointer to FFT data (deprecated API)
  ///@return pointer to complex array
  complex<double> *getfftarray() {return fftdata;} //Ohlala !
  ///get size of FFT array
  ///@return size of FFT array (number of complex values)
  size_t getfftsize() {return fftsize;}
  void putfft(idl& fftobj);
  ~idl(); 
  friend bool operator>>(idl& i,const char *name);
  friend bool operator>>(idl& i,ostream &os);
  friend bool operator<<(idl& i,const char *name);
  friend bool operator<<(idl& i,istream &is);
  bool check(double x,double y);
  bool check(double x,double y,double bias);
  bool checki(size_t i,size_t j,size_t k=0);
  void duplicate(size_t slice=0);
  ///get value at indices (x,y,z)
  ///@param[in] x x-index
  ///@param[in] y y-index
  ///@param[in] z z-index
  double operator() (size_t x,size_t y,size_t z=0) { return data[x][y][z]; }
  void normalize(double x,size_t slice);
  void normalize(double x);
  friend bool operator!(idl &i);
};

bool operator>>(idl& i,const char *name);
bool operator<<(idl& i,const char *name);
#endif
