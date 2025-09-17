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

#include <string>
#include <iostream>
#include <fstream>
#include <math.h>
#include <fftw3.h>
#include <time.h>
#include <algorithm>

#include "idl.h"

///allocate memory for data array initialized with a value, 
///dimensions stored in \a size are used
///@param[out] arr   reference to array which is initialized
///@param[in]  init  value used for initialization
void idl::alloc(dblarr &arr,double init)
{
    arr.resize(size[0]);
    for(size_t i=0;i<size[0];i++) {
	arr[i].resize(size[1]);
	for(size_t j=0;j<size[1];j++)
	    arr[i][j].assign(size[2],init);
    }
}

///initialize data array with constant 
///@param[in]  init  value used for initialization
void idl::init(double val)
{
  for(size_t i=0;i<size[0];i++)
    for(size_t j=0;j<size[1];j++)
      std::fill(data[i][j].begin(), data[i][j].end(), val);
}

///read data file from input stream
///@param[in] is  input stream
void idl::read(istream &is)
{
    float dummy;
    //   if(!is) return;
    is.exceptions(istream::eofbit|istream::failbit|istream::badbit);
    try {
        getline(is,name);
	getline(is,date);
	is>>size[0]>>size[1]>>size[2]>>scan_range[0]>>scan_range[1]>>offset[0]>>offset[1]>>junction_bias[0]>>junction_bias[1]>>setpoint;
	is.ignore(1,'\n');
	if(!headeronly) {
	    alloc(data);
	    for(size_t k=0;k<size[2];k++)
		for(size_t j=0;j<size[1];j++)
		    for(size_t i=0;i<size[0];i++) {
			is.read((char *)&dummy,sizeof(float));
			data[i][j][k]=(double)dummy;
		    }
	}
    } catch (istream::failure const &e) {
	cerr<<"Error reading from stream."<<endl
	    <<e.what()<<endl;
	failed=true;
    }
}

///write data file to output stream
///@param[out] os  output stream
void idl::write(ostream &os)
{
    float dummy;
    os.exceptions(ostream::eofbit|ostream::failbit|ostream::badbit);
    try {
	os<<name<<endl<<date<<endl<<size[0]<<endl<<size[1]<<endl<<size[2]<<endl<<scan_range[0]<<endl<<scan_range[1]<<endl<<offset[0]<<endl<<offset[1]<<endl<<junction_bias[0]<<endl<<junction_bias[1]<<endl<<setpoint<<endl;
	for(size_t k=0;k<size[2];k++)
	    for(size_t j=0;j<size[1];j++)
		for(size_t i=0;i<size[0];i++) { 
		    dummy=data[i][j][k];
		    os.write((char *)&dummy,sizeof(float));
		}
    } catch (ostream::failure const &e) {
	cerr<<"Error writing to stream."<<endl
	    <<e.what()<<endl;
	failed=true;
    }
}

///allocate memory for FFT in \a fftdata, 
///initializes \a fftdata, \a fftn, \a fftsize and \a fftslice    
void idl::fftalloc()
{
    if(fftdata!=NULL) delete fftdata;
    fftn=(size[1]>>1)+1;
    fftsize=size[0]*fftn;
    fftdata=new complex<double>[fftsize];
    fftslice=0;
}

///allocate memory for complex FFT in \a fftdata, 
///initializes \a fftdata, \a fftn, \a fftsize and \a fftslice    
void idl::cfftalloc()
{
    if(fftdata!=NULL) delete fftdata;
    fftn=size[1];
    fftsize=size[0]*size[1];
    fftdata=new complex<double>[fftsize];
    fftslice=0;
}

///calculates FFT of data
///(be aware: FFT is not NAN safe)
///stores FFT of data in \a fftdata
///@param[in] slice  layer which is transformed
void idl::fft(size_t slice)
{
    fftalloc();
    fftslice=slice;
    fftw_plan myplan;
    double normconst=size[0]*size[1]; 
    myplan=fftw_plan_dft_r2c_2d(size[0],size[1],reinterpret_cast<double*>(fftdata),reinterpret_cast<fftw_complex*>(fftdata),0);
    for(size_t i=0;i<fftsize;i++)
	if((((i%fftn)<<1)+1)<size[1])
	    fftdata[i]=complex<double>(data[i/fftn][(i%fftn)<<1][fftslice],data[i/fftn][((i%fftn)<<1)+1][fftslice])/normconst;
	else if(((i%fftn)<<1)<size[1])
	    fftdata[i]=complex<double>(data[i/fftn][(i%fftn)<<1][fftslice],0.0)/normconst;
    fftw_execute(myplan);
    fftw_destroy_plan(myplan);
}

///calculates inverse FFT of data
///(FFT is not NAN safe)
///takes FFT of data from \a fftdata and stores the result in layer \a fftslice
void idl::ifft()
{
  fftw_plan myplan;
  myplan=fftw_plan_dft_c2r_2d(size[0],size[1],reinterpret_cast<fftw_complex*>(fftdata),reinterpret_cast<double*>(fftdata),FFTW_ESTIMATE);
  fftw_execute(myplan);
  //  double normconst=size[0]*size[1];
  for(size_t i=0;i<fftsize;i++) {
    if(((i%fftn)<<1)<size[1]) data[i/fftn][(i%fftn)<<1][fftslice]=fftdata[i].real();///normconst;
    if((((i%fftn)<<1)+1)<size[1]) data[i/fftn][((i%fftn)<<1)+1][fftslice]=fftdata[i].imag();///normconst;
  }
  delete fftdata;
  fftw_destroy_plan(myplan);
  fftdata=NULL;
}

///calculates FFT of complex data
///(be aware: FFT is not NAN safe)
///stores FFT of data in \a fftdata
///@param[in] slice  layer which is transformed
void idl::cfft(size_t slice,bool overwrite)
{
    cfftalloc();
    fftslice=slice;
    fftw_plan myplan;
    double normconst=sqrt((double)size[0]*size[1]); 
    myplan=fftw_plan_dft_2d(size[0],size[1],reinterpret_cast<fftw_complex*>(fftdata),reinterpret_cast<fftw_complex*>(fftdata),FFTW_FORWARD,FFTW_ESTIMATE);
    for(size_t i=0;i<size[0];i++)
      for(size_t j=0;j<size[1];j++)
	fftdata[i+j*size[0]]=complex<double>(data[i][j][fftslice],data[i][j][fftslice+1])/normconst;
    fftw_execute(myplan);
    fftw_destroy_plan(myplan);
    if(overwrite) {
      for(size_t i=0;i<size[0];i++)
	for(size_t j=0;j<size[1];j++) {
	  data[i][j][slice]=fftdata[i+j*size[0]].real();
	  data[i][j][slice+1]=fftdata[i+j*size[0]].imag();
	}
    }
}

///calculates inverse FFT of complex data
///(FFT is not NAN safe)
///takes FFT of data from \a fftdata and stores the result in layer \a fftslice
void idl::icfft(size_t slice, bool inplace)
{
  if(inplace) {
    cfftalloc();
    fftslice=slice;
    double normconst=sqrt((double)size[0]*size[1]);
    for(size_t i=0;i<size[0];i++)
      for(size_t j=0;j<size[1];j++)
	fftdata[i+j*size[0]]=complex<double>(data[i][j][slice],data[i][j][slice+1])/normconst;
  }
  fftw_plan myplan;
  myplan=fftw_plan_dft_2d(size[0],size[1],reinterpret_cast<fftw_complex*>(fftdata),reinterpret_cast<fftw_complex*>(fftdata),FFTW_BACKWARD,FFTW_ESTIMATE);
  fftw_execute(myplan);
  //  double normconst=size[0]*size[1];
  for(size_t i=0;i<size[0];i++)
    for(size_t j=0;j<size[1];j++){
      data[i][j][fftslice]=fftdata[i+j*size[0]].real();
      data[i][j][fftslice+1]=fftdata[i+j*size[0]].imag();
    }
  delete fftdata;
  fftw_destroy_plan(myplan);
  fftdata=NULL;
}

///calculate correlation between two data sets which have the same size
///(correlation is not NAN safe - inherited from FFT)
///@param[in] idlfile  reference to IDL class with which correlation is calculated (must not be the same, use copy to calculate correlation between different slices of one object)
///@param[in] slice    slice of \a this object which is used for correlation
///@param[in] slice2   slice of \a idlfile used for correlation
///@param[in] norm     if true, normalize correlation such that 1 is correlated and -1 anticorrelated
void idl::correlate(idl &idlfile,size_t slice,size_t slice2,bool norm) 
{
    double n1=0.0,n2=0.0;
    if(norm) {
	n1=sqrt(averagesqr(slice));
	n2=sqrt(idlfile.averagesqr(slice2));
    }
    fft(slice); idlfile.fft(slice2);
    complex<double> *a=fftdata,*b=idlfile.getfftarray();
    for(size_t i=0;i<fftsize;i++,a++,b++)
        *a=(*a)*conj(*b);
    ifft();
    if(norm)
      //normalize(1.0/(n1*n2)/size[0]/size[1],slice);
      //division by dimension shouldn't be required any more because FFT is now normalized
      normalize(1.0/(n1*n2),slice);
    //another division by size[0]*size[1] required for second FFT
}

///calculate autocorrelation of one slice of a data set
///(autocorrelation if not NAN safe - inherited from FFT)
///@param[in] slice   slice which is used for autocorrelation
///@param[in] norm    if true, normalize correlation such that 1 is correlated and -1 anticorrelated
void idl::autocorrelate(size_t slice,bool norm) 
{
    double n=0.0;
    if(norm)
	n=sqrt(averagesqr(slice));
    fft(slice);
    complex<double> *a=fftdata;
    for(size_t i=0;i<fftsize;i++,a++)
        *a=(*a)*conj(*a);
    ifft();
    if(norm)
	normalize(1.0/(n*n),slice);
}

///find maximum in complete data set returning the position (x,y) in (0,1)-plane (NAN-safe)
///@param[out]  x  x-value (0th dimension)
///@param[out]  y  y-value (1st dimension)
///@return maximum value, maximum is \a -HUGE_VAL if not found
///@todo return also slice
///@todo maximum should be NAN if not found
double idl::max(size_t &x,size_t &y)
{
  double max=-HUGE_VAL;
  x=0; y=0;
  for(size_t i=0;i<size[0];i++)
    for(size_t j=0;j<size[1];j++)
      for(size_t k=0;k<size[2];k++)
	if((!isnan(data[i][j][k])) && data[i][j][k]>max) {
	  max=data[i][j][k];
	  x=i; y=j;
	}
  return max;
}

///find maximum of plane \a slice (NAN-safe)
///@param[in]  slice  slice in which to look for maximum (2nd dimension)
///@return maximum value in slice, maximum is \a -HUGE_VAL if not found
///@todo generalize dimensions to find maximum in any dimension
///@todo maximum should be NAN if not found
double idl::max(size_t slice)
{
  double max=-HUGE_VAL;
  for(size_t i=0;i<size[0];i++)
    for(size_t j=0;j<size[1];j++)
	if((!isnan(data[i][j][slice])) && data[i][j][slice]>max)
	  max=data[i][j][slice];
  return max;
}

///find minimum in complete data set returning the position (x,y) in (0,1)-plane (NAN-safe)
///@param[out]  x  x-value (0th dimension)
///@param[out]  y  y-value (1st dimension)
///@return minimum value, minimum is \a HUGE_VAL if not found
///@todo return also slice
///@todo minimum should be NAN if not found
double idl::min(size_t &x,size_t &y)
{
  double min=HUGE_VAL;
  x=0; y=0;
  for(size_t i=0;i<size[0];i++)
    for(size_t j=0;j<size[1];j++)
      for(size_t k=0;k<size[2];k++)
	if((!isnan(data[i][j][k])) && data[i][j][k]<min) {
	  min=data[i][j][k];
	  x=i; y=j;
	}
  return min;
}

///find minimum of plane \a slice (NAN-safe)
///@param[in]  slice  slice in which to look for minimum (2nd dimension)
///@return minimum value in slice, minimum is \a HUGE_VAL if not found
///@todo generalize dimensions to find minimum in any dimension
///@todo minimum should be NAN if not found
double idl::min(size_t slice)
{
  double min=HUGE_VAL;
  for(size_t i=0;i<size[0];i++)
    for(size_t j=0;j<size[1];j++)
	if((!isnan(data[i][j][slice])) && data[i][j][slice]<min)
	  min=data[i][j][slice];
  return min;
}

///find minimum of spectrum at (x,y) in (0,1)-plane (NAN-safe)
///@param[in]  x    x-coordinate of spectrum (0th dimension)
///@param[in]  y    y-coordinate of spectrum (1st dimension)
///@param[in]  bias true: return bias voltage instead of value of minimum
///@return minimum value or bias of minimum value at (x,y), minimum is \a -HUGE_VAL if not found
///@todo generalize dimensions to find minimum in any dimension
///@todo minimum should be NAN if not found
double idl::min_pos(size_t x,size_t y,bool bias)
{
    size_t slice=0;
    double min=HUGE_VAL;
    for(size_t k=0;k<size[2];k++)
	if((!isnan(data[x][y][k])) && data[x][y][k]<min) {
	    slice=k;
	    min=data[x][y][k];
	}
    if(bias) return getbias(slice);
    else return min;
}

///find maximum of spectrum at (x,y) in (0,1)-plane (NAN-safe)
///@param[in]  x    x-coordinate of spectrum (0th dimension)
///@param[in]  y    y-coordinate of spectrum (1st dimension)
///@param[in]  bias true: return bias voltage instead of value of maximum
///@return maximum value or bias of maximum value at (x,y), maximum is \a -HUGE_VAL if not found
///@todo generalize dimensions to find maximum in any dimension
///@todo maximum should be NAN if not found
double idl::max_pos(size_t x,size_t y,bool bias)
{
    size_t slice=0;
    double max=-HUGE_VAL;
    for(size_t k=0;k<size[2];k++)
	if((!isnan(data[x][y][k])) && data[x][y][k]>max) {
	    max=data[x][y][k];
	    slice=k;
	}
    if(bias) return getbias(slice);
    else return max;
}

///calculate average of slice (2nd dimension) (NAN-safe)
///@param[in]  k    slice of which average is calculated (2nd dimension)
///@return average of slice k and NAN if all values are NAN
///@todo generalize dimensions to calculate average of any dimensions
double idl::average(size_t k)
{
    double average=0.0;
    size_t avgcnt=0;
    for(size_t i=0;i<size[0];i++)
	for(size_t j=0;j<size[1];j++)
	    if(!isnan(data[i][j][k])) { 
		average+=data[i][j][k];
		avgcnt++;
	    }
    if(avgcnt==0) return NAN;
    else return average/(double)avgcnt;
}

///calculate median of slice (2nd dimension) (NAN-safe)
///@param[in]  k    slice of which median is calculated (2nd dimension)
///@return median of slice k and NAN if all values are NAN
///@todo generalize dimensions to calculate average of any dimensions
double idl::median(size_t k)
{
    vector<double> vallist;
    size_t valcnt=0;
    vallist.reserve(size[0]*size[1]);
    for(size_t i=0;i<size[0];i++)
      for(size_t j=0;j<size[1];j++)
	if(!isnan(data[i][j][k])) { 
	  vallist.push_back(data[i][j][k]);
	  valcnt++;
	}
    if(valcnt==0) return NAN;
    sort(vallist.begin(),vallist.end());
    if(valcnt&1) {
      valcnt=valcnt>>1;
      return vallist[valcnt];
    } else {
      valcnt=valcnt>>1;
      return 0.5*(vallist[valcnt]+vallist[valcnt+1]);
    }
}

///calculate mode of slice (2nd dimension) (NAN-safe)
///@param[in]  k    slice of which mode is calculated (2nd dimension)
///@return mode of slice k and NAN if all values are NAN
///@todo generalize dimensions to calculate mode of any dimensions
double idl::mode(size_t k,size_t bins)
{
  double minval=min(k),maxval=max(k);
  if(isnan(minval) || isnan(maxval)) return NAN;
  vector<size_t> vallist;
  vallist.assign(bins,0);
  for(size_t i=0;i<size[0];i++)
    for(size_t j=0;j<size[1];j++)
      if(!isnan(data[i][j][k])) { 
	double val=(data[i][j][k]-minval)/(maxval-minval)*bins;
	vallist[(size_t)round(val)]++;
      }
  size_t maxind=0;
  for(size_t i=1;i<bins;i++)
    if(vallist[i]>vallist[maxind])
      maxind=i;
  return (double)maxind/bins*(maxval-minval)+minval;
}

///calculate \f$<x^2>\f$ (to obtain variance) of slice (2nd dimension) (NAN-safe)
///@param[in]  k    slice of which average is calculated (2nd dimension)
///@return average of square of slice k and 0.0 if all values are NAN
///@todo generalize dimensions to calculate average of square of any dimensions
///@todo return value should be NAN if all values are NAN
double idl::averagesqr(size_t k)
{
    double average=0.0;
    size_t avgcnt=0;
    for(size_t i=0;i<size[0];i++)
	for(size_t j=0;j<size[1];j++)
	    if(!isnan(data[i][j][k])) {
		average+=data[i][j][k]*data[i][j][k];
		avgcnt++;
	    }
    if(avgcnt==0) return NAN;
    else return average/(double)avgcnt;
}

///calculate average of spectrum at position (x,y) (0th and 1st dimension) (NAN-safe)
///@param[in]  x    x-coordinate of spectrum (0th dimension)
///@param[in]  y    y-coordinate of spectrum (1st dimension)
///@todo generalize dimensions to calculate average of any dimension
double idl::average(size_t x,size_t y)
{
    double average=0.0;
    size_t avgcnt=0;
    for(size_t k=0;k<size[2];k++)
	if(!isnan(data[x][y][k])) {
	    average+=data[x][y][k];
	    avgcnt++;
	}
    if(avgcnt==0) return NAN;
    else return average/(double)avgcnt;
}

///crop image area from (x0,y0,z0) (upper left corner) with size (xs,ys,zs)
///range is checked - nothing is done if check fails. 
///\a scan_range, \a offset and \a bias_range are adjusted
///@param[in]  x0    x-coordinate of upper left corner
///@param[in]  y0    y-coordinate of upper left corner
///@param[in]  z0    z-coordinate of upper left corner
///@param[in]  xs    size in x-direction
///@param[in]  ys    size in y-direction
///@param[in]  zs    size in z-direction
///@return -1 on error (range check), 0 if succeeded
int idl::cut(size_t x0,size_t y0,size_t z0,size_t xs,size_t ys,size_t zs)
{
  if((x0<0) || (y0<0) || (z0<0) || (x0+xs>size[0]) || (y0+ys>size[1]) || (z0+zs>size[2]))
    return -1;
  for(size_t i=0;i<xs;i++)
    {
	for(size_t j=0;j<ys;j++) {
	    for(size_t k=0;k<zs;k++)
		data[i][j][k]=data[x0+i][y0+j][z0+k];
	    data[i][j].resize(zs);
	}
	data[i].resize(ys);
    }
  data.resize(xs);
  offset[0]+=x0/size[0]*scan_range[0];
  offset[1]+=y0/size[1]*scan_range[1];
  scan_range[0]=scan_range[0]*xs/size[0];
  scan_range[1]=scan_range[1]*ys/size[1];
  setbiasrange(getbias(z0),getbias(z0+zs-1));
  size[0]=xs; size[1]=ys; size[2]=zs;
  return 0;
}

///crop image area from (x0,y0,*) (upper left corner) with size (xs,ys,*)
///range is checked - nothing is done if check fails. 
///\a scan_range and \a offset are adjusted
///calls cut(x0,y0,0,xs,ys,size[2])
///@param[in]  x0    x-coordinate of upper left corner
///@param[in]  y0    y-coordinate of upper left corner
///@param[in]  xs    size in x-direction
///@param[in]  ys    size in y-direction
///@return -1 on error (range check), 0 if succeeded
int idl::cut(size_t x0,size_t y0,size_t xs,size_t ys)
{
    return cut(x0,y0,0,xs,ys,size[2]);
}

///extend image area to new size (xns,yns,zns)
///\a scan_range, \a offset and \a bias_range are adjusted
///@param[in]  x0    offset in x-direction
///@param[in]  y0    offset in y-direction
///@param[in]  z0    offset in z-direction
///@param[in]  xns    size in x-direction
///@param[in]  yns    size in y-direction
///@param[in]  zns    size in z-direction
///@return -1 on error (range check), 0 if succeeded
int idl::extend(size_t x0,size_t y0,size_t z0,size_t xns,size_t yns,size_t zns)
{
  if((x0<0) || (y0<0) || (z0<0))
    return -1;
  data.resize(xns);
  for(size_t i=0;i<xns;i++)
    {
      data[i].resize(yns);
      for(size_t j=0;j<yns;j++)
	data[i][j].resize(zns,0.0);
    }
  offset[0]+=x0/size[0]*scan_range[0];
  offset[1]+=y0/size[1]*scan_range[1];
  scan_range[0]=scan_range[0]*xns/size[0];
  scan_range[1]=scan_range[1]*yns/size[1];
  setbiasrange(getbias(z0),getbias(z0+zns-1));
  size[0]=xns; size[1]=yns; size[2]=zns;
  return 0;
}

///extend image area to size (xns,yns,*)
///\a scan_range and \a offset are adjusted
///calls extend(x0,y0,0,xs,ys,size[2])
///@param[in]  x0    x-coordinate of upper left corner
///@param[in]  y0    y-coordinate of upper left corner
///@param[in]  xns    size in x-direction
///@param[in]  yns    size in y-direction
///@return -1 on error (range check), 0 if succeeded
int idl::extend(size_t x0,size_t y0,size_t xns,size_t yns)
{
    return extend(x0,y0,0,xns,yns,size[2]);
}

///range check of index l in dimension dim
///@param[in]  dim   dimension
///@param[out] l     index
bool idl::rangecheck(size_t dim,int &l)
{
  bool change=false;
  if(l<0) { l=0; change=true; }
  if(l>=(int)size[dim]) { l=(int)size[dim]-1; change=true; }
  return change;
}

///shift image area by (sx,sy)
///@param[in]  sx    shift in x (0th dimension)
///@param[in]  sy    shift in y (1st dimension)
///@todo generalize dimensions
void idl::shift(int sx,int sy)
{
    if(sx>0) { sx=size[0]-sx; }
    else { sx=-sx; }
    if(sy>0) { sy=size[1]-sy; }
    else { sy=-sy; }
    dblarr ndata;
    alloc(ndata);
    for(size_t i=0;i<size[0];i++)
	for(size_t j=0;j<size[1];j++) 
	    for(size_t k=0;k<size[2];k++)
		ndata[i][j][k]=data[(i+sx)%size[0]][(j+sy)%size[1]][k];
    data=ndata;
}

///get dimensions of image
///@param[out]  xs    x- or 0th dimension
///@param[out]  ys    y- or 1st dimension
///@todo generalize dimensions
void idl::dimensions(size_t &xs,size_t &ys)
{
  xs=size[0]; ys=size[1];
}

///get dimensions of map
///@param[out]  sa    array containing all three dimensions
void idl::dimensions(size_t sa[3])
{
  for(size_t d=0;d<3;d++)
    sa[d]=size[d];
}

///Constructor which does nothing
idl::idl():fftdata(NULL)
{
}

///Read IDL data from input stream
///@param[in] is input stream from which data is obtained
void idl::open(istream &is) 
{
  read(is);
}

///Constructor reading data from input stream
///@param[in] is input stream from which data is obtained
///@param[in] h  if true, only header is read and the rest of the data ignored
idl::idl(istream &is,bool h):headeronly(h),failed(false),fftdata(NULL)
{
  read(is);
}

///Read IDL data from file or stream
///@param[in] name name of file to be read, if name is -, data is read from console
void idl::open(const char *name)
{
  if(string(name)=="-")
    read(cin);
  else {
    ifstream is;
    is.exceptions(ifstream::eofbit|ifstream::failbit|ifstream::badbit);
    try {
      is.open(name);
    } catch (ifstream::failure const &e) {
      cerr<<"Error while opening file "<<name<<" for reading."<<endl
	  <<e.what()<<endl;
      failed=true;
      return;
    }
    read(is);
    is.close();
  }
}

///Constructor reading data from file or stream
///@param[in] name name of file to be read, if name is -, data is read from console
///@param[in] h    if true, only header is read and the rest of the data ignored
idl::idl(const char *name,bool h):headeronly(h),failed(false),fftdata(NULL)
{
  open(name);
}

///Constructor generating an empty IDL object
///@param[in] x  number of pixels in x- or 0th dimension
///@param[in] y  number of pixels in y- or 1st dimension
///@param[in] z  number of pixels in z- or 2nd dimension
///@param[in] xd image size in x- or 0th dimension (typically in AA)
///@param[in] yd image size in y- or 1st dimension (typically in AA)
///@param[in] lb lower boundary of voltage range in 2nd dimension (typically in V or mV)
///@param[in] hb higher boundary of voltage range in 2nd dimension (typically in V or mV)
///@param[in] sp setpoint current (typically in nA)
///@param[in] xp image offset in x- or 0th dimension (typically in AA)
///@param[in] yp image offset in y- or 1st dimension (typically in AA)
idl::idl(int x,int y,int z,double xd,double yd,double lb,double hb,double sp,double xp,double yp):headeronly(false),failed(false)
{
    fftdata=NULL;
    scan_range[0]=xd; scan_range[1]=yd; 
    size[0]=x; size[1]=y; size[2]=z;
    offset[0]=xp; offset[1]=yp; 
    junction_bias[0]=lb; junction_bias[1]=(hb==DBL_MAX)?lb:hb;
    setpoint=sp;
    time_t t=time(NULL);
    date=string(ctime(&t));
    date.resize(date.length()-1);
    alloc(data);
}

///Set name string
///@param[in] nname new name or description
void idl::setname(string nname)
{
    name=nname;
}

///Get name string
///@return string containing name or description
string idl::getname()
{
    return name;
}

///Obtain date string
///@return string containing date and time stamp
string idl::getdate()
{
    return date;
}

///Write FFT into an IDL structure
///@param[out] fft reference to idl object
void idl::putfft(idl &fftobj) 
{
  size_t kxs,kys;
  fftdimensions(kxs,kys);
  fftobj.size[0]=kxs; fftobj.size[1]=kys; fftobj.size[2]=2;
  fftobj.scan_range[0]=2.0*M_PI/scan_range[0];
  fftobj.scan_range[1]=2.0*M_PI/scan_range[1];
  fftobj.alloc(data);
  for(size_t i=0;i<kxs;i++)
    for(size_t j=0;j<kys;j++) {
      fftobj.set(i,j,0,getfft(i,j).real());
      fftobj.set(i,j,1,getfft(i,j).imag());
    }
}

///Desctructor
idl::~idl()
{
  if(fftdata) delete fftdata;
}

///output operator
///@param[in] i  idl class to be written
///@param[in] os output stream where the data is written to
///@return returns true if stream is valid
bool operator>>(idl& i,ostream &os)
{
  if(!os) return false;
  else i.write(os);
  return true;
}

///output operator
///@param[in] i  idl class to be written
///@param[in] name  name of file to be written, if name is -, data will be written to console
///@return returns true if valid stream could be generated from name
bool operator>>(idl& i,const char *name)
{
    if(string(name)=="-") return i>>cout;
    else {
	ofstream os;
	os.exceptions(ofstream::eofbit|ofstream::failbit|ofstream::badbit);
	try {
	    os.open(name);
	} catch (ofstream::failure const &e) {
	    cerr<<"Error while opening file "<<name<<" for writing."<<endl
		<<e.what()<<endl;
	    i.failed=true;
	}
	bool result=i>>os;
	os.close();
	return result;
    }
}

///input operator
///@param[in] i  idl class to which the data be written
///@param[in] is input stream from which the data read
///@return returns true if stream is valid
bool operator<<(idl& i,istream &is)
{
  if(!is) { 
      i.failed=true;
      return false;
  }
  else i.read(is);
  return true;
}

///input operator
///@param[in] i  idl class to which the data be written
///@param[in] name name of file from which the data should be read, if name is -, data will be read from console
///@return returns true if valid stream could be generated from name
bool operator<<(idl& i,const char *name)
{
    if(string(name)=="-") return i<<cin;
    else {
	ifstream is;
	is.exceptions(ifstream::eofbit|ifstream::failbit|ifstream::badbit);
	try {
	    is.open(name);
	} catch (ifstream::failure const &e) {
	    cerr<<"Error while opening file "<<name<<" for reading."<<endl
		<<e.what()<<endl;
	    i.failed=true;
	}
	bool result=i<<is;
	is.close();
	return result;
    }
}

///Obtain the status of the last I/O operation.
///@return true if o.k., false if last operation failed
bool operator!(idl &i) {return i.failed; }

///Check whether given coordinates are within the data range of the IDL object
///@param[in] x x-coordinate (or 0th dimension)
///@param[in] y y-coordinate (or 1st dimension)
///@return false, if (x,y) is not within the range of the object
bool idl::check(double x,double y)
{
    return ((x>offset[0]) && (x<offset[0]+scan_range[0]) && (y>offset[1]) && (y<offset[1]+scan_range[1]));
}

///Check whether given coordinates are within the data range of the IDL object
///@param[in] x x-coordinate (or 0th dimension)
///@param[in] y y-coordinate (or 1st dimension)
///@param[in] bias bias voltage (or 2nd dimension)
///@return false, if (x,y,bias) is not within the range of the object
bool idl::check(double x,double y,double bias)
{
    return check(x,y) && (bias>junction_bias[0]) && (bias<junction_bias[1]);
}

///Check whether given indices are within the data range of the IDL object
///@param[in] i x-index (or 0th dimension)
///@param[in] j y-index (or 1st dimension)
///@param[in] k z-index (or 2nd dimension)
///@return false, if (i,j,k) is not within the range of the object
bool idl::checki(size_t i,size_t j,size_t k)
{
    return (i<size[0])&&(j<size[1])&&(k<size[2]);
}

void idl::duplicate(size_t slice)
{
  setbiasrange(getbias(0),getbias(size[2]+1));
  for(size_t i=0;i<size[0];i++)
    for(size_t j=0;j<size[1];j++)
      data[i][j].push_back(data[i][j][slice]);
  size[2]++;
}

///Fill one layer with a value
///@param[in] val value to be filled in
///@param[in] k   slice, which is to be filled (2nd-dimension)
///@todo use generalized dimensions
void idl::fill(double val,size_t k)
{
    for(size_t i=0;i<size[0];i++)
	for(size_t j=0;j<size[1];j++)
	    data[i][j][k]=val;
}

///Set value at given position
///@param[in] i x-index
///@param[in] j y-index
///@param[in] val value
void idl::set(size_t i,size_t j,double val) {
  data[i][j][0]=val;
}

///Set value at given position
///@param[in] i x-index
///@param[in] j y-index
///@param[in] k z-index
///@param[in] val value
void idl::set(size_t i,size_t j,size_t k,double val) {
  data[i][j][k]=val;
}

///Set value at given position
///@param[in] x x-coordinate
///@param[in] y y-coordinate
///@param[in] val value
void idl::set(double x,double y,double val) {
  set((size_t)round((x-offset[0])/scan_range[0]*size[0]),(size_t)round((y-offset[1])/scan_range[1]*size[1]),
      val);
}

///Set value at given position
///@param[in] x x-coordinate
///@param[in] y y-coordinate
///@param[in] bias bias voltage
///@param[in] val value
void idl::set(double x,double y,double bias,double val) {
  set((size_t)round((x-offset[0])/scan_range[0]*size[0]),(size_t)round((y-offset[1])/scan_range[1]*size[1]),
      (size_t)round((bias-junction_bias[0])/(junction_bias[1]-junction_bias[0])*size[2]),val);
}

void idl::add(idl &map) {
  vector<size_t> loops;
  loops.resize(3);
  for(size_t i=0;i<3;i++)
    loops[i]=std::min(size[i],map.size[i]);
  for(size_t i=0;i<loops[0];i++)
    for(size_t j=0;j<loops[1];j++)
      for(size_t k=0;k<loops[2];k++)
	data[i][j][k]+=map(i,j,k);
}

///Get relative x-coordinate for given x-index
///@param[in] i x-index
///@return x-coordinate relative to upper left corner of image
///@todo use generalized dimensions, join with getrely
double idl::getrelx(int i)
{
    if(size[0]==1) return scan_range[0];
    else return scan_range[0]*i/(size[0]-1);
}

///Get relative y-coordinate for given y-index
///@param[in] i y-index
///@return y-coordinate relative to upper left corner of image
///@todo use generalized dimensions, join with getrelx 
double idl::getrely(int i)
{
    if(size[1]==1) return scan_range[1];
    else return scan_range[1]*i/(size[1]-1);
}

///Get absolute x-coordinate for given x-index
///@param[in] i x-index
///@return absolute x-coordinate
///@todo use generalized dimensions, join with gety
double idl::getx(int i)
{
  return getrelx(i)+offset[0];
}

///Get absolute y-coordinate for given y-index
///@param[in] i y-index
///@return absolute y-coordinate
///@todo use generalized dimensions, join with getx
double idl::gety(int i)
{
  return getrely(i)+offset[1];
}

///Get x-index for given x-coordinate
///@param[in] x absolute x-coordinate
///@return x-index
///@todo use generalized dimensions, join with getiy
double idl::getidx(double x)
{
    return ((x-offset[0])/scan_range[0]*size[0]);
}

///Get y-index for given y-coordinate
///@param[in] y absolute y-coordinate
///@return y-index
///@todo use generalized dimensions, join with getix
double idl::getidy(double y)
{
    return ((y-offset[1])/scan_range[1]*size[1]);
}

///Get x-index for given x-coordinate
///@param[in] x absolute x-coordinate
///@return x-index
///@todo use generalized dimensions, join with getiy
size_t idl::getix(double x)
{
    return (size_t)getidx(x);
}

///Get y-index for given y-coordinate
///@param[in] y absolute y-coordinate
///@return y-index
///@todo use generalized dimensions, join with getix
size_t idl::getiy(double y)
{
    return (size_t)getidy(y);
}

///Get bias value for given slice
///@param[in] i slice index
///@return bias voltage
///@todo use generalized dimensions
double idl::getbias(int i)
{
    if(size[2]==1) return junction_bias[0];
    else return (junction_bias[1]-junction_bias[0])*i/(size[2]-1.0)+junction_bias[0];
}

///Get slice index for given bias voltage 
///@param[in] bias slice index
///@return slice index
///@todo use generalized dimensions
int idl::getbiasindex(double bias)
{
    return (int)round((bias-junction_bias[0])/(junction_bias[1]-junction_bias[0])*(size[2]-1.0));
}

///Set bias range of data
///@param[in] lb lower boundary
///@param[in] hb higher boundary
void idl::setbiasrange(double lb,double hb)
{
    junction_bias[0]=lb; junction_bias[1]=hb;
}

///Get bias range of data
///@param[out] lb lower boundary
///@param[out] hb higher boundary
void idl::getbiasrange(double &lb,double &hb)
{
    lb=junction_bias[0]; hb=junction_bias[1];
}

///Perform plane subtraction on one slice of the data
///@param[in] x0 x-index of range in which plane is fitted
///@param[in] y0 y-index of range in which plane is fitted
///@param[in] xs size in x of fit range
///@param[in] ys size in y of fit range
///@param[in] k  index of slice to be used
///@todo make NAN safe ?
///@todo range checking on x0,y0,xs,ys
///@bug definition of xs and ys is wrong
///@note rather use the idlback functions for plane or polynomial compensation
void idl::plane_compensation(size_t x0,size_t y0,size_t xs,size_t ys,size_t k)
{
  double a=0.0,b=0.0,c=average(k);
  if(!xs) xs=size[0];
  if(!ys) ys=size[1];
  for(size_t i=x0;i<xs;i++)
    for(size_t j=y0;j<ys;j++) {
      if(i) a+=(data[i][j][k]-data[i-1][j][k]);
      if(j) b+=(data[i][j][k]-data[i][j-1][k]);
    }
  a/=(xs-1)*ys;
  b/=xs*(ys-1);
  c-=a*(size[0]-1.0)/2.0+b*(size[1]-1.0)/2.0;
  for(size_t i=0;i<size[0];i++)
    for(size_t j=0;j<size[1];j++) 
      data[i][j][k]-=(a*i+b*j+c);
}

///Subtract average of each slice from each slice of the data (NAN-safe)
void idl::removeconstant()
{
    for(size_t k=0;k<size[2];k++) {
	double c=average(k);
	for(size_t i=0;i<size[0];i++)
	    for(size_t j=0;j<size[1];j++)
		data[i][j][k]-=c;
    }
}

///Subtract average of slice from each slice of the data (NAN-safe)
///@param[in] slice layer from which constant is to be removed
void idl::removeconstant(size_t slice)
{
  double c=average(slice);
  for(size_t i=0;i<size[0];i++)
    for(size_t j=0;j<size[1];j++)
      data[i][j][slice]-=c;
}

///Filter data by convolution with filter
///@param[in] filter 2D array containing the filter; size must be odd
///@param[in] ignorenan if true, ignore fields containing NAN
///@todo adopt to generalized coordinates ?
void idl::lfilter2d(dblcol &filter,bool ignorenan,bool normalize)
{
  int xfs=(filter.size()>>1),yfs=(filter[0].size()>>1);
  dblarr ndata;
  alloc(ndata);
  for(int i=0;i<(int)size[0];i++)
    for(int j=0;j<(int)size[1];j++) 
      for(int k=0;k<(int)size[2];k++) {
	double val=0.0,norm=0.0;
	for(int ii=-xfs;ii<=xfs;ii++)
	  for(int jj=-yfs;jj<=yfs;jj++) {
	    if(((i+ii)>=0) && ((i+ii)<(int)size[0]) &&
	       ((j+jj)>=0) && ((j+jj)<(int)size[1]))
	      if(!(isnan(data[i+ii][j+jj][k])&&ignorenan)) {
		val+=filter[ii+xfs][jj+yfs]*data[i+ii][j+jj][k];
		norm+=filter[ii+xfs][jj+yfs];
	      }
	  }
	if(normalize) ndata[i][j][k]=val/norm;
	else ndata[i][j][k]=val;
      }
  data=ndata;
}

void idl::lfilter2d(dblcol &filter,size_t dim0, size_t dim1, bool ignorenan,bool normalize)
{
  int xfs=(filter.size()>>1),yfs=(filter[0].size()>>1);
  dblarr ndata;
  alloc(ndata);
  size_t dim2;
  for(dim2=0;(dim2<3) && ((dim2==dim0) || (dim2==dim1));dim2++);
  for(int i=0;i<(int)size[0];i++)
    for(int j=0;j<(int)size[1];j++) 
      for(int k=0;k<(int)size[2];k++) {
	double val=0.0,norm=0.0;
	for(int ii=-xfs;ii<=xfs;ii++)
	  for(int jj=-yfs;jj<=yfs;jj++) {
	    int nind[3]={i,j,k};
	    nind[dim0]+=ii; nind[dim1]+=jj;
	    if((nind[dim0]>=0) && (nind[dim0]<(int)size[dim0]) &&
	       (nind[dim1]>=0) && (nind[dim1]<(int)size[dim1]))
	      if(!(isnan(data[nind[0]][nind[1]][nind[2]])&&ignorenan)) {
		val+=filter[ii+xfs][jj+yfs]*data[nind[0]][nind[1]][nind[2]];
		norm+=filter[ii+xfs][jj+yfs];
	      }
	  }
	if(normalize) ndata[i][j][k]=val/norm;
	else ndata[i][j][k]=val;
      }
  data=ndata;
}

///Filter data by convolution with filter
///@param[in] filter 2D array containing the filter; size must be odd
///@param[in] ignorenan if true, ignore fields containing NAN
///@todo adopt to generalized coordinates ?
void idl::lfilter2dslice(dblcol &filter,size_t slice,bool ignorenan,bool normalize)
{
  int xfs=(filter.size()>>1),yfs=(filter[0].size()>>1);
  dblarr ndata;
  alloc(ndata);
  for(int i=0;i<(int)size[0];i++)
    for(int j=0;j<(int)size[1];j++) {
      double val=0.0,norm=0.0;
      for(int ii=-xfs;ii<=xfs;ii++)
	for(int jj=-yfs;jj<=yfs;jj++) {
	  if(((i+ii)>=0) && ((i+ii)<(int)size[0]) &&
	     ((j+jj)>=0) && ((j+jj)<(int)size[1]))
	    if(!(isnan(data[i+ii][j+jj][slice])&&ignorenan)) {
	      val+=filter[ii+xfs][jj+yfs]*data[i+ii][j+jj][slice];
	      norm+=filter[ii+xfs][jj+yfs];
	    }
	  if(normalize) ndata[i][j][slice]=val/norm;
	  else ndata[i][j][slice]=val;
	}
    }
  for(int i=0;i<(int)size[0];i++)
    for(int j=0;j<(int)size[1];j++)
      data[i][j][slice]=ndata[i][j][slice];
}

///Filter data by convolution with filter
///@param[in] filter 3D array containing the filter
void idl::lfilter(dblarr &filter,bool ignorenan,bool normalize)
{
  int xfs=(filter.size()>>1),yfs=(filter[0].size()>>1),zfs=(filter[0][0].size()>>1);
  dblarr ndata;
  alloc(ndata);
  for(int i=0;i<(int)size[0];i++)
    for(int j=0;j<(int)size[1];j++) 
      for(int k=0;k<(int)size[2];k++) {
	double val=0.0,norm=0.0;
	for(int ii=-xfs;ii<=xfs;ii++)
	  for(int jj=-yfs;jj<=yfs;jj++) 
	    for(int kk=-zfs;kk<=zfs;kk++) {
	      if(((i+ii)>=0) && ((i+ii)<(int)size[0]) &&
		 ((j+jj)>=0) && ((j+jj)<(int)size[1]) &&
		 ((k+kk)>=0) && ((k+kk)<(int)size[2]))
		if(!(isnan(data[i+ii][j+jj][k+kk])&&ignorenan)) {
		  val+=filter[ii+xfs][jj+yfs][kk+zfs]*data[i+ii][j+jj][k+kk];
		  norm+=filter[ii+xfs][jj+yfs][kk+zfs];
		}
	    }
	if(normalize) ndata[i][j][k]=val/norm;
	else ndata[i][j][k]=val;
      }
  data=ndata;
}

///Median filter data in 2D (spatial coordinates) using nxn pixels
///@param[in] n lateral size of filter matrix
void idl::median2d(int n)
{
  median(n,n,1);
}

///Median filter data in 3D using nxxnyxnz pixels
///@param[in] nx size of filter matrix in 0th-dimension
///@param[in] ny size of filter matrix in 1st-dimension
///@param[in] nz size of filter matrix in 2nd-dimension
void idl::median(int nx,int ny,int nz)
{
  unsigned int xind,yind,zind;
  int nx2=(nx>>1),ny2=(ny>>1),nz2=(nz>>1);
  dblval v;
  dblarr ndata;
  alloc(ndata);
  for(int i=0;i<(int)size[0];i++)
    for(int j=0;j<(int)size[1];j++) 
      for(int k=0;k<(int)size[2];k++) {
	v.clear();
	for(int ii=-nx2;ii<=nx2;ii++)
	  for(int jj=-ny2;jj<=ny2;jj++) 
	    for(int kk=-nz2;kk<=nz2;kk++) {
	      if(((i+ii)<0) || ((i+ii)>=(int)size[0])) continue;
	      else xind=(unsigned int)(i+ii);
	      if(((j+jj)<0) || ((j+jj)>=(int)size[1])) continue;
	      else yind=(unsigned int)(j+jj);
	      if(((k+kk)<0) || ((k+kk)>=(int)size[2])) continue;
	      else zind=(unsigned int)(k+kk);
	      v.push_back(data[xind][yind][zind]);
	    }
	sort(v.begin(),v.end());
	ndata[i][j][k]=v[v.size()>>1];
      }
  data=ndata;
}

///Calculates numerical derivative of data in given dimension with finite difference
///the resulting data set contains one layer less than the original one and the
///data range is shifted.
void idl::differentiate(size_t dimension)
{
  double step;
  if(dimension<2)
    step=scan_range[dimension]/(size[dimension]-1);
  else step=(junction_bias[1]-junction_bias[0])/(size[2]-1);
  size[dimension]--;
  size_t i[3],j[3];
  for(i[0]=0;i[0]<size[0];i[0]++) {
    for(i[1]=0;i[1]<size[1];i[1]++) {
      for(i[2]=0;i[2]<size[2];i[2]++) {
	j[0]=i[0]; j[1]=i[1]; j[2]=i[2]; j[dimension]++;
	data[i[0]][i[1]][i[2]]=(data[j[0]][j[1]][j[2]]-data[i[0]][i[1]][i[2]])/step;
      }
      if(dimension==2) data[i[0]][i[1]].resize(size[2]);
    }
    if(dimension==1) data[i[0]].resize(size[1]);
  }
  if(dimension==0) data.resize(size[0]);
  if(dimension<2) {
    scan_range[dimension]-=step;
    offset[dimension]+=0.5*step;
  } else {
    junction_bias[1]-=0.5*step; junction_bias[0]+=0.5*step;
  }
}

///Calculates numerical integral of data in given dimension by summing
void idl::integrate(size_t dimension)
{
  size_t latdim0=0,latdim1=1;
  double step;
  if(dimension<2) {
    if(dimension) latdim1=2;
    else latdim0=2;
    step=scan_range[dimension]/(size[dimension]-1);
  } else step=(junction_bias[1]-junction_bias[0])/(size[2]-1);
  size_t i[3],j[3];
  for(i[latdim0]=0;i[latdim0]<size[latdim0];i[latdim0]++)
    for(i[latdim1]=0;i[latdim1]<size[latdim1];i[latdim1]++) {
      i[dimension]=0;
      double last=data[i[0]][i[1]][i[2]]*step/2.0;
      data[i[0]][i[1]][i[2]]=last;
      for(i[dimension]=1;i[dimension]<size[dimension];i[dimension]++) {
	double act=data[i[0]][i[1]][i[2]]*step/2.0;
	for(size_t k=0;k<3;k++)
	  j[k]=i[k];
	j[dimension]--;
	data[i[0]][i[1]][i[2]]=act+last+data[j[0]][j[1]][j[2]];
	last=act;
      }
    }
}

void idl::getdata(double *arr)
{
  for(size_t k=0;k<size[2];k++)
    for(size_t j=0;j<size[1];j++)
      for(size_t i=0;i<size[0];i++)
	*(arr+i+j*size[0]+k*size[0]*size[1])=data[i][j][k];
}

void idl::setdata(double *arr)
{
  for(size_t k=0;k<size[2];k++)
    for(size_t j=0;j<size[1];j++)
      for(size_t i=0;i<size[0];i++)
	data[i][j][k]=*(arr+i+j*size[0]+k*size[0]*size[1]);
}

///Subtract line average from each line
///acts currently only on 0-th layer
///@param[in] slice layers to be destriped
///@todo use generalized coordinates ?
void idl::destripe(size_t slice)
{
    double av;
    for(size_t j=0;j<size[1];j++) {
	av=0.0;
	for(size_t i=0;i<size[0];i++)
	    av+=data[i][j][slice];
	av/=(double) size[0];
	for(size_t i=0;i<size[0];i++)
	    data[i][j][slice]-=av;
    }
}

///Multiply layer with value
///@param[in] x factor
///@param[in] slice layers which is to be used
void idl::normalize(double x,size_t slice)
{
    for(size_t i=0;i<size[0];i++)
	for(size_t j=0;j<size[1];j++)
	    data[i][j][slice]=data[i][j][slice]*x;
}

///Multiply layer with value
///@param[in] x factor
void idl::normalize(double x)
{
  for(size_t i=0;i<size[0];i++)
    for(size_t j=0;j<size[1];j++)
      for(size_t k=0;k<size[2];k++)
	data[i][j][k]=data[i][j][k]*x;
}
