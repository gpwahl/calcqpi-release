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

#define NS_PRIVATE_IMPLEMENTATION
#define CA_PRIVATE_IMPLEMENTATION
#define MTL_PRIVATE_IMPLEMENTATION

#include "qpi.h"
#include "qpimetal.h"


using namespace std;

//note: nkpts needs to be at least (n+1)
void MetalQPI::copygf(vector<vector<gsl_matrix_complex *> > &g0) {
    dblcomplex *gpug0mem= (dblcomplex *)_mg0->contents();
    int n2=nkpts>>1;
#pragma omp parallel for
  for(int i=-n2;i<n2;i++)
    for(int j=-n2;j<n2;j++)
      for(size_t k=0;k<bands;k++)
	for(size_t l=0;l<bands;l++) {
	  gsl_complex c=gsl_matrix_complex_get(g0[(kpoints+i)%kpoints][(kpoints+j)%kpoints],k,l);
	  *(gpug0mem+IDX4C((nkpts+i)%nkpts,(nkpts+j)%nkpts,k,l,nkpts,bands))=(dblcomplex){(gpufloat)GSL_REAL(c),(gpufloat)GSL_IMAG(c)};
	}
}

MetalQPI::MetalQPI(size_t wanniern, size_t kpoints,size_t n,size_t window, size_t oversamp,size_t bands,size_t maxband, bool spin,vector<wannierfunctions> &wf):wanniern(wanniern),kpoints(kpoints),bands(bands),spin(spin) {
  _mDevice = MTL::CreateSystemDefaultDevice();
    
  _mCommandQueue = _mDevice->newCommandQueue();
  if (_mCommandQueue == nullptr) {
    cout << "Failed to find the command queue." << std::endl;
    return;
  }
  nkpts=n+2*window+2;
  size_t scatsize=sizeof(dblcomplex)*bands*bands,g0size=sizeof(dblcomplex)*bands*bands*nkpts*nkpts,
    ldossize=wanniern*wanniern*sizeof(gpufloat), wfsize=maxband*(2*window+1)*oversamp*(2*window+1)*oversamp*sizeof(gpufloat),
    bufsize=sizeof(dblcomplex)*bands*bands*n*n;
  arraysize=n*n; gpumem=0;
  _mqpiPS=NULL;
  gridSize = MTL::Size::Make(arraysize, 1, 1);
  // Calculate a threadgroup size.
  _mscat=_mDevice->newBuffer(scatsize,MTL::ResourceStorageModeShared); gpumem+=scatsize;
  if(_mscat==NULL) { cerr<<"Cannot allocate memory for scattering matrix."<<endl; return; }
  _mg0=_mDevice->newBuffer(g0size,MTL::ResourceStorageModeShared); gpumem+=g0size;
  if(_mg0==NULL) { cerr<<"Cannot allocate memory for Green's function."<<endl; return; }
  _mcontg=NULL;
  _mldos=_mDevice->newBuffer(ldossize,MTL::ResourceStorageModeShared); gpumem+=ldossize;
  if(_mldos==NULL) { cerr<<"Cannot allocate memory for local density of states."<<endl; return; }
  _mwf=_mDevice->newBuffer(wfsize,MTL::ResourceStorageModeShared); gpumem+=wfsize;
  if(_mwf==NULL) { cerr<<"Cannot allocate memory for wave functions."<<endl; return; }
  _mqpiinfo=_mDevice->newBuffer(sizeof(qpigpuinfo),MTL::ResourceStorageModeShared); gpumem+=sizeof(qpigpuinfo);
  if(_mqpiinfo==NULL) { cerr<<"Cannot allocate memory for QPI info block."<<endl; return; }
  _mlocalbuffer=_mDevice->newBuffer(bufsize,MTL::ResourceStorageModePrivate); gpumem+=bufsize;
  if(_mlocalbuffer==NULL) { cerr<<"Cannot allocate memory for local buffer."<<endl; return; }
  _mlocalcldos=_mDevice->newBuffer(bufsize,MTL::ResourceStorageModePrivate); gpumem+=bufsize;
  if(_mlocalcldos==NULL) { cerr<<"Cannot allocate memory for local cldos."<<endl; return; }
  gpufloat *gpuwf= (gpufloat *)_mwf->contents();
  for(size_t i=0;i<maxband;i++)
    for(size_t j=0;j<(2*window+1)*oversamp;j++)
      for(size_t k=0;k<(2*window+1)*oversamp;k++)
	*(gpuwf+IDX3C(i,j,k,(2*window+1)*oversamp))=wf[i].getwave_cached(j,k);
  qpigpuinfo *qpiinfo=(qpigpuinfo *)_mqpiinfo->contents();
  qpiinfo->wanniern=wanniern;
  qpiinfo->kpoints=nkpts;
  qpiinfo->n=n;
  qpiinfo->window=window;
  qpiinfo->oversamp=oversamp;
  qpiinfo->bands=bands;
  qpiinfo->maxband=maxband;
  _mpos=NULL; _mflaglist=NULL; _mflagofs=NULL; _mflagentries=NULL;
}

MetalQPI::MetalQPI(size_t wanniern, size_t kpoints,size_t n,size_t window, size_t oversamp,size_t bands,flaglist &flist):wanniern(wanniern),kpoints(kpoints),bands(bands) {
  _mDevice = MTL::CreateSystemDefaultDevice();
    
  _mCommandQueue = _mDevice->newCommandQueue();
  if (_mCommandQueue == nullptr) {
    cout << "Failed to find the command queue." << std::endl;
    return;
  }
  nkpts=n+2*window+2;
  size_t winn=2*window+1,nwinn=winn*winn,nwinn2=nwinn*nwinn,tflentries=0;
  size_t scatsize=sizeof(dblcomplex)*bands*bands,g0size=sizeof(dblcomplex)*bands*bands*nkpts*nkpts,
    ldossize=wanniern*wanniern*sizeof(gpufloat),bufsize=sizeof(dblcomplex)*bands*bands*n*n;
  arraysize=n*n; gpumem=0;
  _mqpiPS=NULL;
  gridSize = MTL::Size::Make(arraysize, 1, 1);
  // Calculate a threadgroup size.
  _mscat=_mDevice->newBuffer(scatsize,MTL::ResourceStorageModeShared); gpumem+=scatsize;
  if(_mscat==NULL) { cerr<<"Cannot allocate memory for scattering matrix."<<endl; return; }
  _mg0=_mDevice->newBuffer(g0size,MTL::ResourceStorageModeShared); gpumem+=g0size;
  if(_mg0==NULL) { cerr<<"Cannot allocate memory for Green's function."<<endl; return; }
  _mcontg=NULL;
  _mldos=_mDevice->newBuffer(ldossize,MTL::ResourceStorageModeShared); gpumem+=ldossize;
  if(_mldos==NULL) { cerr<<"Cannot allocate memory for local density of states."<<endl; return; }
  _mqpiinfo=_mDevice->newBuffer(sizeof(qpigpuinfo),MTL::ResourceStorageModeShared); gpumem+=sizeof(qpigpuinfo);
  if(_mqpiinfo==NULL) { cerr<<"Cannot allocate memory for QPI info block."<<endl; return; }
  _mlocalbuffer=_mDevice->newBuffer(bufsize,MTL::ResourceStorageModePrivate); gpumem+=bufsize;
  if(_mlocalbuffer==NULL) { cerr<<"Cannot allocate memory for local buffer."<<endl; return; }
  _mlocalcldos=_mDevice->newBuffer(bufsize,MTL::ResourceStorageModePrivate); gpumem+=bufsize;
  if(_mlocalcldos==NULL) { cerr<<"Cannot allocate memory for local cldos."<<endl; return; }
  
  _mflagofs=_mDevice->newBuffer(nwinn2*sizeof(gpuuint),MTL::ResourceStorageModeShared); gpumem+=nwinn2*sizeof(gpuuint);
  if(_mflagofs==NULL) { cerr<<"Cannot allocate memory for flag offsets."<<endl; return; }
  _mflagentries=_mDevice->newBuffer(nwinn2*sizeof(gpuuint),MTL::ResourceStorageModeShared); gpumem+=nwinn2*sizeof(gpuuint);
  if(_mflagentries==NULL) { cerr<<"Cannot allocate memory for flag entries."<<endl; return; }
  gpuuint *flgofs=(gpuuint *)_mflagofs->contents(),
    *flgentr=(gpuuint *)_mflagentries->contents();
  for(size_t i=0;i<winn;i++)
    for(size_t j=0;j<winn;j++)
      for(size_t k=0;k<winn;k++)
	for(size_t l=0;l<winn;l++) {
	  size_t ofs=IDX4CC(i,j,k,l,winn);
	  *(flgofs+ofs)=tflentries;
	  *(flgentr+ofs)=flist[i][j][k][l].size();
	  tflentries+=flist[i][j][k][l].size();
	}
  size_t flsize=tflentries*sizeof(qpigpuflaglist);
  _mflaglist=_mDevice->newBuffer(flsize,MTL::ResourceStorageModeShared); gpumem+=flsize;
  if(_mflaglist==NULL) { cerr<<"Cannot allocate memory for flag list."<<endl; return; }
  qpigpuflaglist *gpuflg= (qpigpuflaglist *)_mflaglist->contents();
  for(size_t i=0;i<winn;i++)
    for(size_t j=0;j<winn;j++)
      for(size_t k=0;k<winn;k++)
	for(size_t l=0;l<winn;l++) {
	  size_t ofs=IDX4CC(i,j,k,l,winn),
	    flpos=*(flgofs+ofs),
	    fllen=*(flgentr+ofs);
	  for(size_t m=0;m<fllen;m++) {
	    (gpuflg+flpos+m)->i=flist[i][j][k][l][m].i;
	    (gpuflg+flpos+m)->j=flist[i][j][k][l][m].j;
	    (gpuflg+flpos+m)->o1=flist[i][j][k][l][m].o1;
	    (gpuflg+flpos+m)->o2=flist[i][j][k][l][m].o2;
	    (gpuflg+flpos+m)->factor=flist[i][j][k][l][m].factor;
	  }
	}
  qpigpuinfo *qpiinfo=(qpigpuinfo *)_mqpiinfo->contents();
  qpiinfo->wanniern=wanniern;
  qpiinfo->kpoints=nkpts;
  qpiinfo->n=n;
  qpiinfo->window=window;
  qpiinfo->oversamp=oversamp;
  qpiinfo->bands=bands;
  _mwf=NULL; _mpos=NULL;
}

MetalQPI::MetalQPI(size_t kpoints,size_t n,size_t bands,size_t maxband,bool spin,vector<vector<double> > &pos,vector<double> &prearr):wanniern(n),kpoints(kpoints),bands(bands),spin(spin) {
  _mDevice = MTL::CreateSystemDefaultDevice();
    
  _mCommandQueue = _mDevice->newCommandQueue();
  if (_mCommandQueue == nullptr) {
    cout << "Failed to find the command queue." << std::endl;
    return;
  }
  nkpts=kpoints;
  size_t g0size=sizeof(dblcomplex)*bands*bands*nkpts*nkpts,
    ldossize=wanniern*wanniern*sizeof(gpufloat);
  arraysize=n*n; gpumem=0;
  _mqpiPS=NULL;
  gridSize = MTL::Size::Make(arraysize, 1, 1);
  // Calculate a threadgroup size.
  _mg0=_mDevice->newBuffer(g0size,MTL::ResourceStorageModeShared); gpumem+=g0size;
  if(_mg0==NULL) { cerr<<"Cannot allocate memory for Green's function."<<endl; return; }
  _mldos=_mDevice->newBuffer(ldossize,MTL::ResourceStorageModeShared); gpumem+=ldossize;
  if(_mldos==NULL) { cerr<<"Cannot allocate memory for local density of states."<<endl; return; }

  if(pos.size()) {
    size_t possize=pos.size()*3*sizeof(gpufloat);
    _mpos=_mDevice->newBuffer(possize,MTL::ResourceStorageModeShared); gpumem+=possize;
    if(_mpos==NULL) { cerr<<"Cannot allocate memory for position information."<<endl; return; }
    gpufloat *gpupos= (gpufloat *)_mpos->contents();
    for(size_t i=0;i<pos.size();i++) {
      for(size_t j=0;j<2;j++)
	*(gpupos+IDX2C(j,i,3))=pos[i][j];
      *(gpupos+IDX2C(2,i,3))=prearr[i%prearr.size()];
    }
  }
  _mqpiinfo=_mDevice->newBuffer(sizeof(qpigpuinfo),MTL::ResourceStorageModeShared); gpumem+=sizeof(qpigpuinfo);
  if(_mqpiinfo==NULL) { cerr<<"Cannot allocate memory for QPI info block."<<endl; return; }
  qpigpuinfo *qpiinfo=(qpigpuinfo *)_mqpiinfo->contents();
  qpiinfo->wanniern=wanniern;
  qpiinfo->kpoints=nkpts;
  qpiinfo->n=n;
  qpiinfo->bands=bands;
  qpiinfo->maxband=maxband;
  _mflaglist=NULL; _mflagofs=NULL; _mflagentries=NULL; _mwf=NULL; _mcontg=NULL; _mscat=NULL; _mlocalbuffer=NULL; _mlocalcldos=NULL;
}

void MetalQPI::printinfo(ostream &os)
{
  if(_mDevice) {
    os<<"Running continuum QPI on Metal GPU."<<endl
      <<"Additional memory requirements: "<<std::fixed<<std::setprecision(2)<<(double)gpumem/1024.0/1024.0/1024.0<<"GB"<<endl;
  }
}

void MetalQPI::wannierldos(gsl_matrix_complex *scat, vector<vector<gsl_matrix_complex *> > &g0)
{
  if(_mqpiPS==NULL) {
    auto str=NS::String::string("", NS::ASCIIStringEncoding);
    if(spin)
      str= NS::String::string("gpucalcwannierldosspin", NS::ASCIIStringEncoding);
    else
      str= NS::String::string("gpucalcwannierldos", NS::ASCIIStringEncoding);
    _mqpiPS=getfunctionpipeline(str);
    NS::UInteger threadGroupSize = _mqpiPS->maxTotalThreadsPerThreadgroup();
    if (threadGroupSize > arraysize)
      {
	threadGroupSize = arraysize;
      }
    threadgroupSize = MTL::Size::Make(threadGroupSize, 1, 1);
  }
  dblcomplex *gpuscat = (dblcomplex *)_mscat->contents();
  //scattering matrix
  for(size_t i=0;i<bands;i++)
    for(size_t j=0;j<bands;j++) {
      gsl_complex c=gsl_matrix_complex_get(scat,i,j);
      *(gpuscat+IDX2C(i,j,bands))=(dblcomplex){(gpufloat)GSL_REAL(c),(gpufloat)GSL_IMAG(c)};
    }
  copygf(g0);
  // Create a command buffer to hold commands.
  commandBuffer = _mCommandQueue->commandBuffer();
  assert(commandBuffer != nullptr);
  
  // Start a compute pass.
  MTL::ComputeCommandEncoder *computeEncoder = commandBuffer->computeCommandEncoder();
  assert(computeEncoder != nullptr);
  computeEncoder->setComputePipelineState(_mqpiPS);
  computeEncoder->setBuffer(_mg0, 0, 0);
  computeEncoder->setBuffer(_mldos, 0, 1);
  computeEncoder->setBuffer(_mscat, 0, 2);
  computeEncoder->setBuffer(_mwf, 0, 3);
  computeEncoder->setBuffer(_mlocalbuffer, 0, 4);
  computeEncoder->setBuffer(_mlocalcldos, 0, 5);
  computeEncoder->setBuffer(_mqpiinfo, 0, 6);
    
  // Encode the compute command.
  computeEncoder->dispatchThreads(gridSize, threadgroupSize);

  // End the compute pass.
  computeEncoder->endEncoding();
  // Execute the command.
  commandBuffer->commit();
}

void MetalQPI::wannierjosephson(gsl_matrix_complex *scat, vector<vector<gsl_matrix_complex *> > &g0,gsl_complex tip)
{
  if(_mqpiPS==NULL) {
    auto str=NS::String::string("", NS::ASCIIStringEncoding);
    if(spin)
      str= NS::String::string("gpucalcwannierjosephsonspin", NS::ASCIIStringEncoding);
    else
      str= NS::String::string("gpucalcwannierjosephson", NS::ASCIIStringEncoding);
    _mqpiPS=getfunctionpipeline(str);
    NS::UInteger threadGroupSize = _mqpiPS->maxTotalThreadsPerThreadgroup();
    if (threadGroupSize > arraysize)
      {
	threadGroupSize = arraysize;
      }
    threadgroupSize = MTL::Size::Make(threadGroupSize, 1, 1);
  }
  dblcomplex *gpuscat = (dblcomplex *)_mscat->contents();
  //scattering matrix
  for(size_t i=0;i<bands;i++)
    for(size_t j=0;j<bands;j++) {
      gsl_complex c=gsl_matrix_complex_get(scat,i,j);
      *(gpuscat+IDX2C(i,j,bands))=(dblcomplex){(gpufloat)GSL_REAL(c),(gpufloat)GSL_IMAG(c)};
    }
  copygf(g0);
  qpigpuinfo *qpiinfo=(qpigpuinfo *)_mqpiinfo->contents();
  qpiinfo->tip=(dblcomplex){(gpufloat)GSL_REAL(tip),(gpufloat)GSL_IMAG(tip)};
  // Create a command buffer to hold commands.
  commandBuffer = _mCommandQueue->commandBuffer();
  assert(commandBuffer != nullptr);
  
  // Start a compute pass.
  MTL::ComputeCommandEncoder *computeEncoder = commandBuffer->computeCommandEncoder();
  assert(computeEncoder != nullptr);
  computeEncoder->setComputePipelineState(_mqpiPS);
  computeEncoder->setBuffer(_mg0, 0, 0);
  computeEncoder->setBuffer(_mldos, 0, 1);
  computeEncoder->setBuffer(_mscat, 0, 2);
  computeEncoder->setBuffer(_mwf, 0, 3);
  computeEncoder->setBuffer(_mlocalbuffer, 0, 4);
  computeEncoder->setBuffer(_mlocalcldos, 0, 5);
  computeEncoder->setBuffer(_mqpiinfo, 0, 6);
    
  // Encode the compute command.
  computeEncoder->dispatchThreads(gridSize, threadgroupSize);

  // End the compute pass.
  computeEncoder->endEncoding();
  // Execute the command.
  commandBuffer->commit();
}

void MetalQPI::wannierldoslist(gsl_matrix_complex *scat, vector<vector<gsl_matrix_complex *> > &g0)
{
  if(_mqpiPS==NULL) {
    auto str= NS::String::string("gpucalcwannierldoslist", NS::ASCIIStringEncoding);
    _mqpiPS=getfunctionpipeline(str);
    NS::UInteger threadGroupSize = _mqpiPS->maxTotalThreadsPerThreadgroup();
    if (threadGroupSize > arraysize)
      {
	threadGroupSize = arraysize;
      }
    threadgroupSize = MTL::Size::Make(threadGroupSize, 1, 1);
  }
  dblcomplex *gpuscat = (dblcomplex *)_mscat->contents();
  //scattering matrix
  for(size_t i=0;i<bands;i++)
    for(size_t j=0;j<bands;j++) {
      gsl_complex c=gsl_matrix_complex_get(scat,i,j);
      *(gpuscat+IDX2C(i,j,bands))=(dblcomplex){(gpufloat)GSL_REAL(c),(gpufloat)GSL_IMAG(c)};
    }
  copygf(g0);
  // Create a command buffer to hold commands.
  commandBuffer = _mCommandQueue->commandBuffer();
  assert(commandBuffer != nullptr);
  
  // Start a compute pass.
  MTL::ComputeCommandEncoder *computeEncoder = commandBuffer->computeCommandEncoder();
  assert(computeEncoder != nullptr);
  computeEncoder->setComputePipelineState(_mqpiPS);
  computeEncoder->setBuffer(_mg0, 0, 0);
  computeEncoder->setBuffer(_mldos, 0, 1);
  computeEncoder->setBuffer(_mscat, 0, 2);
  
  computeEncoder->setBuffer(_mflaglist, 0, 3);
  computeEncoder->setBuffer(_mflagofs, 0, 4);
  computeEncoder->setBuffer(_mflagentries, 0, 5);
  
  computeEncoder->setBuffer(_mlocalbuffer, 0, 6);
  computeEncoder->setBuffer(_mlocalcldos, 0, 7);
  computeEncoder->setBuffer(_mqpiinfo, 0, 8);
    
  // Encode the compute command.
  computeEncoder->dispatchThreads(gridSize, threadgroupSize);

  // End the compute pass.
  computeEncoder->endEncoding();
  // Execute the command.
  commandBuffer->commit();
}

void MetalQPI::spf(vector<vector<gsl_matrix_complex *> > &g0)
{
  if(_mqpiPS==NULL) {
    auto str=NS::String::string("", NS::ASCIIStringEncoding);
    str= NS::String::string("gpucalcspf", NS::ASCIIStringEncoding);
    _mqpiPS=getfunctionpipeline(str);
    NS::UInteger threadGroupSize = _mqpiPS->maxTotalThreadsPerThreadgroup();
    if (threadGroupSize > arraysize)
      {
	threadGroupSize = arraysize;
      }
    threadgroupSize = MTL::Size::Make(threadGroupSize, 1, 1);
  }
  copygf(g0);
  // Create a command buffer to hold commands.
  commandBuffer = _mCommandQueue->commandBuffer();
  assert(commandBuffer != nullptr);
  
  // Start a compute pass.
  MTL::ComputeCommandEncoder *computeEncoder = commandBuffer->computeCommandEncoder();
  assert(computeEncoder != nullptr);
  computeEncoder->setComputePipelineState(_mqpiPS);
  computeEncoder->setBuffer(_mg0, 0, 0);
  computeEncoder->setBuffer(_mldos, 0, 1);
  computeEncoder->setBuffer(_mqpiinfo, 0, 2);
    
  // Encode the compute command.
  computeEncoder->dispatchThreads(gridSize, threadgroupSize);

  // End the compute pass.
  computeEncoder->endEncoding();
  // Execute the command.
  commandBuffer->commit();
}

void MetalQPI::uspf(vector<vector<gsl_matrix_complex *> > &g0)
{
  if(_mqpiPS==NULL) {
    auto str=NS::String::string("", NS::ASCIIStringEncoding);
    if(spin)
      str= NS::String::string("gpucalcuspfspin", NS::ASCIIStringEncoding);
    else
      str= NS::String::string("gpucalcuspf", NS::ASCIIStringEncoding);
    _mqpiPS=getfunctionpipeline(str);
    NS::UInteger threadGroupSize = _mqpiPS->maxTotalThreadsPerThreadgroup();
    if (threadGroupSize > arraysize)
      {
	threadGroupSize = arraysize;
      }
    threadgroupSize = MTL::Size::Make(threadGroupSize, 1, 1);
  }
  copygf(g0);
  // Create a command buffer to hold commands.
  commandBuffer = _mCommandQueue->commandBuffer();
  assert(commandBuffer != nullptr);
  
  // Start a compute pass.
  MTL::ComputeCommandEncoder *computeEncoder = commandBuffer->computeCommandEncoder();
  assert(computeEncoder != nullptr);
  computeEncoder->setComputePipelineState(_mqpiPS);
  computeEncoder->setBuffer(_mg0, 0, 0);
  computeEncoder->setBuffer(_mpos, 0, 1);
  computeEncoder->setBuffer(_mldos, 0, 2);
  computeEncoder->setBuffer(_mqpiinfo, 0, 3);
    
  // Encode the compute command.
  computeEncoder->dispatchThreads(gridSize, threadgroupSize);

  // End the compute pass.
  computeEncoder->endEncoding();
  // Execute the command.
  commandBuffer->commit();
}

void MetalQPI::retrieveResult(vector<vector<double> > &ldos)
{
  gpufloat *gpuldosmem= (gpufloat *)_mldos->contents();
  commandBuffer->waitUntilCompleted();
  ldos.resize(wanniern);
#pragma omp parallel
  {
#pragma omp for
    for(size_t i=0;i<wanniern;i++)
      ldos[i].resize(wanniern);
#pragma omp for
    for(size_t i=0;i<wanniern;i++)
      for(size_t j=0;j<wanniern;j++)
	ldos[i][j]=*(gpuldosmem+IDX2C(i,j,wanniern));
  }
}

void MetalQPI::retrieveResult(idl &map,size_t layer)
{
  gpufloat *gpuldosmem= (gpufloat *)_mldos->contents();
  commandBuffer->waitUntilCompleted();
#pragma omp for
    for(size_t i=0;i<wanniern;i++)
      for(size_t j=0;j<wanniern;j++)
	map.set(i,j,layer,*(gpuldosmem+IDX2C(i,j,wanniern)));
}

void MetalQPI::retrieveResult(double *ldos)
{
  gpufloat *gpuldosmem= (gpufloat *)_mldos->contents();
  commandBuffer->waitUntilCompleted();
#pragma omp parallel for
  for(size_t i=0;i<wanniern*wanniern;i++)
    *(ldos+i)=*(gpuldosmem+i); 
}

MetalQPI::~MetalQPI()
{
  if(_mg0) _mg0->release();
  if(_mldos) _mldos->release();
  if(_mscat) _mscat->release();
  if(_mwf) _mwf->release();
  if(_mflaglist) _mflaglist->release();
  if(_mflagofs) _mflagofs->release();
  if(_mflagentries) _mflagentries->release();
  if(_mpos) _mflagentries->release();
  if(_mlocalbuffer) _mlocalbuffer->release();
  if(_mlocalcldos) _mlocalcldos->release();
  if(_mqpiinfo) _mqpiinfo->release();
  if(_mcontg) _mcontg->release();
}

MTL::ComputePipelineState *MetalQPI::getfunctionpipeline(NS::String *str)
{
  MTL::Library *defaultLibrary = _mDevice->newDefaultLibrary();
  
  if (defaultLibrary == nullptr) {
    cout << "Failed to find the default library." << std::endl;
    return NULL;
  }
  
  MTL::Function *newFunction = defaultLibrary->newFunction(str);
  
  if (newFunction == nullptr) {
    cout << "Failed to find the GPU-implementation of CLDOS." << std::endl;
    return NULL;
  }
  NS::Error *error = nullptr;
  // Create a compute pipeline state object.
   MTL::ComputePipelineState *_mQPIFunctionPS = _mDevice->newComputePipelineState(newFunction, &error);
  
  if (_mQPIFunctionPS == nullptr) {
    //  If the Metal API validation is enabled, you can find out more information about what
    //  went wrong.  (Metal API validation is enabled by default when a debug build is run
    //  from Xcode)
    cout << "Failed to created pipeline state object, error " << error << "." << std::endl;
    return NULL;
  }
  return _mQPIFunctionPS;
}
