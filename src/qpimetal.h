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

#pragma once

#include <Foundation/Foundation.hpp>
#include <Metal/Metal.hpp>
#include <QuartzCore/QuartzCore.hpp>

#include "idl.h"
#include "gpuqpi.h"
#include "qpimetalkernel.h"

class MetalQPI: public GPUQPI
{
public:
  MTL::Device *_mDevice;
  size_t wanniern, kpoints, nkpts,bands;
  size_t arraysize,gpumem;
  bool spin;
  
  // The compute pipeline generated from the compute kernel in the .metal shader file.
  MTL::ComputePipelineState *_mqpiPS;
  
  // The command queue used to pass commands to the device.
  MTL::CommandQueue *_mCommandQueue;

  MTL::CommandBuffer *commandBuffer;
  
  // Buffers to hold data.
  MTL::Buffer *_mg0,*_mcontg,*_mscat,*_mwf,*_mldos,*_mqpiinfo,*_mctspecinfo,*_mlocalbuffer,*_mlocalcldos,*_mpos;
  MTL::Buffer *_mflaglist,*_mflagofs,*_mflagentries;

  MTL::Size gridSize;
  MTL::Size threadgroupSize;

  MetalQPI(size_t wanniern, size_t kpoints,size_t n,size_t window, size_t oversamp,size_t bands,size_t maxband, bool spin,vector<wannierfunctions> &wf);
  MetalQPI(size_t wanniern, size_t kpoints,size_t n,size_t window, size_t oversamp,size_t bands,flaglist &flist);
  MetalQPI(size_t kpoints,size_t n,size_t bands,size_t maxbands,bool spin,vector<vector<double> > &pos,vector<double> &prearr);
  
  void printinfo(ostream &os);
  void wannierldos(gsl_matrix_complex *scat, vector<vector<gsl_matrix_complex *> > &g0);
  void wannierjosephson(gsl_matrix_complex *scat, vector<vector<gsl_matrix_complex *> > &g0,gsl_complex tip);
  void wannierldoslist(gsl_matrix_complex *scat, vector<vector<gsl_matrix_complex *> > &g0);
  void spf(vector<vector<gsl_matrix_complex *> > &g0);
  void uspf(vector<vector<gsl_matrix_complex *> > &g0);
  void retrieveResult(vector<vector<double> > &ldos);
  void retrieveResult(idl &map,size_t layer);
  void retrieveResult(double *ldos);
  ~MetalQPI();
private:
  MTL::ComputePipelineState *getfunctionpipeline(NS::String *str);
  void copygf(vector<vector<gsl_matrix_complex *> > &g0);
};
