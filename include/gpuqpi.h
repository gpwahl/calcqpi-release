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

#ifndef _gpu_h
#define _gpu_h

class GPUQPI
{
public:
  GPUQPI() {}
  virtual void printinfo(ostream &os)=0;
  virtual void wannierldos(gsl_matrix_complex *scat, vector<vector<gsl_matrix_complex *> > &g0)=0;
  virtual void wannierjosephson(gsl_matrix_complex *scat, vector<vector<gsl_matrix_complex *> > &g0,gsl_complex tip)=0;
  virtual void wannierldoslist(gsl_matrix_complex *scat, vector<vector<gsl_matrix_complex *> > &g0)=0;
  virtual void spf(vector<vector<gsl_matrix_complex *> > &g0)=0;
  virtual void uspf(vector<vector<gsl_matrix_complex *> > &g0)=0;
  virtual void retrieveResult(vector<vector<double> > &ldos)=0;
  virtual void retrieveResult(idl &map,size_t layer)=0;
  virtual void retrieveResult(double *ldos)=0;
  virtual ~GPUQPI() {}
};

#endif
