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

#ifndef _qpimetalkernel_h
#define _qpimetalkernel_h

typedef float gpufloat;
typedef uint gpuuint;
typedef int gpuint;

struct dblcomplex {
  gpufloat real, imag;
};

struct qpigpuinfo {
  gpuuint wanniern,kpoints,n,bands,oversamp,window,maxband;
  gpufloat xorig,yorig;
  dblcomplex tip;
};

struct qpigpuflaglist {
  gpuuint i,j,o1,o2;
  gpufloat factor;
};

struct gfgpuinfo {
  gpuuint totalkpoints,bands;
  gpufloat omega,eta;
};

//indexing for 2D arrays
#define IDX2C(i,j,ld) (((j)*(ld))+(i))
//indexing for 4D arrays (Green's function)
#define IDX4C(i,j,k,l,ld0,ld1) ((((j)*(ld0))+(i))*(ld1)*(ld1)+(((l)*(ld1))+(k)))
#define IDX4CC(i,j,k,l,ld) ((((((i)*(ld))+(j))*(ld)+(k))*(ld))+(l))
//indexing for 3D array
#define IDX3C(i,j,k,ld0) (((i)*((ld0)*(ld0)))+(((j)*(ld0))+(k)))

#endif
