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

#ifndef _MPIDEFS_H
#define _MPIDEFS_H

#ifdef _mpi_version

#define ExecuteCPU0 if(world_rank==0)

#ifdef _MPI_MAIN
int world_size=1,world_rank=0;
#else
extern int world_size, world_rank;
#endif

#else
#define ExecuteCPU0
#endif
#endif
