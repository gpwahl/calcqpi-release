#This file is part of calcQPI.
#calcQPI is free software: you can redistribute it and/or modify
#it under the terms of the GNU General Public License as published
#by the Free Software Foundation, either version 3 of the License,
#or (at your option) any later version.
#calcQPI is distributed in the hope that it will be useful, but
#WITHOUT ANY WARRANTY; without even the implied warranty of
#MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
#See the GNU General Public License for more details.
#You should have received a copy of the GNU General Public License along with calcQPI.
#If not, see <https://www.gnu.org/licenses/>.


.DEFAULT_GOAL := all

include ../config.mk

INCLUDE += -I../include

.SUFFIXES: .o .c

SRC1	= mkwavefunctions.cc wannierfunctions.cc loadxcf.cc gitversion.cc idl.cc
OBJ1	= mkwavefunctions.o wannierfunctions.o loadxcf.o gitversion.o idl.o
PROJ1	= mkwavefunctions

SRC3	= calcqpi.cc qpi.cc tightbinding.cc wannierfunctions.cc idl.cc loadxcf.cc gitversion.cc gethostdata.cc $(GPUSRCS)
OBJ3	= calcqpi.o qpi.o tightbinding.o wannierfunctions.o idl.o loadxcf.o gitversion.o gethostdata.o $(GPUOBJS)
PROJ3	= calcqpi

#.cc.o: ; $(CC) -c -g -Wall -pedantic $(INCLUDE) $*.cc
.cc.o: ; $(CC) -c -g -Wall -pedantic $(CCFLAGS) $(INCLUDE) $*.cc
#.cc.o: ; $(CC) -c -g -Wall -pedantic -D_enable_cs -D_mpi_version $(INCLUDE) $*.cc


all: $(PROJ1) $(PROJ3)

$(PROJ1): $(OBJ1)
	$(LD) $(OBJ1) -o $(PROJ1) -lgsl $(LIBCBLAS) $(LIBFFTW) $(LDFLAGS)

$(PROJ3): $(OBJ3) $(GPULIB)
	$(LD) $(OBJ3) -lgsl $(LIBCBLAS) $(LIBFFTW) $(LDFLAGS) -o $(PROJ3) 

gitversion.cc: ../.git/HEAD ../.git/index
	echo "const char *gitversion = \"$(shell git rev-parse HEAD)\", *gitbranch = \"$(shell git rev-parse --abbrev-ref HEAD)\";" > $@

default:
	$(MAKE) all

install:
	$(install) $(PROJ1) $(PROJ3) $(instdir)

depend:
	@makedepend -Y $(SRC1) $(SRC3) 2> /dev/null

clean:
	-rm -f *.o gitversion.cc

