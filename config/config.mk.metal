CC	= mpic++
LD	= mpic++
CPP	= $(CC)
install = install
instdir = $(HOME)/bin
KL1P	= $(HOME)/src/KL1p-0.4.2
LBFGS	= $(HOME)/src/liblbfgs-1.10
CCFLAGS	=  -Xclang -fopenmp -O3 -std=c++17 -I./metal-cpp -O2 -D_mpi_version -DGSL_RANGE_CHECK_OFF -DGSL_C99_INLINE -DHAVE_INLINE -D_METAL -D_GPU
LDFLAGS	= -L/opt/homebrew/lib -lpthread -ldl -lomp -fno-objc-arc -framework Metal -framework Foundation -framework MetalKit -framework QuartzCore
INCLUDE	= -I/opt/homebrew/include -I$(HOME)/src/metal-cpp
METALINCLUDE=
LIBCBLAS	= -lgslcblas
LIBFFTW	= -lfftw3_omp -lfftw3

GPUSRCS = qpimetal.cc
GPUOBJS = qpimetal.o

GPULIB = default.metallib

%.air : %.metal
	xcrun -sdk macosx metal -c $< -o $@

default.metallib : qpimetalkernel.air
	xcrun -sdk macosx metallib $< -o default.metallib
