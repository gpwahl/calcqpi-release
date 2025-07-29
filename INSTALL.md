# Installation

This document describes how to compile and install calcQPI.

## Prerequisites

The following libraries are required to compile and use calcQPI:

* Gnu Scientific Library (GSL)
* FFTW3

For the MPI version:
* an MPI library, for example OpenMPI

For the GPU versions:

* for Apple Silicon, the Metal-cpp interface (see <https://developer.apple.com/metal/cpp/>)
* for the Cuda implementation, the Cuda libraries and NVidia compiler
* for AMD GPUs, the HIP libraries and compilter

## Obtaining the source code

The source code is best cloned from the github repository using
'git clone git@github.com:gpwahl/wannier-qpi.git'
which will clone the release version from github.

## Configuring the Makefile

The Makefile includes a file `config.mk` which should contain anything that the user might need to modify. Importantly, it has the name of compiler and linker as well as include and library paths. For some standard installations, there are example config files. In many cases, a symlink to one of the config files using
>`ln -s config/config.mk.example ./config.mk`

will be sufficient (where example is changed to select on of the files in the config subdirectory). Alternatively, one of the files can be copied and modified further. On standard Linux distributions, the main edits which may be required are related to whether or not to compile with MPI support. For MPI support, the MPI C++ compiler and linker must be chosen for `CC` and `LD`, and the flag `_mpi_version` must be set. If the GSL and FFTW3 libraries are not in the standard path, the library path needs to be explicitly provided.

There are specific config.mk files tested on the following systems:
| Filename 	   	       | Description |
| ---			       | --- |
| config.mk.archer2	       | Archer2 (tier-1 HPC facility), CPU version with MPI, using Cray compiler |
| config.mk.archer2-hip	       | Archer2, CPU/GPU version using HIP (for AMD GPUs) and MPI |
| config.mk.cirrus	       | Cirrus (tier-2 HPC facility), CPU version with MPI |
| config.mk.cirrus-cuda	       | Cirrus with CUDA for CPU/GPU version |
| config.mk.hypatia	       | Local cluster Hypatia at the University of St Andrews, CPU version |
| config.mk.hypatia-cuda       | Hypatia with CPU/GPU support |
| config.mk.kennedy	       | Old local cluster Kennedy at the University of St Andrews, CPU version |
| config.mk.kennedy-cuda       | Kennedy with CPU/GPU support |
| config.mk.kennedy-singlenode | Kennedy without MPI support |
| config.mk.macos 	       | MAC OS with CPU and MPI support |
| config.mk.macos-nompi        | MAC OS with CPU (no MPI support) |
| config.mk.metal 	       | MAC OS with CPU/GPU support, using M1 GPU |
| config.mk.marvin 	       | HPC Cluster Marvin of University of Bonn, CPU/MPI |
| config.mk.marvin-gpu 	       | Marvin, CPU/MPI/GPU |
| config.mk.suse 	       | Suse linux with MPI (CPU version) |
| config.mk.suse-nompi 	       | Suse linux without MPI (CPU version) |
| config.mk.gpucluster 	       | For GPU cluster at the University of St Andrews (Cuda, with MPI/GPU support) |
| config.mk.dtc 	       | For AMD cluster at the University of Edinburgh (CPU/MPI) |
| config.mk.wsl-ubuntu 	       | Windows subsystem for Linux (CPU/MPI) |
| config.mk.wsl-ubuntu-nompi   | Windows subsystem for Linux (CPU) |

## Compiling calcQPI

To compile calcQPI, just type
>`make all`

this can be sped up by compiling on multiple cores in parallel with, e.g.,
>`make -j4 all`

to compile using four simultaneous processes.

## Installation

If desired, the code can be installed in a directory specified in the config.mk file, the default is `$(HOME)/bin`, but this can be changed.
