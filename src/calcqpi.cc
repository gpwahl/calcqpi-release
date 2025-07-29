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

//Code to calculate QPI based on a tight-binding model saved in a Wannier-style file
//Supported optimizations:
//  * multiprocessor support via MPI (only for continuum LDOS, compile with _mpi_version)
//required libraries
//  * FFTW3, GSL
//  * MPI for MPI version

#include <iostream>
#include <iomanip>
#include <unistd.h>
#include <sstream>
#include <string.h>
#include <fstream>
#include <vector>
#include <stdio.h>

#ifdef _mpi_version
#include <mpi.h>
#endif

#define _MPI_MAIN
#include "mpidefs.h"

#include <omp.h>

using namespace std;


#include "idl.h"
#include "tightbinding.h"
#include "qpi.h"

#include "addtimestamp.h"
#include "parser.h"
#include "recordtimings.h"
#include "gitversion.h"
#include "gethostdata.h"

void help()
{
  cerr<<"calcqpi (branch: "<<gitbranch<<", commit: "<<gitversion<<")"<<endl<<endl
      <<"Syntax:"<<endl
      <<"calcqpi configfile"<<endl<<endl
      <<"Parameters of the configuration file:"<<endl
      <<"Input files:"<<endl
      <<" tbfile     : file name of file containing TB model"<<endl
      <<"Output files:"<<endl
      <<" qpifile    : output file for QPI"<<endl
      <<" output     : output mode for QPI (one of wannier, josephson, spf, uspf, nomode; default: wannier)"<<endl
      <<" logfile    : output log to logfile (default: cout)"<<endl
      <<" wffile     : output wave functions to wffile"<<endl
      <<"Parameters for bandstructure histogram:"<<endl
      <<" bsfile     : output file for band structure"<<endl
      <<" bslattice  : size in pixels (default: 201)"<<endl
      <<" bsoversamp : n-times oversampling (default: 8)"<<endl
      <<" bsenergies : energy range (default: -0.1, 0.1)"<<endl
      <<" bslayers   : number of layers (default: 21)"<<endl
      <<"Parameters for DOS output:"<<endl
      <<" dosfile    : output of total density of states"<<endl
      <<" dosenergies: energy range (default: -0.1, 0.1)"<<endl
      <<" doslayers  : number of layers (default: 101)"<<endl
      <<"Dimensions of output array for QPI:"<<endl
      <<" lattice    : size in pixels (default: 201)"<<endl
      <<" oversamp   : n-times oversampling (default: 4)"<<endl
      <<" energies   : energy range (default: -0.1, 0.1)"<<endl
      <<" layers     : number of layers (default: 21)"<<endl
      <<"Parameters for Green's function:"<<endl
      <<" green      : one of surface, bulk, normal, default: normal"<<endl
      <<" stbfile    : file name of file containing TB model of surface layer, if none specified, the same TB model as above is used"<<endl
      <<" epserr     : convergence criterion for sum of squares of the matrix elements of alpha and beta (default: 1e-5)"<<endl
      <<"Parameters of t-matrix calculation:"<<endl
      <<" fermi      :  Fermi energy (default: 0.0)"<<endl
      <<" scattering : specify scattering matrix as list of factors. This could be 1 or 0, but also other values to set specific scattering potentials for specific orbitals (default: all 1)."<<endl
      <<" phase      : scattering phase (as complex number)"<<endl
      <<" eta        : broadening parameter eta (default: 0.005)"<<endl
      <<" kpoints    : k-point grid, use same as lattice size if not specified"<<endl
      <<" spin       : if true assumes that half the bands are spin up, and the other half spin down"<<endl
      <<"Parameters for the continuum LDOS:"<<endl
      <<" window     : window of size n (default: 0 (auto), if too small, a warning will be issued)"<<endl
      <<"              if no window specified, the value will be estimated automatically"<<endl
      <<" radius     : use radius for size of orbital functions (default: 0.5)"<<endl
      <<" angle      : rotate wannier functions by specified angle (default: 0.0)"<<endl
      <<" anglearr   : additional angle by which individual Wannier functions are being rotated (default: none)"<<endl
      <<" prearr     : array of prefactors for orbitals (default: all 1)"<<endl
      <<" orbitals   : list of orbitals used for Wannier functions"<<endl
      <<" pos[0], ...: positions of the orbitals (default: 0,0)"<<endl
      <<" zheight    : height of z-layer above top-most atom (when using Wannier functions from DFT, negative height is below bottom atom; used also for atomic-like orbitals)"<<endl
      <<" orbitalfiles: list of files used for Wannier functions"<<endl
      <<" idlorbitalfile: idl file containing the Wannier functions"<<endl
      <<"Parameters for Bogoljubov QPI:"<<endl
      <<" magscat    : fraction of magnetic scattering, 0: only non-magnetic scattering, 1: only magnetic scattering (default: 0)"<<endl
      <<" scmodel    : assume model is superconducting, if 'true' only first half of bands is considered for QPI."<<endl
      <<"Parameters for Josephson mode:"<<endl
      <<" deltat     : gap size of tip"<<endl
      <<" etat       : broadening for tip, default is same as sample"<<endl
      <<"Creating proximity map:"<<endl
      <<" threshold  : threshold value, default: NAN (i.e. use window)."<<endl;
}

void checkmax(double val, double &max) {
  if(val>max) max=val;
}

void checkwavefunctions(ostream &os,vector<wannierfunctions> &wfs)
{
  double max=0.0;
  for(size_t i=0;i<wfs.size();i++)
    checkmax(wfs[i].getmaxboundary()/wfs[i].getmax(),max);
  os<<"Maximum rel. value at boundary: "<<max<<endl;
  if(max>0.05) os<<"Warning: boundary too close, use larger window size!"<<endl;
}

enum outputmode {wanniermode,josephsonmode,spfmode,uspfmode,nomode};

enum greensfunction {surfacegreen,bulkgreen,normalgreen};

int main(int argc, char *argv[])
{
  size_t n=201,layers=21,oversamp=4,window=2,kpoints=0;
  outputmode mode=wanniermode;
  greensfunction gf=normalgreen;
  bool spin=false,scmodel=false;
  double lenergy=-1.0,henergy=1.0,eta=0.005,efermi=0.0,radius=0.5,angle=0.0,zheight=5.0,epserr=1.0e-5,magscat=0.0,threshold=NAN,deltat=0.001,etat=NAN;
  complex<double> scatphase(1.0,0.0);
  string tbfile(""),stbfile(""),qpifile("");
  vector<complex<double> > scattering;
  vector<string> orbitals;
  vector<wannierfunctions> wfs;
  ofstream logfilestr;
  recordtimings proctimings;
  if(argc>1) {
#ifdef _mpi_version
    MPI_Init(&argc,&argv);
    //int world_size,world_rank;
    MPI_Comm_size(MPI_COMM_WORLD,&world_size);
    MPI_Comm_rank(MPI_COMM_WORLD,&world_rank);
#endif
    streambuf *oldcerrstreambuf = cerr.rdbuf(), *oldcoutstreambuf=cout.rdbuf();
    AddTimeStamp ats(cerr),ats2(cout),*atslog=NULL;
    ostringstream errbuf;
    cerr.rdbuf( errbuf.rdbuf() );
    loadconfig lf(argv[1]);
    if(!lf) {
      cerr.rdbuf(oldcerrstreambuf);
      if(errbuf.str().length()) cerr<<errbuf.str()<<endl;
      return -1;
    }
    if(lf.probefield("logfile")) {
      string logfile=lf.getstring("logfile");
      logfilestr.open(logfile.c_str(),ofstream::out|ofstream::app);
      if(logfilestr.is_open()) {
	atslog=new AddTimeStamp(logfilestr);
	cout.rdbuf(atslog);
	cerr.rdbuf(atslog);
      }
    } else cerr.rdbuf(oldcerrstreambuf); 
    ExecuteCPU0 {
      cout<<"calcqpi (branch: "<<gitbranch<<", commit: "<<gitversion<<")"<<endl
	  <<"New calculation started, using configuration from "<<argv[1]<<endl;
      cout<<"Running on "<<omp_get_max_threads()<<" cores";
#ifdef _mpi_version
      cout<<" and "<<world_size<<" tasks";
#else
      cout<<"."<<endl;
#endif
      if(errbuf.str().length()) cerr<<"Error while reading configuration file:"<<endl<<errbuf.str();
    }
    vector<string> hostnames;
    string username;
    gethostdata(hostnames,username);
    ExecuteCPU0 {
      cout<<" as user "<<username<<" on machine";
      if(hostnames.size()>1) cout<<"s ";
      else cout<<" ";
      for(size_t i=0;i<hostnames.size();i++) {
	if(i) cout<<", ";
	cout<<hostnames[i];
      }
      cout<<endl;
    }
#ifdef  _mpi_version
    cout<<"Initializing processor "<<world_rank<<"/"<<world_size<<"."<<endl;
#endif
    if(lf.probefield("tbfile")) tbfile=lf.getstring("tbfile");
    else {
      ExecuteCPU0 cerr<<"No tightbinding model specified ... exiting."<<endl;
      return 0;
    }
    if(lf.probefield("stbfile")) stbfile=lf.getstring("stbfile");
    if(lf.probefield("fermi")) efermi=lf.getvalue("fermi");
    if(lf.probefield("epserr")) epserr=lf.getvalue("epserr");
    if(lf.probefield("green")) {
      if(lf.getfield("green")=="surface") gf=surfacegreen;
      else if(lf.getfield("green")=="bulk") gf=bulkgreen;
      else if(lf.getfield("green")=="normal") gf=normalgreen;
      else ExecuteCPU0 cerr<<"Invalid value for green: "<<lf.getfield("green")<<endl;
    }
    if(lf.probefield("spin")) {
      if(lf.getfield("spin")=="true") spin=true;
      else if(lf.getfield("spin")=="false") spin=false;
      else ExecuteCPU0 cerr<<"Invalid value for spin: "<<lf.getfield("spin")<<endl;
    }
    if(lf.probefield("scmodel")) {
      if(lf.getfield("scmodel")=="true") scmodel=true;
      else if(lf.getfield("scmodel")=="false") scmodel=false;
      else ExecuteCPU0 cerr<<"Invalid value for scmodel: "<<lf.getfield("scmodel")<<endl;
    }
    if(lf.probefield("output")) {
      string outname=lf.getfield("output");
      if(outname=="wannier") mode=wanniermode;
      else if(outname=="josephson") mode=josephsonmode;
      else if(outname=="nomode") mode=nomode;
      else if(outname=="spf") mode=spfmode;
      else if(outname=="uspf") mode=uspfmode;
      else {
	ExecuteCPU0 cerr<<"Unrecognized output mode.";
	return 0;
      }
    }
    if(lf.probefield("qpifile")) qpifile=lf.getstring("qpifile");
    if(lf.probefield("lattice")) n=lf.getintvalue("lattice");
    if(lf.probefield("oversamp")) oversamp=lf.getintvalue("oversamp");
    if(lf.probefield("window")) window=lf.getintvalue("window");
    if(lf.probefield("threshold")) threshold=lf.getvalue("threshold");
    if(lf.probefield("layers")) layers=lf.getintvalue("layers");
#ifdef _mpi_version
    if(((layers%world_size)!=0)&&(world_rank==0)) cout<<"Warning - Number of layers ("<<layers<<") not a multiple of number of MPI tasks ("<<world_size<<")."<<endl;
#endif
    if(lf.probefield("energies")) {
      vector<double> energies=lf.getvector("energies");
      lenergy=energies[0];
      henergy=energies[1];
    }
    if(lf.probefield("eta")) eta=lf.getvalue("eta");
    if(lf.probefield("etat")) etat=lf.getvalue("etat");
    if(isnan(etat)) etat=eta;
    if(lf.probefield("deltat")) deltat=lf.getvalue("deltat");
    if(lf.probefield("kpoints")) kpoints=lf.getintvalue("kpoints");
    if(!kpoints) kpoints=n;
    if(lf.probefield("phase")) {
      vector<double> phase=lf.getvector("phase");
      scatphase=complex<double>(phase[0],phase[1]);
      ExecuteCPU0 cout<<"Using scattering phase ("<<phase[0]<<","<<phase[1]<<")"<<endl;
    }
    if(lf.probefield("scattering")) {
      vector<double> scatterlist=lf.getvector("scattering");
      scattering.resize(scatterlist.size());
      for(size_t i=0;i<scatterlist.size();i++)
	scattering[i]=scatterlist[i]*scatphase;
      ExecuteCPU0 {
	cout<<"Scattering: "<<endl;
	for(size_t i=0;i<scatterlist.size();i++)
	  if(i)
	    cout<<",("<<scattering[i].real()<<","<<scattering[i].imag()<<")";
	  else cout<<"("<<scattering[i].real()<<","<<scattering[i].imag()<<")";
	cout<<endl;
      }
    }
    if(lf.probefield("magscat")) magscat=lf.getvalue("magscat");
    tightbind *model,*smodel=NULL;
    size_t maxbands;
    ExecuteCPU0 cout<<"Loading TB model "<<tbfile<<endl;
    model=new tightbind(tbfile.c_str(),efermi,scmodel);
    if(!*model) {
      ExecuteCPU0 cerr<<"Error loading TB model from "<<tbfile<<", exiting."<<endl;
      return 0;
    }
    if(scmodel) {
      maxbands=model->getbands()>>1;
      ExecuteCPU0 cout<<"Assuming TB model to be superconducting, only first half of bands used."<<endl;
      //duplicate scattering matrix
      size_t scatlen=scattering.size();
      if(scatlen==maxbands) {
	for(size_t i=0;i<scatlen;i++)
	  scattering.push_back((-1.0+2.0*magscat)*scattering[i]);
      }
    } else
      maxbands=model->getbands();
    if(stbfile.length()>0) {
      ExecuteCPU0 cout<<"Loading surface TB model "<<stbfile<<endl;
      smodel=new tightbind(stbfile.c_str(),efermi,scmodel);
      if(!*smodel) {
	ExecuteCPU0 cerr<<"Error loading surface TB model from "<<stbfile<<", exiting."<<endl;
	return 0;
      }
    }
    if(gf!=normalgreen) {
      if(!(model->checkprincipallayer()))
	ExecuteCPU0 cout<<"Warning: Tight-binding model in file "<<tbfile<<" is not in principal layer form."<<endl;
      if(smodel)
	if(!(smodel->checkprincipallayer()))
	  ExecuteCPU0 cout<<"Warning: Surface tight-binding model in file "<<stbfile<<" is not in principal layer form."<<endl;
    }
    ExecuteCPU0 if(lf.probefield("bsfile")) {
#ifdef _mpi_version
      cout<<"Calculating band structure histogram on processor "<<world_rank<<"/"<<world_size<<"."<<endl;
#endif
      size_t bslattice=201, bslayers=21, bsoversamp=8;
      double bslenergy=-0.1,bshenergy=0.1;
      string bsfile=lf.getstring("bsfile");
      if(lf.probefield("bslayers")) bslayers=lf.getintvalue("bslayers");
      if(lf.probefield("bsenergies")) {
	vector<double> energies=lf.getvector("bsenergies");
	bslenergy=energies[0];
	bshenergy=energies[1];
      }
      if(lf.probefield("bslattice")) bslattice=lf.getintvalue("bslattice");
      if(lf.probefield("bsoversamp")) bsoversamp=lf.getintvalue("bsoversamp");
      cout<<"Calculating band structure histogram in energy range ("<<bslenergy<<","<<bshenergy<<") with "<<bslayers<<" layers and oversampling "<<bsoversamp<<"."<<endl;
      model->calcbst3dhisthr(bsfile.c_str(),bslattice,bslenergy,bshenergy,bslayers,bsoversamp);
      cout<<"... done and saved to "<<bsfile<<"."<<endl;
      proctimings.addtimeentry("Bandstructure calculation");
    }
    double maxhopping=model->getmaxhopping();
    ExecuteCPU0 if(maxhopping/n<eta)
	cout<<"Note: Broadening eta ("<<eta<<") large compared to maximum hopping ("<<maxhopping<<") at given k-point resolution (hopping/kpoints="<<maxhopping/kpoints<<")."<<endl;
    
    // model.writeinfo(cout);
    if(mode==spfmode)
      kpoints=n;
    tmatrix *tm=new tmatrix(model,n,kpoints,eta,spin);
    if(lf.probefield("dosfile")) {
      size_t doslayers=101;
      double doslenergy=-0.1,doshenergy=0.1;
      string dosfile=lf.getstring("dosfile");
      if(lf.probefield("doslayers")) doslayers=lf.getintvalue("doslayers");
      if(lf.probefield("dosenergies")) {
	vector<double> energies=lf.getvector("dosenergies");
	doslenergy=energies[0];
	doshenergy=energies[1];
      }
      ExecuteCPU0 cout<<"Calculating density of states in energy range ("<<doslenergy<<","<<doshenergy<<") with "<<doslayers<<" layers."<<endl;
      double step=(doshenergy-doslenergy)/doslayers;
      double *dosdata;
#ifdef _mpi_version
      size_t doslayersperprocess=doslayers/world_size;
      if(doslayersperprocess*world_size<doslayers)
	doslayersperprocess++;
      size_t dosmpilayers=doslayersperprocess*world_size;
      if(world_rank)
	dosdata=new double[doslayersperprocess];
      else dosdata=new double[dosmpilayers];
      size_t dosstartlayer=doslayersperprocess*world_rank,dosendlayer=dosstartlayer+doslayersperprocess;
      if(dosendlayer>doslayers) dosendlayer=doslayers;
      for(size_t k=0;k<doslayersperprocess;k++) {
	double energy=doslenergy+(dosstartlayer+k)*step;
#else
      dosdata=new double[doslayers];
      for(size_t k=0;k<doslayers;k++) {
	double energy=doslenergy+k*step;
#endif
	switch(gf) {
	case surfacegreen:
	  tm->calcsurfacegreensfunction(energy,epserr,true,smodel);
	  break;
	case bulkgreen:
	  tm->calcsurfacegreensfunction(energy,epserr,false,smodel);
	  break;
	case normalgreen:
	  tm->setgreensfunction(energy);
	  break;
	}
	tm->calcrealspacegreensfunction();
	dosdata[k]=tm->calcdos(maxbands);
      }
#ifdef _mpi_version
      MPI_Gather(dosdata,doslayersperprocess,MPI_DOUBLE,dosdata,doslayersperprocess,MPI_DOUBLE,0,MPI_COMM_WORLD);
#endif
      ExecuteCPU0 {
	ofstream dosout(dosfile.c_str());
        for(size_t k=0;k<doslayers;k++)
  	  dosout<<doslenergy+k*step<<" "<<dosdata[k]<<endl;
        cout<<"... done and saved to "<<dosfile<<"."<<endl;
        proctimings.addtimeentry("Calculation of density of states");
      }
      delete[] dosdata;
    }
    if(mode!=nomode) {
    if(scattering.size()>0) 
      tm->setscatteringphase(scattering);
    else tm->setscatteringphase(scatphase);
    ExecuteCPU0 {
      cout<<"Parameters of calculation:"<<endl
	  <<"         Tight-binding model  : "<<tbfile<<endl
	  <<"         Surface tight-binding model: "<<stbfile<<endl
	  <<"         Output file for QPI  : "<<qpifile<<endl
	  <<"         Bands:               : "<<maxbands<<endl
	  <<"         Lattice              : "<<n<<"x"<<n<<endl
	  <<"         kpoints              : "<<kpoints<<"x"<<kpoints<<endl
	  <<"         energy range         : ("<<lenergy<<","<<henergy<<")"<<endl
	  <<"         energy layers        : "<<layers<<endl
	  <<"         energy spacing       : "<<(henergy-lenergy)/((double)layers-1.0)<<endl
	  <<"         Fermi energy         : "<<efermi<<endl
	  <<"         broadening parameter : "<<eta<<endl
	  <<"         window               : "<<window<<endl
	  <<"         oversamp             : "<<oversamp<<endl
	  <<"         epserr               : "<<epserr<<endl;
      if(!isnan(threshold))
	cout<<"         threshold            : "<<threshold<<endl;
      cout<<"         Green's function     : ";
      switch(gf) {
      case surfacegreen:
	cout<<"surface"<<endl;
	break;
      case bulkgreen:
	cout<<"bulk"<<endl;
	break;
      case normalgreen:
	cout<<"normal"<<endl;
	break;
      }
      if(spin) cout<<"         spin-polarized calculation"<<endl;
      else cout<<"         paramagnetic calculation"<<endl;
      cout<<"Parallelization:"<<endl
		<<"         Number of threads    : "<<omp_get_max_threads()<<endl;
#ifdef _mpi_version
      cout<<"         MPI - number of tasks: "<<world_size<<endl;
#endif
      double greenmem=(double)model->getbands()*model->getbands()*kpoints*kpoints*2.0*sizeof(double)/1024.0/1024.0/1024.0,
	fftmem=(double)kpoints*kpoints*2.0*sizeof(double)/1024.0/1024.0/1024.0;
      double evectmem=0.0,evalmem=0.0;
      if(gf==normalgreen) {
	evectmem=greenmem;
	evalmem=(double)model->getbands()*kpoints*kpoints*sizeof(double)/1024.0/1024.0/1024.0;
      }
      cout<<"Expected memory usage per task:"<<endl
		<<"         Green's function:         "<<std::fixed<<std::setprecision(2)<<greenmem<<" GB"<<endl
		<<"         Buffering of hamiltonian: "<<std::fixed<<std::setprecision(2)<<evectmem+evalmem<<" GB"<<endl
		<<"         FFT buffer:               "<<std::fixed<<std::setprecision(2)<<fftmem<<" GB"<<endl
		<<"         ---------------------------------"<<endl
		<<"         Total:                    "<<std::fixed<<std::setprecision(2)<<(greenmem+fftmem+evectmem+evalmem)<<" GB"<<endl;     
    }
    switch(mode) {
    case spfmode :
    case uspfmode :
    case wanniermode :
    case josephsonmode : {
      vector<vector<double> > posarr;
      if(mode==wanniermode) {
	ExecuteCPU0 cout<<"QPI calculations in Wannier-mode (cLDOS)."<<endl;
      } else if(mode==josephsonmode) {
	ExecuteCPU0 cout<<"QPI calculations in Josephson-mode."<<endl
			<<"   using deltat="<<deltat<<", etat="<<etat<<"."<<endl;
	if(!scmodel) {
	  ExecuteCPU0 cerr<<"Error: Calculation in josephson-mode with scmodel=false makes no sense - exiting."<<endl;
	  exit(1);
	}
      } else if(mode==spfmode) {
	ExecuteCPU0 cout<<"Calculations of spectral function."<<endl;
      } else if(mode==uspfmode) {
	ExecuteCPU0 cout<<"Calculations of unfolded spectral function."<<endl;
	for(size_t i=0;i<maxbands;i++) {
	  ostringstream posstr;
	  posstr<<"pos["<<i<<"]";
	  if(lf.probefield(posstr.str())) {
	    vector<double> pos=lf.getvector(posstr.str());
	    if(pos.size()==2) pos.push_back(0.0);
	    posarr.push_back(pos);
	  } else {
	    vector<double> pos(2,0.0);
	    posarr.push_back(pos);
	    ExecuteCPU0 cout<<"No position found for band "<<i<<", using (0,0,0)."<<endl;
	  }
	}
      }
      vector<double> prearr;
      if(lf.probefield("prearr")) prearr=lf.getvector("prearr");
      prearr.resize(maxbands,1.0);
      if(lf.probefield("idlorbitalfile")) { //Initialization of orbitals from an IDL file written by calcqpi
	string orbitalfile=lf.getstring("idlorbitalfile");
	ExecuteCPU0 cout<<"Using IDL orbital file for vacuum overlap."<<endl
			      <<"Loading wave function file "<<orbitalfile<<"."<<endl;
	idl orbitals(orbitalfile.c_str());
	if(!orbitals) {
	  ExecuteCPU0 cerr<<"Error loading orbitals from file "<<orbitalfile<<", exiting."<<endl;
	  return 1;
	}
	size_t orbs=orbitals.layers();
	prearr.resize(orbs,1.0);
	vector<vector<double> > arr;
	size_t xs,ys;
	orbitals.dimensions(xs,ys);
	if((xs!=ys) || (xs!=(2*window+1)*oversamp)) {
	  ExecuteCPU0 cerr<<"Orbital functions in "<<orbitalfile<<" have unsuitable dimensions."<<endl;
	   return 0;
	}
	arr.resize(xs);
	for(size_t i=0;i<xs;i++)
	  arr[i].assign(ys,0.0);
	for(size_t k=0;k<orbs;k++) {
	  for(size_t i=0;i<xs;i++)
	    for(size_t j=0;j<ys;j++)
	      arr[i][j]=orbitals(i,j,k);
	  wfs.push_back(wannierfunctions(arr,prearr[k]));
	}
	if(((maxbands>>1)==orbs) && spin) {
	  ExecuteCPU0 cout<<"Found half as many orbitals as needed in spin-polarized calculation, duplicating wave function set."<<endl;
	  for(size_t k=0;k<orbs;k++)
	    wfs.push_back(wfs[k]);
	} else if(maxbands>orbs) {
	  ExecuteCPU0 cout<<"Warning: Fewer number of orbitals in "<<orbitalfile<<" than there are bands, padding with zeros."<<endl;
	  for(size_t i=orbs;i<maxbands;i++)
	    wfs.push_back(wannierfunctions("z",radius,zheight,0.0,0.0,0.0,0.0,0.0));
	}
      } else if(lf.probefield("orbitals")) { //Initialization of orbitals for Gausian orbital mode
	orbitals=lf.getlist("orbitals");
	if(lf.probefield("radius")) radius=lf.getvalue("radius");
	if(lf.probefield("angle")) angle=lf.getvalue("angle");
	if(lf.probefield("zheight")) zheight=lf.getvalue("zheight");
	ExecuteCPU0 cout<<"Using gaussian orbitals for vacuum overlap."<<endl
			      <<"    Parameters:"<<endl
			      <<"             Radius:         "<<radius<<endl
			      <<"             Rotation angle: "<<angle<<endl
			      <<"             Height:         "<<zheight<<endl;
	vector<double> anglearr;
	if(lf.probefield("anglearr")) anglearr=lf.getvector("anglearr");
	else anglearr.assign(orbitals.size(),0.0);
	prearr.resize(orbitals.size(),1.0);
	for(size_t i=0;i<orbitals.size();i++) {
	  ostringstream posstr;
	  posstr<<"pos["<<i<<"]";
	  if(lf.probefield(posstr.str())) {
	    vector<double> pos=lf.getvector(posstr.str());
	    if(pos.size()==2) pos.push_back(0.0);
	    wfs.push_back(wannierfunctions(orbitals[i],radius,zheight-pos[2],pos[0],pos[1],(angle+anglearr[i])*M_PI/180.0,0.0,prearr[i]));
	  } else 
	    wfs.push_back(wannierfunctions(orbitals[i],radius,zheight,0.0,0.0,(angle+anglearr[i])*M_PI/180.0,0.0,prearr[i]));
	}
	if(((maxbands>>1)==orbitals.size()) && spin) {
	  ExecuteCPU0 cout<<"Found half as many orbitals as needed in spin-polarized calculation, duplicating wave function set."<<endl;
	  for(size_t k=0;k<orbitals.size();k++)
	    wfs.push_back(wfs[k]);
	} else if(orbitals.size()<maxbands) {
	  ExecuteCPU0 cout<<"Warning: Fewer number of orbitals specified than there are bands, padding with zeros."<<endl;
	  for(size_t i=orbitals.size();i<maxbands;i++)
	    wfs.push_back(wannierfunctions("z",radius,zheight,0.0,0.0,0.0,0.0,0.0));
	}
	ExecuteCPU0 {
	  cout<<"Parameters for Wannier functions:"<<endl;
	  for(size_t i=0;i<wfs.size();i++) {
	    double x,y,h;
	    wfs[i].getpos(x,y,h);
	    cout<<"Orbital "<<i<<": "<<wfs[i].getname()<<"@("<<x<<","<<y<<"), height "<<h<<", theta "<<wfs[i].gettheta()*180.0/M_PI<<"deg "<<endl;
	  }
	}
	if(window) {
	  double val=radius*sqrt(log(100.0)+abs(zheight));
	  ExecuteCPU0 cout<<"Recommended window size for r="<<radius<<": "<<ceil(val*val-zheight*zheight)<<endl
			  <<"chosen window size: "<<window<<endl;
	} else {
	  for(size_t i=0;i<wfs.size();i++) {
	    double newwin=ceil(wfs[i].getrange(0.01));
	    if(newwin>window)
	      window=ceil(newwin);
	  }
	  ExecuteCPU0 cout<<"Using recommended window size for r="<<radius<<": "<<window<<endl;
	}
	ExecuteCPU0 cout<<"Precalculating wave functions."<<endl;
	for(size_t i=0;i<wfs.size();i++)
	  wfs[i].precalculate(window,oversamp);
	ExecuteCPU0 cout<<"Done."<<endl;
	ExecuteCPU0 checkwavefunctions(cout,wfs); 
      } else if(lf.probefield("orbitalfiles")) { //Initialization of orbitals from DFT/Wannier90 output
	if(lf.probefield("zheight")) zheight=lf.getvalue("zheight");
	ExecuteCPU0 cout<<"Using Wannier90/XSF-orbitals for vacuum overlap."<<endl
			      <<"    Parameters:"<<endl
			      <<"             zheight:         "<<zheight<<endl;
	vector<string> orbitalfiles=lf.getstringlist("orbitalfiles");
	prearr.resize(orbitalfiles.size(),1.0);
	for(size_t i=0;i<orbitalfiles.size();i++) {
	  ExecuteCPU0 cout<<"Loading wave function file "<<orbitalfiles[i]<<"."<<endl;
	  wfs.push_back(wannierfunctions(orbitalfiles[i],zheight,window,oversamp,prearr[i]));
	  if(!wfs[i]) {
	    ExecuteCPU0 cerr<<"Error loading wave function file "<<orbitalfiles[i]<<endl;
	    return 0;
	  }
	}
	if(((maxbands>>1)==orbitalfiles.size()) && spin) {
	  ExecuteCPU0 cout<<"Found half as many orbitals as needed in spin-polarized calculation, duplicating wave function set."<<endl;
	  for(size_t k=0;k<orbitalfiles.size();k++)
	    wfs.push_back(wfs[k]);
	} else if(orbitalfiles.size()<maxbands) {
	  ExecuteCPU0 cout<<"Fewer number of orbital files specified than there are bands, padding with zeros."<<endl;
	  for(size_t i=orbitals.size();i<maxbands;i++)
	    wfs.push_back(wannierfunctions("z",radius,zheight,0.0,0.0,0.0,0.0,0.0));
	}
      } 
      ExecuteCPU0 if(lf.probefield("wffile")) {
#ifdef _mpi_version
	cout<<"Writing wave function file from processor "<<world_rank<<"/"<<world_size<<"."<<endl;
#endif
	size_t xs,ys;
	wfs[0].getcachesize(xs,ys);
	idl wfout(xs,ys,wfs.size());
	for(size_t l=0;l<wfs.size();l++)
	  for(size_t i=0;i<xs;i++)
	    for(size_t j=0;j<ys;j++)
	      wfout.set(i,j,l,wfs[l].getwave_cached(i,j));
	ostringstream comment;
	comment<<"Wannier function file for "<<tbfile<<" with window="<<window<<" and oversamp="<<oversamp;
	wfout.setname(comment.str());
	cout<<"Writing wave function file to "<<lf.getstring("wffile")<<endl;
	wfout>>lf.getstring("wffile").c_str();
      }
      //flagarray flags;
      flaglist fls;
      if(!isnan(threshold)) {
	//ExecuteCPU0 cout<<"Creating threshold map ..."<<flush;
	//size_t count=mkthresholdmap(flags,wfs,window,oversamp,threshold,maxbands,spin);
	size_t count;
	ExecuteCPU0 cout<<"Creating threshold list ..."<<flush;
	if(lf.probefield("spinarr")) {
	  vector<int> spinarr=lf.getintvector("spinarr");
	  count=mkthresholdlist(fls,wfs,window,oversamp,threshold,spinarr);
	} else
	  count=mkthresholdlist(fls,wfs,window,oversamp,threshold,maxbands,spin);
	ExecuteCPU0 cout<<" done"<<endl;
	size_t entries=(2*window+1)*(2*window+1)*(2*window+1)*(2*window+1)*oversamp*oversamp*wfs.size()*wfs.size();
	//ExecuteCPU0 cout<<"Threshold map has "<<entries<<" entries of which "<<count<<" are true ("<<(double)count/entries*100.0<<"%), requiring "<<entries*sizeof(bool)/1024.0/1024.0/1024.0<<"GB of memory."<<endl;
	ExecuteCPU0 cout<<"Threshold list has "<<count<<"/"<<entries<<" entries ("<<(double)count/entries*100.0<<"%), requiring "<<count*sizeof(flaglistentry)/1024.0/1024.0/1024.0<<"GB of memory."<<endl;
      }   
      /*ExecuteCPU0 {*/
	ostringstream comment;
	switch(mode) {
	case wanniermode:
	  ExecuteCPU0 comment<<"CQPI calculation for "<<model->getname()<<" with window="<<window<<" and oversamp="<<oversamp;
	  n=oversamp*n;
	  break;
	case josephsonmode:
	  ExecuteCPU0 comment<<"Josephson calculation for "<<model->getname()<<" with window="<<window<<" and oversamp="<<oversamp;
	  n=oversamp*n;
	  break;
	case spfmode:
	  ExecuteCPU0 comment<<"SPF calculation for "<<model->getname();
	  break;
	case uspfmode:
	  ExecuteCPU0 comment<<"USPF calculation for "<<model->getname()<<" with kpoints="<<kpoints;
	  break;
	case nomode: break;
	}
	idl ldosmap(n,n,layers,2.0,2.0,lenergy,henergy,0.0,-1.0,-1.0);
	ExecuteCPU0 ldosmap.setname(comment.str());
	/*}*/
      proctimings.addtimeentry("Initialization of calculation");
      size_t kgflayer=(size_t)-1;
#ifdef _mpi_version
      size_t layersperprocess=layers/world_size;
      if(layersperprocess*world_size<layers)
	layersperprocess++;
      size_t mpilayers=layersperprocess*world_size;
      double *data;
      if(world_rank)
	data=new double[n*n*layersperprocess];
      else data=new double[n*n*mpilayers];
      size_t startlayer=layersperprocess*world_rank,endlayer=startlayer+layersperprocess;
      if(endlayer>layers) endlayer=layers;
      for(size_t k=startlayer;k<endlayer;k++) {
#else
      for(size_t k=0;k<layers;k++) {
#endif
	if(k!=kgflayer) {
	  double energy=ldosmap.getbias(k);
#ifdef _mpi_version
	  cout<<world_rank<<"/"<<world_size<<": Calculating layer "<<k<<" at energy "<<std::fixed<<std::setprecision(4)<<energy<<"."<<endl;
#else
	  cout<<"Calculating layer "<<k<<" at energy "<<std::fixed<<std::setprecision(4)<<energy<<" - "<<flush;
#endif	  
	  switch(gf) {
	  case surfacegreen:
	    tm->calcsurfacegreensfunction(energy,epserr,true,smodel);
	    break;
	  case bulkgreen:
	    tm->calcsurfacegreensfunction(energy,epserr,false,smodel);
	    break;
	  case normalgreen:
	    tm->setgreensfunction(energy);
	    break;
	  }
	  //proctimings.addtimeentry("Calculation of Green's function");
	  if((mode!=spfmode)&&(mode!=uspfmode))
	    tm->calcrealspacegreensfunction();
	  //proctimings.addtimeentry("FFT of Green's function");
	}
	switch(mode) {
	case wanniermode : {
	  if(!isnan(threshold))
	    tm->calcwannierldos(oversamp,window,fls);
	  else tm->calcwannierldos(oversamp,window,wfs,maxbands);
	}
	  break;
	case josephsonmode: {
	  gsl_complex ocomp=gsl_complex_rect(ldosmap.getbias(k),etat);
	  gsl_complex tip=gsl_complex_div(gsl_complex_rect(0.0,-copysign(1.0,ldosmap.getbias(k))*M_PI*deltat),
					  gsl_complex_sqrt(gsl_complex_sub_real(gsl_complex_mul(ocomp,ocomp),deltat*deltat)));
	  tm->calcwannierjosephson(oversamp,window,wfs,tip);
	}
	  break;
	case spfmode:
	  tm->calcspf(maxbands);
	  break;
	case uspfmode:
	  tm->calcuspf(maxbands,posarr,prearr);
	  break;
	case nomode:
	  break;
	}
#ifdef _GPU
	//calculate GF for next layer while GPU does CLDOS calculation
#ifdef _mpi_version
	if(k+1<endlayer) {
#else
	if(k+1<layers) {
#endif
	  kgflayer=k+1;
	  double energy=ldosmap.getbias(kgflayer);
#ifdef _mpi_version
	  cout<<world_rank<<"/"<<world_size<<": Calculating layer "<<kgflayer<<" at energy "<<std::fixed<<std::setprecision(4)<<energy<<"."<<endl;
#else
	  cout<<"Calculating layer "<<kgflayer<<" at energy "<<std::fixed<<std::setprecision(4)<<energy<<" - "<<flush;
#endif	  
	  switch(gf) {
	  case surfacegreen:
	    tm->calcsurfacegreensfunction(energy,epserr,true,smodel);
	    break;
	  case bulkgreen:
	    tm->calcsurfacegreensfunction(energy,epserr,false,smodel);
	    break;
	  case normalgreen:
	    tm->setgreensfunction(energy);
	    break;
	  }
	  //proctimings.addtimeentry("Calculation of Green's function");
	  if((mode!=spfmode)&&(mode!=uspfmode))
	    tm->calcrealspacegreensfunction();
	  //proctimings.addtimeentry("FFT of Green's function");
	}
#endif
#ifdef _mpi_version
	tm->ldos2array(data+n*n*(k-startlayer));
	cout<<world_rank<<"/"<<world_size<<": layer "<<k<<" done."<<endl;
#else
	tm->ldos2idl(ldosmap,k);
	cout<<" done"<<endl;
#endif
	proctimings.addtimeentry("Calculation of CLDOS");
	}
#ifdef _mpi_version
	cout<<"Calculation finished on node "<<world_rank<<"/"<<world_size<<"."<<endl;
	MPI_Datatype type; //workaround for large data sets
	MPI_Type_contiguous( n*n, MPI_DOUBLE, &type );
	MPI_Type_commit(&type);
	if(world_rank)
	  MPI_Gather(data,layersperprocess,type,data,layersperprocess,type,0,MPI_COMM_WORLD);
	else {
	  MPI_Gather(MPI_IN_PLACE,layersperprocess,type,data,layersperprocess,type,0,MPI_COMM_WORLD);
	  cout<<"All calculations finished."<<endl;
	  ldosmap.setdata(data);
	  cout<<"Saving to "<<qpifile<<endl;
	  ldosmap>>qpifile.c_str();
	  cout<<"Done."<<endl;
	}
	MPI_Type_free(&type);
	delete data;
#else
	cout<<"Calculation finished."<<endl
	    <<"Saving to "<<qpifile<<endl;
	ldosmap>>qpifile.c_str();
	cout<<"Done."<<endl;
#endif
      proctimings.addtimeentry("Saving data");
    }
      break;
	  case nomode: break;
    }
    }
#ifdef _mpi_version
      MPI_Finalize();
#endif
      delete tm;
      delete model;
      ExecuteCPU0 {
	proctimings.printtimings(cout);
      }
      cerr.flush(); cout.flush();
      cerr.rdbuf(oldcerrstreambuf); cout.rdbuf(oldcoutstreambuf);
      logfilestr.close();
      if(atslog) delete atslog;
      return 0;
    }
  else help();
  return 1;
}






