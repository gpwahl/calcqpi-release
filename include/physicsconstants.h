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

#ifndef _constants_h
#define _constants_h

#include <math.h>

//constants taken from NIST Reference on constants etc.
//http://physics.nist.gov/cuu/Constants/index.html
namespace SIunits {
  const double hbar=1.054571596e-34, //(Js)
      h=hbar*2*M_PI,            //(Js)
      e=1.602176462e-19,        //elementary charge (C)
      me=9.10938188e-31,        //electron mass (kg)
      mp=1.67262158e-27,        //proton mass (kg)
      mn=1.67492716e-27,        //neutron mass (kg)
      kb=1.3806503e-23,         //Boltzmann constant (J/K)
      mu_b=927.400899e-26,      //Bohr magneton (J/T)
      c=299792458,              //Speed of light (m/s)
      mu_0=4.0*M_PI*1.0e-7,     //mu_0 (N/A^2=VS/A)
      epsilon_0=1.0/mu_0/c/c,   //epsilon_0 (AS/Vm=A^2s^2/N/m^2) 
      g=9.80665,                //standard acceleration of gravity (m/s^2)
      G=6.6742e-11;             //constant of gravitation (m^3/kg/s^2)
}

namespace evunits {
  const double kb=SIunits::kb/SIunits::e,      //Boltzmann constant in eV/K 
      //me/hbar^2 in 1/(eV*Ang^2)
      m_o_hbar2=SIunits::me/SIunits::hbar/SIunits::hbar*SIunits::e/1.0e20,
      me_o_hbar2=m_o_hbar2,
      //mp/hbar^2 in 1/(eV*Ang^2)
      mp_o_hbar2=SIunits::mp/SIunits::hbar/SIunits::hbar*SIunits::e/1.0e20,
      //mn/hbar^2 in 1/(eV*Ang^2)
      mn_o_hbar2=SIunits::mp/SIunits::hbar/SIunits::hbar*SIunits::e/1.0e20,
      e=1.0,
      hbar=SIunits::hbar/SIunits::e*1.0e15,    //hbar in eV*fs        
      h=hbar*2.0*M_PI,                         //h in eV*fs
      hc=h*SIunits::c*1.0e-6,                  //hc in eV*nm
      mu_b=SIunits::mu_b/SIunits::e,           //bohr magneton mu_b in eV/T
      //1/(4 pi epsilon_0) in eV*Ang
      coulomb_pre=SIunits::e*1.0e10/4.0/M_PI/SIunits::epsilon_0,
      //conversion atomic units to Angstroem
      au_in_ang=1.0/me_o_hbar2/coulomb_pre;
}

const double avogadro=6.02214199e23; //in mol^-1

#endif
