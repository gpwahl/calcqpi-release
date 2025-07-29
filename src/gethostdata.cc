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

#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <pwd.h>

#include <string>
#include <vector>

using namespace std;

#ifdef _mpi_version
#include <mpi.h>
#endif

#include "mpidefs.h"

void gethostdata(vector<string> &hostnames,string &username)
{
#ifdef _mpi_version  
  if(world_rank==0) {
#endif
    char hostnamebuf[_POSIX_HOST_NAME_MAX];
    int result;
    result = gethostname(hostnamebuf, _POSIX_HOST_NAME_MAX);
    if (result) {
      hostnames.push_back(string(""));
    } else hostnames.push_back(string(hostnamebuf));
#ifdef _mpi_version
    size_t lastindex=0;
    for(int i=1;i<world_size;i++) {
      //receive MPI
      MPI_Recv(hostnamebuf,_POSIX_HOST_NAME_MAX,MPI_BYTE,i,0,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
      string hostname=string(hostnamebuf);
      if(hostname!=hostnames[lastindex]) {
	hostnames.push_back(hostname);
	lastindex++;
      }
    }
#endif
    uid_t uid=geteuid();
    struct passwd *pw=getpwuid(uid);
    if(pw)
      username=string(pw->pw_name);
    else username=string("");
    // char usernamebuf[_POSIX_LOGIN_NAME_MAX];
    // result = getlogin_r(usernamebuf, _POSIX_LOGIN_NAME_MAX);
    // if (result) {
    //   username=string("");
    // } else
    //   username=string(usernamebuf);
#ifdef _mpi_version
  } else {
    char hostnamebuf[_POSIX_HOST_NAME_MAX];
    int result;
    result = gethostname(hostnamebuf, _POSIX_HOST_NAME_MAX);
    if (result)
      *hostnamebuf=0;
    MPI_Send(hostnamebuf,_POSIX_HOST_NAME_MAX,MPI_BYTE,0,0,MPI_COMM_WORLD);
  }
#endif
}
