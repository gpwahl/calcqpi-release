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
 
#ifndef _recordtimings_h
#define _recordtimings_h
#include <sstream>
#include <string>
#include <time.h>
#include <list>
#include <iterator>
#include <iostream>
#include <iomanip>

using namespace std;

struct timeentry {
  struct timespec realtime,cputime;
  string sectionname;
  size_t labelno;
};
  
class recordtimings {
  struct timespec lastrealtime,lastcputime;
  list<timeentry> timings;
  string timestring(struct timespec &tdiff) {
    const time_t daytick=(time_t)86400,
      hourtick=(time_t)3600,
      minutetick=(time_t)60,
      secondtick=(time_t)1;
    size_t timediff=tdiff.tv_sec;
    size_t days=(size_t)(timediff/daytick),
      hours=(size_t)((timediff%daytick)/hourtick),
      minutes=(size_t)((timediff%hourtick)/minutetick),
      seconds=(size_t)((timediff%minutetick)/secondtick),
      microseconds=(size_t)(tdiff.tv_nsec/1000);
    ostringstream timestr;
    timestr<<setw(2)<<days<<"d, "<<setw(2)<<hours<<"h, "<<setw(2)<<minutes<<"m, "<<setw(2)<<seconds<<"s"<<setw(7)<<microseconds<<"μs";
    return timestr.str();
  }
  struct timespec difftime(struct timespec &newtime, struct timespec &lasttime)
  {
    struct timespec timediff;
    timediff.tv_sec=newtime.tv_sec-lasttime.tv_sec;
    if(newtime.tv_nsec<lasttime.tv_nsec) {
      timediff.tv_sec--;
      timediff.tv_nsec=((size_t)(1e9)+newtime.tv_nsec-lasttime.tv_nsec);
    } else
      timediff.tv_nsec=newtime.tv_nsec-lasttime.tv_nsec;
    return timediff;
  }
  void addtime(struct timespec &acctime, struct timespec &addtime)
  {
    acctime.tv_nsec+=addtime.tv_nsec;
    acctime.tv_sec+=addtime.tv_sec+(acctime.tv_nsec)/1e9;
    acctime.tv_nsec=acctime.tv_nsec%(size_t)(1e9);
  }
  double timefraction(struct timespec &parttime, struct timespec &totaltime)
  {
    double parttimes=(double)parttime.tv_sec+parttime.tv_nsec*1.0e-9,
      totaltimes=(double)totaltime.tv_sec+totaltime.tv_nsec*1.0e-9;
    return parttimes/totaltimes;
  }
 public:
  recordtimings() {
    clock_gettime(CLOCK_REALTIME,&lastrealtime);
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID,&lastcputime);
  }
  void addtimeentry(string text) {
    struct timespec newrealtime,newcputime;
    clock_gettime(CLOCK_REALTIME,&newrealtime);
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID,&newcputime);
    struct timespec realdiff=difftime(newrealtime,lastrealtime), cpudiff=difftime(newcputime,lastcputime);
    timeentry newentry={realdiff,cpudiff,text,0};
    timings.push_back(newentry);
    lastrealtime=newrealtime;
    lastcputime=newcputime;
  }
  void printtimings(ostream &os) {
    struct timespec totalcputime={0,0},totalrealtime={0,0};
    std::list<timeentry>::iterator it;
    for(it=timings.begin();it!=timings.end();it++) {
      addtime(totalcputime,it->cputime);
      addtime(totalrealtime,it->realtime);
      std::list<timeentry>::iterator itin=it;
      for(itin++;itin!=timings.end();itin++)
	if(!itin->labelno)
	  if(it->sectionname==itin->sectionname) {
	    addtime(it->realtime,itin->realtime);
	    addtime(it->cputime,itin->cputime);
	    itin->labelno=1;
	  }
    }
    os<<"Summary of timings: "<<endl;
    os<<"     Section                            CPU time                               Real time"<<endl;
    os<<" ------------------------------------------------------------------------------------------------------------------"<<endl;
    for(it=timings.begin();it!=timings.end();it++)
      if(!it->labelno)
	os<<"     "<<setw(33)<<left<<it->sectionname<<" "<<timestring(it->cputime)<<" ("<<setprecision(2)<<setw(5)<<right<<timefraction(it->cputime,totalcputime)*100.0<<"%)   "<<timestring(it->realtime)<<" ("<<setprecision(2)<<setw(5)<<timefraction(it->realtime,totalrealtime)*100.0<<"%)"<<endl;
    os<<" ------------------------------------------------------------------------------------------------------------------"<<endl;
    os<<"     Total time elapsed                "<<timestring(totalcputime)<<"            "<<timestring(totalrealtime)<<endl;
  }
};

#endif
