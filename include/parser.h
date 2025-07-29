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

#ifndef _parser_h
#define _parser_h

#include <iostream>
#include <fstream>
#include <stdlib.h>

using namespace std;

class loadconfig {
  vector<string> name, field, comments;
  vector<size_t> used;
  bool error,linemode;
  size_t currentline;
  void readfile(istream &is) {
    size_t lineno=0;
    while(!is.eof()) {
      string line;     
      getline(is,line);
      if(line.size()>0) {
	lineno++;
	size_t eqsign,semicolon,hashtag;
	hashtag=line.find('#');
	if(hashtag!=string::npos) {
	  comments.push_back(line.substr(hashtag+1));
	  line.erase(hashtag);
	}
	eqsign=line.find('=');
	semicolon=line.find(';');
	if((eqsign!=string::npos)&&(semicolon!=string::npos)) {
	  name.push_back(line.substr(0,eqsign));
	  field.push_back(line.substr(eqsign+1,semicolon-eqsign-1));
	} else
	  if((eqsign!=string::npos) || (semicolon!=string::npos)) {
	    cerr<<"Incorrect syntax, '=' or ';' missing."<<endl
		  <<"Line "<<lineno<<": "<<line<<endl;
	  } 
      }
    }
  }
  string field2string(const string &value) {
    size_t first=value.find_first_of('\"'), last=value.find_last_of('\"');
    if((first!=string::npos)&&(last!=string::npos))
      return value.substr(first+1,last-first-1);
    else return string("");
  }
public:
  loadconfig(char *name, bool linemode=false):linemode(linemode) {
    error=false; currentline=0;
    if(string(name)=="-")
      readfile(cin);
    else {
      ifstream is;
      is.exceptions(ifstream::failbit|ifstream::failbit|ifstream::badbit);
      try {
	is.open(name);
      } catch (ifstream::failure const &e) {
	cerr<<"Error while opening file "<<name<<" for reading."<<endl
	    <<e.what()<<endl;
	error=true;
	return;
      }
      is.exceptions(ifstream::goodbit);
      readfile(is);
      is.close();
    }
    used.assign(this->name.size(),0);
  }
  friend bool operator!(loadconfig &lf);
  void showconfig(ostream &os) {
    for(size_t i=0;i<comments.size();i++)
      os<<"#"<<comments[i]<<endl;
    for(size_t i=0;i<name.size();i++)
      os<<name[i]<<"="<<field[i]<<";"<<endl;
  }
  void saveconfig(char *name) {
    ofstream os(name);
    showconfig(os);
  }
  bool nextline() {
    if(currentline<name.size()) currentline++;
    if(currentline==name.size()) return false;
    else return true;
  }
  bool endoffile() {
    if(currentline==name.size()) return true;
    else return false;
  }
  bool probefield(const string &varname) {
    if(linemode) {
      if(currentline<name.size()) if(name[currentline]==varname) return true;
    } else
      for(size_t i=0;i<name.size(); i++)
	if(name[i]==varname) {
	  used[i]++;
	  return true;
	}
    return false;
  }
  string getfield(const string &varname) {
    string fieldval=string("");
    if(linemode) {
      if(currentline<name.size()) if(name[currentline]==varname) fieldval=field[currentline];
    } else
      for(size_t i=0;i<name.size(); i++)
	if(name[i]==varname) {
	  used[i]++;
	  fieldval=field[i];
	}
    if(fieldval.size()==0) return string("");
    else {
      size_t startindex=fieldval.find_first_not_of(" \t"),
	endindex=fieldval.find_last_not_of(" \t");
      if(endindex==string::npos) endindex=fieldval.length()-1;
      return fieldval.substr(startindex,endindex-startindex+1);
    }
  }
  string getstring(const string &varname) {
    string result=getfield(varname);
    return field2string(result);
  }
  double getvalue(const string &varname) {
    string val=getfield(varname);
    if(val.size()>0)
      return atof(val.c_str());
    else
      return NAN;
  }
  int getintvalue(const string &varname) {
    string val=getfield(varname);
    if(val.size()>0)
      return atoi(val.c_str());
    else
      return 0;
  }
  vector<string> getlist(const string &varname) {
    string val=getfield(varname);
    size_t first=val.find_first_of('('), last=val.find_last_of(')');
    vector<string> result;
    if((first!=string::npos)&&(last!=string::npos)) {
      size_t index=first+1,nindex;
      do {
	nindex=val.find_first_of(",)",index);
	size_t startindex=val.find_first_not_of(" \t",index),
	  endindex=val.find_last_not_of(" \t",nindex-1,nindex-index);
	if(endindex==string::npos) endindex=nindex-1;
	result.push_back(val.substr(startindex,endindex-startindex+1));
	index=nindex+1;
      } while(nindex!=last);
    }
    return result;
  }
  vector<string> getstringlist(const string &varname) {
    vector<string> varlist=getlist(varname),result;
    for(size_t i=0;i<varlist.size();i++)
      result.push_back(field2string(varlist[i]));
    return result;
  }
  vector<double> getvector(const string &varname) {
    vector<string> val=getlist(varname);
    vector<double> result;
    result.assign(val.size(),0.0);
    for(size_t i=0;i<val.size();i++)
      result[i]=atof(val[i].c_str());
    return result;
  }
  vector<int> getintvector(const string &varname) {
    vector<string> val=getlist(varname);
    vector<int> result;
    result.assign(val.size(),0);
    for(size_t i=0;i<val.size();i++)
      result[i]=atoi(val[i].c_str());
    return result;
  }
  string getvarname() {
    return name[currentline];
  }
  ~loadconfig() {
    if(!linemode)
      for(size_t i=0;i<used.size();i++)
	if(!used[i]) cerr<<"Unused line "<<i<<": "<<name[i]<<endl;
  }
};

bool operator!(loadconfig &lf) {return lf.error;}

#endif
