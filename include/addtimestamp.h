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

#ifndef _addtimestamp_h
#define _addtimestamp_h

#include <string>
#include <iostream>
#include <streambuf>
#include <time.h>
#include <assert.h>

string timestamp()
{
  time_t now=time(0);
  struct tm tstruct;
  char buf[80];
  tstruct=*localtime(&now);
  strftime(buf,sizeof(buf),"%Y-%m-%d.%X ", &tstruct);
  return string(buf);
}

class AddTimeStamp : public std::streambuf
{
public:
    AddTimeStamp( std::basic_ios< char >& out )
        : out_( out )
        , sink_()
        , newline_( true )
    {
        sink_ = out_.rdbuf( this );
        assert( sink_ );
    }
    ~AddTimeStamp()
    {
        out_.rdbuf( sink_ );
    }
protected:
    int_type overflow( int_type m = traits_type::eof() )
    {
        if( traits_type::eq_int_type( m, traits_type::eof() ) )
            return sink_->pubsync() == -1 ? m: traits_type::not_eof(m);
        if( newline_ )
        {   // --   add timestamp here
            std::ostream str( sink_ );
            if( !(str << timestamp()) ) // add perhaps a seperator " "
                return traits_type::eof(); // Error
        }
        newline_ = traits_type::to_char_type( m ) == '\n';
	int putcode=sink_->sputc( m );
	if(newline_) sink_->pubsync();
        return putcode;
    }
private:
    AddTimeStamp( const AddTimeStamp& );
    AddTimeStamp& operator=( const AddTimeStamp& ); // not copyable
    // --   Members
    std::basic_ios< char > &out_;
    std::streambuf* sink_;
    bool newline_;
};

#endif
