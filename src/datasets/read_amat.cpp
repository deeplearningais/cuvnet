#include <iostream>
#include <vector>
#include <stdexcept>
#include <cstdio>
#include <zzip/lib.h>
#include <iosfwd>                          // streamsize
#include <boost/lexical_cast.hpp>
#include <boost/tokenizer.hpp>
#include <boost/algorithm/string.hpp>
#include <boost/iostreams/stream.hpp>
#include <boost/iostreams/stream_buffer.hpp>
#include <boost/iostreams/operations.hpp> 
#include <boost/iostreams/concepts.hpp>  // source_tag
#include "read_amat.hpp"

namespace io = boost::iostreams;
using namespace std;

class zipped_file {
    private:
        ZZIP_DIR* m_dir;
        ZZIP_FILE* m_file;
        friend class zipped_file_source;
    public:
        zipped_file(const std::string& zipfile, const std::string& zippedfile)
            :m_dir(NULL),m_file(NULL)
        {
            m_dir = zzip_dir_open(zipfile.c_str(),0);
            if(!m_dir)
                throw std::runtime_error("Could not open zip file`" + zipfile + "'");
            m_file = zzip_file_open(m_dir,zippedfile.c_str(), 0);
            if(!m_file)
                throw std::runtime_error("Could not open zipped file`" + zippedfile + "'");
        }
        ~zipped_file(){
            zzip_file_close(m_file);
            zzip_dir_close(m_dir);
        }
        std::streamsize read(char* s, std::streamsize n)
        {
            return zzip_file_read(m_file, s, n);
        }
};

class zipped_file_source : public io::source {
    private:
        zipped_file& m_zf;
    public:
        zipped_file_source(zipped_file& zf)
            :m_zf(zf)
        {
        }
        std::streamsize read(char* s, std::streamsize n)
        {
            return zzip_file_read(m_zf.m_file, s, n);
        }
};

template<class T>
struct amat_reader{
    std::vector<T> data;
    unsigned int   width;
    unsigned int   height;

    amat_reader(const std::string& zipfile_name, const std::string& zippedfile_name)
    :width(0),height(0){
        read(zipfile_name, zippedfile_name);
    }
    void read(const std::string& zipfile_name, const std::string& zippedfile_name){
        std::cout << "reading `"<<zippedfile_name<<"' from zip file `"<<zipfile_name<<"'..."<<std::flush;
        zipped_file zf(zipfile_name, zippedfile_name);
        zipped_file_source zfs(zf);
        io::stream<zipped_file_source> in(zfs);

        typedef boost::split_iterator<std::string::iterator> string_split_iterator;

        boost::char_separator<char> sep(" \t");
        typedef boost::tokenizer<boost::char_separator<char> > tokenizer;
        for(std::string line; std::getline(in, line);height++){
            unsigned int cnt=0;
            tokenizer tok(line, sep);
            for(tokenizer::iterator beg=tok.begin(); beg!=tok.end();++beg,++cnt)
            {
                data.push_back(boost::lexical_cast<T>(*beg));
            }
            if(width==0) width = cnt;
            else if(width!=cnt)
                throw std::runtime_error("line width does not match previous line width!");
        }
        std::cout << "done. width="<<width<<", height="<<height<<std::endl;
    }
};


namespace cuvnet{
    void read_amat(cuv::tensor<float,cuv::host_memory_space>& t, const std::string& zipfile_name, const std::string& zippedfile_name){
        amat_reader<float> ar(zipfile_name, zippedfile_name);
        t = cuv::tensor<float,cuv::host_memory_space>(cuv::indices[cuv::index_range(0,ar.height)][cuv::index_range(0,ar.width)], &ar.data[0]);
    }
    
    void read_amat_with_label(
            cuv::tensor<float,cuv::host_memory_space>& t,
            cuv::tensor<int,cuv::host_memory_space>& l,
            const std::string& zipfile_name, const std::string& zippedfile_name){
    
        amat_reader<float> ar(zipfile_name, zippedfile_name);
        t.resize(cuv::extents[ar.height][ar.width-1]);
        l.resize(cuv::extents[ar.height]);
        for(unsigned int i=0;i<ar.height;i++){
            std::memcpy(t.ptr()+i*(ar.width-1), &ar.data[i*ar.width], sizeof(float)*(ar.width-1));
            l[i] = ar.data[i*ar.width+ar.width-1];
        }
    
    }
}
