#include <fstream>
#include <boost/date_time.hpp>
#include <boost/filesystem.hpp>
#include <boost/lexical_cast.hpp>
#include <tbb/tbb.h>
#include "data_source.hpp"


#include <cstdio>
#include <ctime>
double diffclock(timeval start,timeval end)
{
    double seconds  = end.tv_sec  - start.tv_sec;
    double useconds = end.tv_usec - start.tv_usec;
    double mtime = ((seconds) * 1000 + useconds/1000.0) + 0.5;
    return mtime;
} 


using namespace cuvnet;
namespace fs = boost::filesystem;

struct loop{
    typedef std::vector<cuv::tensor<float,cuv::host_memory_space> > res_t;
    folder_loader&   m_this;
    res_t&           m_res;
    cuvnet::filename_processor*    m_pp;
    loop(folder_loader* fl, res_t& r, cuvnet::filename_processor* pp):m_this(*fl),m_res(r),m_pp(pp){}
    void operator()(const tbb::blocked_range<unsigned int>& r)const;
};

folder_loader::folder_loader(const std::string& pathstr, bool recursive)
:m_idx(0){
    scan(pathstr, recursive);
}
void
folder_loader::scan(const std::string& pathstr, bool recursive){
    m_files.clear();
    fs::path folder(pathstr);

    std::vector<std::string> allowed_extensions;
    allowed_extensions.push_back(".jpg");
    allowed_extensions.push_back(".JPG");
    allowed_extensions.push_back(".jpeg");
    allowed_extensions.push_back(".JPEG");

    std::vector<fs::path> todo;
    todo.push_back(folder);
    while(!todo.empty()){
        folder = todo.back(); todo.pop_back();
        for(fs::directory_iterator it = fs::directory_iterator(folder);
                it != fs::directory_iterator(); it++){
            //std::cout << "fl: "<<boost::lexical_cast<std::string>(*it)<<" ext:"<<fs::extension(*it)<<std::endl;

            if(fs::is_directory(*it)){
                if(recursive)
                    todo.push_back(*it);
                continue;
            }

            if(std::find(allowed_extensions.begin(),
                        allowed_extensions.end(),
                        fs::extension(*it))
                    != allowed_extensions.end()){
                m_files.push_back(cuvnet::detail::file_descr());
                m_files.back().name = it->path().string();
                m_files.back().size = fs::file_size(*it);
            }
        }
    }
    assert(m_files.size()>0);
}

void 
loop::operator()(const tbb::blocked_range<unsigned int>& r)const{
    //std::stringstream ss; ss<<r.begin()<<" "<<r.end()<<(size_t)this<<std::endl;
    //std::cout<<ss.str();
    for(unsigned int i=r.begin(); i!=r.end(); ++i){
        unsigned int idx = (m_this.m_idx+i)%m_this.m_files.size();
        cuvnet::detail::file_descr& fd = m_this.m_files[idx];
        std::cout << "fd.name:" << fd.name << std::endl;
        if(fd.content.size()!=fd.size){
            std::cout << "-> reading `" << fd.name<<"' bytes: "<<fd.size<< std::endl;
            fd.content.resize(fd.size);
            std::ifstream is;
            is.exceptions(std::ifstream::failbit | std::ifstream::badbit);
            is.open(fd.name.c_str(), std::ios::binary | std::ios::in);
            is.read(&fd.content[0], fd.size);
        }
        m_pp->process_filestring(m_res[i], &fd.content[0],fd.size);
    }
};

void
folder_loader::get(std::vector<cuv::tensor<float,cuv::host_memory_space> >& res, const unsigned int n, cuvnet::filename_processor* pp){
    res.resize(n);
    //timeval begin, end;
    //gettimeofday(&begin,NULL);
    //for (int i = 0; i < 60; ++i) {
        tbb::parallel_for(
                tbb::blocked_range<unsigned int>(0,n,4),
                loop(this,res,pp),
                tbb::simple_partitioner());
    //}
    //gettimeofday(&end,NULL);
    //std::cout << "Time per batch: " << double(diffclock(begin,end))/60 << " ms"<< std::endl;
    m_idx += n;
}
