#include "dumper.hpp"
#include <cuvnet/tools/function.hpp>
namespace cuvnet
{

    dumper::dumper():
        need_header_log_file(true),
        id(0)
    {
    }
    void dumper::generate_log_patterns(op_ptr op, boost::function<map_type()> data_gen, std::string file_name, int num_data){
        using namespace std;
        // open file
        m_logfile.open(file_name.c_str(), std::ios::out);
        cuvnet::function f(op);

        try{
            for(int i=0;i<num_data; i++){
                map_type param = data_gen();
                matrix res = f.evaluate();
                log_to_file(res, param); // write to file (repeatedly)
            }
        }catch(max_example_reached_exception){
        }

        m_logfile.close();
    }

    void dumper::log_to_file(matrix& src, map_type& param){
        using namespace std;
        assert(m_logfile.is_open());
        map_type::iterator it;
        if(need_header_log_file){
            m_logfile << "id";
            for ( it=param.begin() ; it != param.end(); it++ )
                m_logfile  << "," << it->first;
            unsigned int s = src.size();
            if(s > 0)
                m_logfile << ",h0";
            for (unsigned int i = 1; i < s; ++i)
                m_logfile << ",h" << i;
            need_header_log_file = false;
            m_logfile << std::endl;
        }

        m_logfile << id;
        for ( it=param.begin() ; it != param.end(); it++ )
            m_logfile  << "," << it->second;
        unsigned int s = src.size();
        if(s>0)
            m_logfile << "," << src[0];
        for (unsigned int i = 1; i < s; ++i)
            m_logfile << "," << src[i];

        m_logfile << std::endl;
        id++;
    }
}
