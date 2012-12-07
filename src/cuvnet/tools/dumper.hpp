#ifndef __DUMPER_HPP__
#     define __DUMPER_HPP__

#include <boost/bind.hpp>
#include <boost/lambda/lambda.hpp>

#include <cuvnet/op_utils.hpp>
#include <cuvnet/ops.hpp>
#include <cuvnet/tools/gradient_descent.hpp>
#include <cuvnet/tools/monitor.hpp>
#include <cuv/libs/cimg/cuv_cimg.hpp>

#include <datasets/random_translation.hpp>
#include "cuvnet/models/relational_auto_encoder.hpp"
#include <cuvnet/models/auto_encoder_stack.hpp>
#include <cuvnet/models/simple_auto_encoder.hpp>
#include <cuvnet/models/logistic_regression.hpp>
#include <cuvnet/models/linear_regression.hpp>
#include <vector>
#include <map>
#include <exception>
#include <cuvnet/tools/dumper.hpp>

namespace cuvnet
{

    

class max_example_reached_exception: public std::exception{
};

/**
 * processes data through a function and writes the outputs and parameters to a stream.
 *
 * @deprecated
 * @ingroup tools
 */
class dumper{
typedef boost::shared_ptr<ParameterInput> input_ptr;
typedef cuv::tensor<float,cuv::host_memory_space> tensor_type;
typedef boost::shared_ptr<Op>     op_ptr;
typedef std::map<std::string,float>      map_type;
    private:
        /// file where we write the hidden activations and parameters
        std::ofstream m_logfile;

        /// if true the header needs to be written in the file
        bool need_header_log_file;
        
        /// the name of the file
        std::string m_file_name;

        /// id of each row
        unsigned int id;

    public:

        /**
         * constructor
         * 
         */
        dumper():
        need_header_log_file(true),
        id(0)
        {
        }


        /**
         * generate random patterns and loggs them to the file
         * 
         */
        void generate_log_patterns(op_ptr op, boost::function<map_type()> data_gen, std::string file_name, int num_data=INT_MAX){
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

        /**
         *  logs the activations and parameters to the file 
         * 
         */
        void log_to_file(matrix& src, map_type& param){
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
};










}

#endif /* __DUMPER_HPP__ */
