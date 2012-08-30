#ifndef __DUMPER_HPP__
#     define __DUMPER_HPP__

#include <boost/bind.hpp>
#include <boost/lambda/lambda.hpp>

#include <cuvnet/op_utils.hpp>
#include <cuvnet/ops.hpp>
#include <tools/gradient_descent.hpp>
#include <tools/monitor.hpp>
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
#include <tools/dumper.hpp>

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
typedef std::map<std::string,int>      map_type;
    private:
        /// file where we write the hidden activations and parameters
        std::ofstream m_logfile;

        /// if true the header needs to be written in the file
        bool need_header_log_file;
        
        /// the name of the file
        std::string m_file_name;

    public:

        /**
         * constructor
         * 
         */
        dumper():
        need_header_log_file(true)
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
                    matrix hidden_act = f.evaluate();
                    log_to_file(hidden_act, param); // write to file (repeatedly)
                }
            }catch(max_example_reached_exception){
            }

            m_logfile.close();
        }

        /**
         *  logs the activations and parameters to the file 
         * 
         */
        void log_to_file(matrix& hidden_act, map_type& param){
            using namespace std;
            assert(m_logfile.is_open());
            map_type::iterator it;
            if(need_header_log_file){
                for ( it=param.begin() ; it != param.end(); it++ ){
                    m_logfile << it->first << ",";
                }
                for (unsigned int i = 0; i < hidden_act.shape(1); ++i)
                {
                    if(i == hidden_act.shape(1) -1){
                        m_logfile << "h" << i;
                    }else{
                        m_logfile << "h" << i << ",";
                    }
                }
                need_header_log_file = false;
                m_logfile << std::endl;
            }

            for ( it=param.begin() ; it != param.end(); it++ ){
                m_logfile << it->second << ",";
            }
            for (unsigned int i = 0; i < hidden_act.shape(1); ++i)
            {
                if(i == hidden_act.shape(1) - 1){
                    m_logfile << hidden_act(0,i);
                }else{
                    m_logfile << hidden_act(0,i) << ",";
                }
            }

            m_logfile << std::endl;
        }
};










}

#endif /* __DUMPER_HPP__ */
