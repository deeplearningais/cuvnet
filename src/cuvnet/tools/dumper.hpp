#ifndef __DUMPER_HPP__
#     define __DUMPER_HPP__

#include <boost/bind.hpp>
#include <boost/lambda/lambda.hpp>
#include <boost/function.hpp>
#include <cuvnet/op_utils.hpp>
#include <cuvnet/ops.hpp>

#include <vector>
#include <map>
#include <exception>



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
        dumper();


        /**
         * Generate random patterns and loggs them to the file. 
         * For example, op is an encoder function, and data_gen is used to generate input data. 
         * op function will return tensor which are hidden unit activations. The function runs until the maximum number of examples
         * is generated "num_data", or if the exception "max_example_reached_exception" is thrown by "data_gen" function.
         *
         *
         * @param op            operator, which is input to cuvnet function, which evaluates the operator and returns tensor. 
         * @param data_gen      function which generates the data  
         * @param file_name     the name of the file where the data is stored   
         * @param num_data      the maximum number of examples which will be generated
         * 
         */
        void generate_log_patterns(op_ptr op, boost::function<map_type()> data_gen, std::string file_name, int num_data=INT_MAX);

        /**
         *  logs the activations and parameters to the file 
         *
         *  @param src data which is logged to file
         *  @param param parameters which were used to generate data are logged to file. 
         * 
         */
        void log_to_file(matrix& src, map_type& param);
};










}

#endif /* __DUMPER_HPP__ */
