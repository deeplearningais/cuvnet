#include <iostream>
#include <sstream>
#include <boost/filesystem.hpp>
#include <boost/format.hpp>
#include <cuvnet/ops.hpp>
#include <cuvnet/op_utils.hpp>
#include <tools/function.hpp>
#include <boost/python/register_ptr_to_python.hpp>
#include "python_helper.hpp"

namespace cuvnet
{
    struct OpWrap
        : public Op, public boost::python::wrapper<Op>
    {
        void fprop(){
            this->get_override("fprop")();
        }
        void bprop(){
            this->get_override("fprop")();
        }
        bool need_derivative()const{
            if(boost::python::override f = this->get_override("need_derivative"))
                return f();
            return Op::need_derivative();
        } 
        bool need_result()const{
            if(boost::python::override f = this->get_override("need_result"))
                return f();
            return Op::need_result();
        }
        void _determine_shapes(){
            if(boost::python::override f = this->get_override("_determine_shapes"))
                f();
            Op::_determine_shapes();
        }
        void _graphviz_node_desc(detail::graphviz_node& desc)const{
            if(boost::python::override f = this->get_override("_graphviz_node_desc"))
                f(desc);
            Op::_graphviz_node_desc(desc);
        }
        void release_data(){
            if(boost::python::override f = this->get_override("release_data"))
                f();
            Op::release_data();
        }
    };

    struct Dummy{};
    matrix evaluate(Op& o){
        cuvnet::function f(o.shared_from_this(), 0, "click");
        return f.evaluate();
    }
    std::string dot(Op& o){
        std::string path = boost::filesystem::unique_path("%%%%-%%%%-%%%%-%%%%.dot").string();
        
        std::ostringstream os(path.c_str());
        std::vector<Op*> v;
        write_graphviz(o, os, v, &o);

        return os.str();
    }
    template<class T>
    inline T* get_node(const boost::shared_ptr<Op>& f, long pointer){
        return (T*) get_node<T>(f, (Op*)pointer);
    }

    void export_ops(){
        using namespace boost::python;
        object main_module = import("__main__");
        scope main(main_module);

        {   scope cn = class_<Dummy>("cuvnet")
            ;

                class_<Op, boost::shared_ptr<OpWrap>, boost::noncopyable >("Op", no_init)
                    .add_property("n_params", &OpWrap::get_n_params, &OpWrap::set_n_params)
                    .add_property("n_results", &OpWrap::get_n_results, &OpWrap::set_n_results)
                    .add_property("need_derivative", 
                            (bool (OpWrap::*)()const)& OpWrap::need_derivative,
                            (void (OpWrap::*)(bool)) & OpWrap::need_derivative)
                    .add_property("need_result", 
                            (bool (OpWrap::*)()const)& OpWrap::need_result,
                            (void (OpWrap::*)(bool)) & OpWrap::need_result)
                    .def("fprop", pure_virtual(&OpWrap::fprop))
                    .def("bprop", pure_virtual(&OpWrap::bprop))
                    .def("dot", dot)
                    .def("get_parameter", 
                            (ParameterInput* (*)(const boost::shared_ptr<Op>&, const std::string&)) get_node, 
                            return_internal_reference<1>())
                    .def("get_parameter", 
                            (ParameterInput* (*)(const boost::shared_ptr<Op>&, long)) get_node, 
                            return_internal_reference<1>())
                    .def("get_node", 
                            (Op* (*)(const boost::shared_ptr<Op>&, long)) get_node, 
                            return_internal_reference<1>())
                    .def("get_sink", 
                            (Sink* (*)(const boost::shared_ptr<Op>&, long)) get_node, 
                            return_internal_reference<1>())
                    .def("evaluate", &evaluate)
                    ;
                
                class_<Sink, boost::shared_ptr<Sink> >("Sink", no_init)
                    .add_property("cdata",
                        make_function(
                            &Sink::cdata,
                            return_internal_reference<>()))
                    .add_property("name", 
                            make_function(
                                (const std::string& (Sink::*)()const)
                                &Sink::name,
                                return_value_policy<copy_const_reference>()))
                    ;
                class_<ParameterInput, boost::shared_ptr<ParameterInput> >("ParameterInput", no_init)
                    .add_property("name", 
                            make_function(
                                (const std::string& (ParameterInput::*)()const)
                                &ParameterInput::name,
                                return_value_policy<copy_const_reference>()))
                    .add_property("derivable", 
                            &ParameterInput::derivable,
                            &ParameterInput::set_derivable)
                    .add_property("data", 
                            make_function(
                                (Op::value_type& (ParameterInput::*)())
                                &ParameterInput::data,
                                return_internal_reference<>()))
                    .add_property("delta", 
                        make_function(
                            (Op::value_type& (ParameterInput::*)())
                            &ParameterInput::delta,
                            return_internal_reference<>()))
                    ;
                ;
                register_ptr_to_python< boost::shared_ptr<Op> >();
                register_ptr_to_python< boost::shared_ptr<ParameterInput> >();

                class_<swiper>("swiper", no_init)
                    .def("fprop", &swiper::fprop)
                    .def("bprop", &swiper::bprop)
                    ;

        }
    }

    boost::python::object initialize_python(){
        using namespace boost::python;
        Py_Initialize();
        object main_module = import("__main__");
        object main_namespace = main_module.attr("__dict__");
        object ignored = exec(
                "import sys\n"
                "sys.argv = ['cuvnet.py']\n" // otherwise, embed() fails below!
                "import cuv_python as cp\n"
                "import sys\n"
                "sys.path.insert(0, '../src/scripts')\n"
                "import visualization\n"
                "import matplotlib.pyplot as plt\n",
                main_namespace);
        return main_namespace;
    }

    void embed_python(){
        using namespace boost::python;
        object main_module = import("__main__");
        object main_namespace = main_module.attr("__dict__");
        object ignored = exec(
                "from IPython import embed\n"
                "embed()\n",
                main_namespace);
    }
}
