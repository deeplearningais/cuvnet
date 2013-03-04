#include <iostream>
#include <sstream>
#include <boost/function.hpp>
#include <boost/mpl/vector.hpp>
#include <boost/filesystem.hpp>
#include <boost/format.hpp>
#include <cuvnet/ops.hpp>
#include <cuvnet/op_utils.hpp>
#include <cuvnet/tools/function.hpp>
#include <boost/python.hpp>
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

    boost::shared_ptr<ParameterInput>
        create_param_input_with_list(boost::python::list l, std::string name){
            std::vector<unsigned int> shape;
            for (int i = 0; i < boost::python::len(l); ++i) {
                shape.push_back(boost::python::extract<int>(l[i]));
            } 
            return boost::make_shared<ParameterInput>(shape, name);
        }

    struct Dummy{};
    matrix evaluate_op(Op& o){
        cuvnet::function f(o.shared_from_this(), 0, "click");
        return f.evaluate();
    }
    matrix evaluate_sink(Sink& o){
        cuvnet::function f(o.param(0)->param_uses[0]->get_op(), 0, "click");
        return f.evaluate();
    }
    std::string dot(Op& o){
        std::string path = boost::filesystem::unique_path("%%%%-%%%%-%%%%-%%%%.dot").string();
        
        std::ostringstream os(path.c_str());
        std::vector<Op*> v;
        write_graphviz(o, os, v, &o);

        return os.str();
    }
    void set_data(boost::shared_ptr<ParameterInput>& f, cuvnet::matrix& m){
        f->data() = m;
    }

    template<class T>
    inline T* get_node(const boost::shared_ptr<Op>& f, long pointer){
        return (T*) get_node<T>(f, (Op*)pointer);
    }

    valid_shape_info get_vsi_1(const boost::shared_ptr<ParameterInput> a, const boost::shared_ptr<ParameterInput> b){
        return valid_shape_info(a,b);
    }
    valid_shape_info get_vsi_2(const boost::shared_ptr<ParameterInput> a, const boost::shared_ptr<Op> b){
        return valid_shape_info(a,b);
    }

    bool operator==(const Op& a, const Op& b){
        return &a == &b;
    }
    bool operator!=(const Op& a, const Op& b){
        return &a != &b;
    }

    void export_module(){
        using namespace boost::python;
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
            .def("get_sink", 
                    (Sink* (*)(const boost::shared_ptr<Op>&, const std::string&)) get_node, 
                    return_internal_reference<1>())
            .def("evaluate", &evaluate_op)
            .def(self == self)
            .def(self != self)
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
            .def("evaluate", &evaluate_sink)
            ;
        class_<ParameterInput, boost::shared_ptr<ParameterInput> >("ParameterInput", no_init)
            .def("__init__", make_constructor(&create_param_input_with_list))
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
                        return_internal_reference<>()),
                        set_data
                    )
            .add_property("delta", 
                    make_function(
                        (Op::value_type& (ParameterInput::*)())
                        &ParameterInput::delta,
                        return_internal_reference<>()))
            ;
        ;
        register_ptr_to_python< boost::shared_ptr<Op> >();
        //register_ptr_to_python< boost::shared_ptr<ParameterInput> >(); // gives warning...

        implicitly_convertible<boost::shared_ptr<ParameterInput>, boost::shared_ptr<Op> >();

        def("get_valid_shape_info", get_vsi_1);
        def("get_valid_shape_info", get_vsi_2);

        class_<swiper>("swiper", no_init)
            .def("fprop", &swiper::fprop)
            .def("bprop", &swiper::bprop)
            ;

        class_<valid_shape_info>("valid_shape_info", init<boost::shared_ptr<Op>,boost::shared_ptr<Op> >())
            .def_readonly("crop_h",&valid_shape_info::crop_h)
            .def_readonly("crop_w",&valid_shape_info::crop_w)
            .def_readonly("scale_h",&valid_shape_info::scale_h)
            .def_readonly("scale_w",&valid_shape_info::scale_w)
            ;
    }
    int export_ops(){
        using namespace boost::python;
        try{
            object main_module = import("__main__");
            scope main(main_module);

            export_module();
            object main_namespace = main_module.attr("__dict__");
            object ignored = exec(
                    "visualization.valid_shape_info = valid_shape_info\n"
                    "visualization.get_valid_shape_info = get_valid_shape_info\n",
                    main_namespace);
        }catch(const boost::python::error_already_set&){
            PyErr_PrintEx(0);
            throw std::runtime_error("python failure in export_ops");
        }
        return 0;
    }

    int export_op(const std::string& name, boost::shared_ptr<Op> op){
        using namespace boost::python;
        try{
            object main_module = import("__main__");
            object main_namespace = main_module.attr("__dict__");
            main_namespace[name] = op;
        }catch(const boost::python::error_already_set&){
            PyErr_PrintEx(0);
            throw std::runtime_error("python failure in export_op");
        }
        return 0;
    }

    int initialize_python(){
        using namespace boost::python;
        try{
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
        }catch(const boost::python::error_already_set&){
            PyErr_PrintEx(0);
            throw std::runtime_error("python failure");
        }
        return 0;
    }

    int embed_python(){
        using namespace boost::python;
        try{
            object main_module = import("__main__");
            object main_namespace = main_module.attr("__dict__");
            object ignored = exec(
                    "from IPython import embed\n"
                    "embed()\n",
                    main_namespace);
        }catch(const boost::python::error_already_set&){
            PyErr_PrintEx(0);
            throw std::runtime_error("python failure in embed_python");
        }
        return 0;
    }

    int export_loadbatch(
            boost::function<void(unsigned int)> load_batch, 
            boost::function<unsigned int(void)> n_batches
            ){
        using namespace boost::python;
        try{
            object main_module = import("__main__");
            scope main(main_module);
            def("load_batch", 
                    make_function(
                        load_batch, 
                        default_call_policies(),
                        boost::mpl::vector<void, unsigned int>()));
            def("n_batches", 
                    make_function(
                        n_batches, 
                        default_call_policies(),
                        boost::mpl::vector<unsigned int>()));
        }catch(const boost::python::error_already_set&){
            PyErr_PrintEx(0);
            throw std::runtime_error("python failure in export_loadbatch");
        }
        return 0;
        
    }
}
