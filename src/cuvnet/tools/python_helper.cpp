#include <iostream>
#include <sstream>
#include <boost/function.hpp>
#include <boost/mpl/vector.hpp>
#include <boost/filesystem.hpp>
#include <boost/format.hpp>
#include <cuvnet/ops.hpp>
#include <cuvnet/op_utils.hpp>
#include <cuvnet/tools/function.hpp>
#include <cuvnet/tools/gradient_descent.hpp>
#include <cuvnet/tools/normalization.hpp>
#include <boost/python.hpp>
#include <boost/python/register_ptr_to_python.hpp>
#include <boost/python/list.hpp>
#include "python_helper.hpp"

namespace cuvnet
{

    struct gradient_descent_wrap{
        static
            boost::shared_ptr<gradient_descent>
            init_wrapper(boost::shared_ptr<Op> loss,
                    unsigned int res,
                    boost::python::list l, float lr, float wd){
                std::vector<Op*> params;
                boost::python::ssize_t n = boost::python::len(l);
                for(boost::python::ssize_t i=0;i<n;i++) {
                    boost::python::object elem = l[i];
                    boost::shared_ptr<Op> p = boost::python::extract<boost::shared_ptr<Op> >(elem);
                    params.push_back(p.get());
                }
                return boost::make_shared<gradient_descent>(loss, res, params, lr, wd);
            }
    };


    // Parses the value of the active python exception
    // NOTE SHOULD NOT BE CALLED IF NO EXCEPTION
    std::string parse_python_exception(){
        PyObject *type_ptr = NULL, *value_ptr = NULL, *traceback_ptr = NULL;
        // Fetch the exception info from the Python C API
        PyErr_Fetch(&type_ptr, &value_ptr, &traceback_ptr);
        namespace py = boost::python;

        // Fallback error
        std::string ret("Unfetchable Python error");
        // If the fetch got a type pointer, parse the type into the exception string
        if(type_ptr != NULL){
            py::handle<> h_type(type_ptr);
            py::str type_pstr(h_type);
            // Extract the string from the boost::python object
            py::extract<std::string> e_type_pstr(type_pstr);
            // If a valid string extraction is available, use it 
            //  otherwise use fallback
            if(e_type_pstr.check())
                ret = e_type_pstr();
            else
                ret = "Unknown exception type";
        }
        // Do the same for the exception value (the stringification of the exception)
        if(value_ptr != NULL){
            py::handle<> h_val(value_ptr);
            py::str a(h_val);
            py::extract<std::string> returned(a);
            if(returned.check())
                ret +=  ": " + returned();
            else
                ret += std::string(": Unparseable Python error: ");
        }
        // Parse lines from the traceback using the Python traceback module
        if(traceback_ptr != NULL){
            py::handle<> h_tb(traceback_ptr);
            // Load the traceback module and the format_tb function
            py::object tb(py::import("traceback"));
            py::object fmt_tb(tb.attr("format_tb"));
            // Call format_tb to get a list of traceback strings
            py::object tb_list(fmt_tb(h_tb));
            // Join the traceback strings into a single string
            py::object tb_str(py::str("\n").join(tb_list));
            // Extract the string, check the extraction, and fallback in necessary
            py::extract<std::string> returned(tb_str);
            if(returned.check())
                ret += ": " + returned();
            else
                ret += std::string(": Unparseable Python traceback");
        }
        return ret;
    }


    struct OpWrap
        : public Op, public boost::python::wrapper<Op>
    {
        std::string this_ptr() const{ return boost::lexical_cast<std::string>(this);}
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

    std::string __str__(Op& o){
        detail::graphviz_node desc;
        o._graphviz_node_desc(desc);
        return o.get_group() + " // " + desc.label;
    }
    matrix evaluate_op(Op& o, boost::python::list l){
        cuvnet::function f(o.shared_from_this(), 0, "click");
        
        boost::python::ssize_t n = boost::python::len(l);
        for(boost::python::ssize_t i=0;i<n;i++) {
            boost::shared_ptr<Op> elem = boost::python::extract<boost::shared_ptr<Op> >(l[i]);
            f.add(elem->shared_from_this());
        }
        if(n>0)
            f.set_cleanup_temp_vars(false);

        return f.evaluate();
    }
    matrix evaluate_delta_sink(Op& loss, Op& o, int param){

        cuvnet::delta_function f(loss.shared_from_this(), o.shared_from_this(), 0, param, "click");
        f.forget();
        return f.evaluate();
    }
    matrix evaluate_sink(Sink& o, boost::python::list l){
        cuvnet::function f(o.param(0)->param_uses[0]->get_op(), 0, "click");
        
        boost::python::ssize_t n = boost::python::len(l);
        for(boost::python::ssize_t i=0;i<n;i++) {
            boost::shared_ptr<Op> elem = boost::python::extract<boost::shared_ptr<Op> >(l[i]);
            f.add(elem);
        }
        if(n>0)
            f.set_cleanup_temp_vars(false);

        return f.evaluate();
    }
    std::string dot(Op& o, bool verbose){
        std::string path = boost::filesystem::unique_path("%%%%-%%%%-%%%%-%%%%.dot").string();
        
        std::ostringstream os(path.c_str());
        std::vector<Op*> v;
        write_graphviz(o, os, verbose, v, v, &o);

        return os.str();
    }
    void set_data(boost::shared_ptr<ParameterInput> f, cuvnet::matrix& m){
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
    valid_shape_info get_vsi_3(const boost::shared_ptr<Op> a, const boost::shared_ptr<Op> b){
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
            .add_property("ptr", &OpWrap::this_ptr)
            .def("fprop", pure_virtual(&OpWrap::fprop))
            .def("bprop", pure_virtual(&OpWrap::bprop))
            .def("dot", dot, (arg("op"), arg("verbose")=false))
            .def("result", &OpWrap::result, (arg("index")=0), return_value_policy<return_by_value>())
            .def("param",  &OpWrap::param, (arg("index")=0),  return_value_policy<return_by_value>())
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
            .def("evaluate", &evaluate_op, (arg("additional_res")=boost::python::list()))
            .def("__str__", &__str__)
            .def(self == self)
            .def(self != self)
            ;

        class_<Sink, boost::shared_ptr<Sink>, bases<Op>, boost::noncopyable >("Sink", no_init)
            .add_property("cdata",
                    make_function(
                        &Sink::cdata,
                        return_internal_reference<>()))
            .add_property("name", 
                    make_function(
                        (const std::string& (Sink::*)()const)
                        &Sink::name,
                        return_value_policy<copy_const_reference>()))
            .def("evaluate", &evaluate_sink, (arg("additional_res")=boost::python::list()))
            ;
        class_<DeltaSink, boost::shared_ptr<DeltaSink>, bases<Op>, boost::noncopyable>("DeltaSink", no_init)
            .add_property("cdata",
                    make_function(
                        &DeltaSink::cdata,
                        return_internal_reference<>()))
            .add_property("name",
                    make_function(
                        (const std::string& (DeltaSink::*)()const)
                        &DeltaSink::name,
                        return_value_policy<copy_const_reference>()))
            ;

        class_<delta_function, boost::shared_ptr<delta_function>, boost::noncopyable>
            ("delta_function",init<boost::shared_ptr<Op>, boost::shared_ptr<Op>, int, int, const std::string&>())
                .def("evaluate", &delta_function::evaluate,
                        return_internal_reference<>())
                .def("forget", &delta_function::forget)
        ;

        class_<ParameterInput, boost::shared_ptr<ParameterInput>, bases<Op>, boost::noncopyable >("ParameterInput", no_init)
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
        class_<Noiser, boost::shared_ptr<Noiser>, bases<Op>, boost::noncopyable >("Noiser", no_init)
            .add_property("active", 
                    &Noiser::is_active,
                    &Noiser::set_active)
            .add_property("param", 
                    &Noiser::get_param,
                    &Noiser::set_param)
            ;
        register_ptr_to_python< boost::shared_ptr<Op> >();
        //register_ptr_to_python< boost::shared_ptr<ParameterInput> >(); // gives warning...

        //implicitly_convertible<boost::shared_ptr<ParameterInput>, boost::shared_ptr<Op> >();
        //implicitly_convertible<boost::shared_ptr<Sink>, boost::shared_ptr<Op> >();

        typedef cuvnet::detail::op_result<matrix> result_t;
        typedef cuvnet::detail::op_param<matrix> param_t;

        class_<result_t, boost::shared_ptr<result_t> >("Result", no_init)
            .add_property("n_uses", &result_t::n_uses)
            .add_property("op",     &result_t::get_op)
            .def("use", (boost::shared_ptr<param_t> (result_t::*)(unsigned int))&result_t::use, (arg("index")=0u))
            ;

        class_<param_t, boost::shared_ptr<param_t> >("Param", no_init)
            .add_property("n_uses", &param_t::n_uses)
            .add_property("op", make_function(
                        &param_t::get_op, return_internal_reference<>()))
            .def("use", (boost::shared_ptr<result_t>& (param_t::*)(unsigned int))&param_t::use, return_value_policy<return_by_value>(), (arg("index")=0))
            ;

        register_ptr_to_python< boost::shared_ptr<result_t> >();
        register_ptr_to_python< boost::shared_ptr<param_t> >();


        def("get_valid_shape_info", get_vsi_1);
        def("get_valid_shape_info", get_vsi_2);
        def("get_valid_shape_info", get_vsi_3);

        def("project_to_mean", project_to_mean<cuv::dev_memory_space>);
        def("project_to_mean", project_to_mean<cuv::host_memory_space>);
        def("project_to_unit_ball", project_to_unit_ball<cuv::dev_memory_space>);
        def("project_to_unit_ball", project_to_unit_ball<cuv::host_memory_space>);

        class_<swiper, boost::shared_ptr<swiper> >("swiper", no_init)
            .def("dump", &swiper::dump)
            .def("fprop", &swiper::fprop)
            .def("bprop", &swiper::bprop
                    , boost::python::default_call_policies()
                    , (arg("set_last_delta_to_one")=true))
            ;

        class_<gradient_descent, boost::shared_ptr<gradient_descent>, boost::noncopyable >
            ("gradient_descent", no_init)
            .def("__init__", make_constructor(
                        &gradient_descent_wrap::init_wrapper
                        , boost::python::default_call_policies()
                        , (arg("loss"), arg("result")=0, arg("params")=boost::python::list(),
                         arg("lr")=0.f, arg("wd")=0.f)))
            .add_property("swiper",
                    make_function(
                        &gradient_descent::get_swiper,
                        return_internal_reference<>()))
            ;

        class_<std::pair<float, float> >("float_pair", init<float, float>())
            .def_readwrite("first", &std::pair<float, float>::first)
            .def_readwrite("second", &std::pair<float, float>::second)
            ;

        class_<valid_shape_info>("valid_shape_info", init<boost::shared_ptr<Op>,boost::shared_ptr<Op> >())
            .def_readonly("o2i_scale",&valid_shape_info::o2i_scale)
            .def_readonly("i_margin_l",&valid_shape_info::i_margin_r)
            .def_readonly("i_margin_r",&valid_shape_info::i_margin_r)
            .def("o2i",&valid_shape_info::o2i)
            .def("i2o",&valid_shape_info::i2o)
            ;
    }
    int export_ops(){
        using namespace boost::python;
        try{
            object main_module = import("__main__");
            scope main(main_module);

            export_module();
            object main_namespace = main_module.attr("__dict__");
            //object ignored = exec(
            //        "import visualization\n"
            //        "visualization.valid_shape_info = valid_shape_info\n"
            //        "visualization.get_valid_shape_info = get_valid_shape_info\n",
            //        main_namespace);
        }catch(const boost::python::error_already_set&){
            std::string perror_str = parse_python_exception();
            throw std::runtime_error("python failure in export_ops: " + perror_str);
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
            std::string perror_str = parse_python_exception();
            throw std::runtime_error("python failure in export_op: " + perror_str);
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
                "import cuv_python as cp\n",
                main_namespace);
        }catch(const boost::python::error_already_set&){
            std::string perror_str = parse_python_exception();
            throw std::runtime_error("python failure in initialize_python: " + perror_str);
        }
        return 0;
    }

    int embed_python(bool IPython){
        using namespace boost::python;
        try{
            object main_module = import("__main__");
            object main_namespace = main_module.attr("__dict__");
            exec("import sys\nsys.argv = ['cuvnet.py']\n", main_namespace); // otherwise, embed() fails below!
            if(!IPython){
                exec("import os\nexecfile(os.path.expanduser('~/.pythonrc'))\n", main_namespace);
                exec("import pdb\npdb.set_trace()()\n", main_namespace);
            }else{
                exec("import IPython\nIPython.embed()\n", main_namespace);
            }
        }catch(const boost::python::error_already_set&){
            std::string perror_str = parse_python_exception();
            throw std::runtime_error("python failure in embed_python: " + perror_str);
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
            std::string perror_str = parse_python_exception();
            throw std::runtime_error("python failure in export_loadbatch: " + perror_str);
        }
        return 0;
        
    }
}
