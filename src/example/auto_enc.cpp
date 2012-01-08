// vim:ts=4:sw=4:et
#include <fstream>
#include <cuvnet/op.hpp>
#include <cuvnet/op_utils.hpp>
#include <cuvnet/ops/output.hpp>
#include <cuvnet/ops/input.hpp>
#include <cuvnet/ops/pow.hpp>
#include <cuvnet/ops/axpby.hpp>
#include <cuvnet/ops/tanh.hpp>
#include <cuvnet/ops/prod.hpp>
#include <cuvnet/ops/noiser.hpp>
#include <cuvnet/ops/sum.hpp>

using namespace cuvnet;
typedef boost::shared_ptr<Op> optr_t;
#define C1(Class,Arg) (boost::make_shared<Class>(Arg)->result())
#define C2(Class,A1,A2) (boost::make_shared<Class>(A1,A2))->result()
#define C3(Class,A1,A2,A3) (boost::make_shared<Class>(A1,A2,A3))->result()
#define C4(Class,A1,A2,A3,A4) (boost::make_shared<Class>(A1,A2,A3,A4))->result()

struct auto_encoder{
    boost::shared_ptr<Input>  m_input;
    boost::shared_ptr<Input>  m_weights;
    boost::shared_ptr<Op>     m_func;
    boost::shared_ptr<Output> m_out;

    auto_encoder(unsigned int inp0, unsigned int inp1, unsigned int hl, float std=0.1f)
    :m_input(new Input(cuv::extents[inp0][inp1]))
    ,m_weights(new Input(cuv::extents[inp1][hl]))
    {
        // Sum ( input - tanh( x W ) W' )^2
        m_func = 
            boost::make_shared<Sum>(
                    C2(Pow,2.f,
                        C4(Axpby,
                            m_input->result(),
                            C4(Prod,
                                C1(Tanh,
                                    C2(Prod,
                                        C2(Noiser, m_input->result(), std),
                                        m_weights->result())),
                                m_weights->result(),
                                'n','t'),
                            1.f,-1.f)));
        m_out = boost::make_shared<Output>(m_func->result());

        float diff = 4.f*std::sqrt(6./(inp1+hl));
        float mult = 2.f*diff;
        for(unsigned int i=0;i<hl*inp0;i++){
            m_weights->data()[i] = (float)(drand48()*mult-diff);
        }
    }
};

int main(int argc, char **argv)
{
    cuv::initialize_mersenne_twister_seeds();
    auto_encoder ae(3,4,5);

    for(unsigned int i=0;i<12;i++)
        ae.m_input->data()[i] = (float)(i%2);
    std::vector<Op*> params(1,ae.m_weights.get());
    swiper swipe(*ae.m_func,true,params);


    for(unsigned int epoch=0;epoch<1000;epoch++){
        swipe.fprop();
        {
            std::ofstream os("ae.dot");
            write_graphviz(*ae.m_out,os,swipe.m_topo.plist);
        }
        std::cout << ae.m_out->cdata()[0]<<std::endl;
        swipe.bprop();

        ae.m_weights->data() -= 0.001f*ae.m_weights->result()->delta.cdata();
    }

    return 0;
}
