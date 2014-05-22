#include <vector>
#include <algorithm>

#include <boost/lexical_cast.hpp>
#include <cuvnet/tools/argsort.hpp>
#include <cuvnet/tools/preprocess.hpp>
#include <cuvnet/tools/orthonormalization.hpp>
#include <cuvnet/tools/normalization.hpp>

#include <boost/test/unit_test.hpp>

#define V(X) #X<<":"<<(X)<<", "

double nrand(){
    double  sum, f;
    int k;
    
    sum = 0.00;
    for (k=0; k<12; k++) {
        f = drand48();  /* uniform random [0,1) */
        sum += f;
    }
    sum -= 6;
    
    return sum;
}


BOOST_AUTO_TEST_SUITE( t_argsort )
BOOST_AUTO_TEST_CASE(normal){
    std::vector<float> v(50);
    std::generate(v.begin(),v.end(),drand48);
    std::vector<unsigned int> idx = argsort(v.begin(), v.end());
    BOOST_CHECK_EQUAL(50, idx.size());
    for(unsigned int i=1;i<50;i++){
        BOOST_CHECK_LT(v[idx[i-1]] , v[idx[i]]);
    }
}
BOOST_AUTO_TEST_CASE(inv){
    std::vector<float> v(50);
    std::generate(v.begin(),v.end(),drand48);
    std::vector<unsigned int> idx = argsort(v.begin(), v.end(), std::greater<float>());
    BOOST_CHECK_EQUAL(50, idx.size());
    for(unsigned int i=1;i<50;i++){
        BOOST_CHECK_GT(v[idx[i-1]] , v[idx[i]]);
    }
}
BOOST_AUTO_TEST_SUITE_END()

void
t_normalization_dim0(unsigned int n_src_maps, unsigned int n_flt_pix, unsigned int n_dst_maps){
    cuv::tensor<float,cuv::host_memory_space> m(cuv::extents[n_src_maps][n_flt_pix * n_dst_maps]);
    cuv::tensor<float,cuv::host_memory_space> s(cuv::extents[n_src_maps]);

    s = 0.f;

    unsigned int cnt = 0;
    for (unsigned int i = 0; i < m.shape(0); ++i)
    {
        for (unsigned int j = 0; j < m.shape(1); ++j)
        {
            float f = 10.f*drand48();
            m(i,j) = f;
            s(i) += f * f;
        }
    }
    float thresh = std::sqrt(cuv::mean(s));

    m.reshape(cuv::extents[n_src_maps][n_flt_pix][n_dst_maps]);
    cuvnet::project_to_unit_ball(m, 0, thresh);
    m.reshape(cuv::extents[n_src_maps][n_flt_pix * n_dst_maps]);
    for (unsigned int i = 0; i < m.shape(0); ++i)
    {
        float sum = 0.f;
        for (unsigned int j = 0; j < m.shape(1); ++j)
        {
            sum += m(i,j) * m(i,j);
        }
        if(s(i) > thresh * thresh){
            BOOST_CHECK_CLOSE(sum, thresh*thresh, 0.001f);
        }else{
            BOOST_CHECK_CLOSE(sum, (float) s(i), 0.001f);
        }
    }
}
void
t_normalization_dim2(unsigned int n_src_maps, unsigned int n_flt_pix, unsigned int n_dst_maps){
    cuv::tensor<float,cuv::host_memory_space> m(cuv::extents[n_src_maps * n_flt_pix][n_dst_maps]);
    cuv::tensor<float,cuv::host_memory_space> s(cuv::extents[n_dst_maps]);

    s = 0.f;

    unsigned int cnt = 0;
    for (unsigned int j = 0; j < m.shape(1); ++j)
    {
        for (unsigned int i = 0; i < m.shape(0); ++i)
        {
            float f = 10.f*drand48();
            m(i,j) = f;
            s(j) += f * f;
        }
    }
    float thresh = std::sqrt(cuv::mean(s));

    m.reshape(cuv::extents[n_src_maps][n_flt_pix][n_dst_maps]);
    cuvnet::project_to_unit_ball(m, 2, thresh);
    m.reshape(cuv::extents[n_src_maps * n_flt_pix][n_dst_maps]);
    for (unsigned int j = 0; j < m.shape(1); ++j)
    {
        float sum = 0.f;
        for (unsigned int i = 0; i < m.shape(0); ++i)
        {
            sum += m(i,j) * m(i,j);
        }
        if(s(j) > thresh * thresh){
            BOOST_CHECK_CLOSE(sum, thresh*thresh, 0.001f);
        }else{
            BOOST_CHECK_CLOSE(sum, (float) s(j), 0.001f);
        }
    }

    

}

BOOST_AUTO_TEST_SUITE( t_normalization )
BOOST_AUTO_TEST_CASE(dim0){
    t_normalization_dim0(16, 3*3, 16);
    t_normalization_dim0(7, 3*3, 7);
}
BOOST_AUTO_TEST_CASE(dim2){
    t_normalization_dim2(16, 3*3, 16);
    t_normalization_dim2(7, 3*3, 7);
}
BOOST_AUTO_TEST_SUITE_END()

BOOST_AUTO_TEST_SUITE( orthogonalization )
BOOST_AUTO_TEST_CASE(ortho_all){
    {
        cuv::tensor<float,cuv::host_memory_space> m(cuv::extents[5][2]);
        for (unsigned int i = 0; i < m.shape(0); ++i)
            for (unsigned int j = 0; j < m.shape(1); ++j)
                m(i,j) = drand48();
    
        // columns
        orthogonalize_symmetric(m,true);
        // scalar products between any two rows should now be 0.
        for (unsigned int i = 0; i < m.shape(1); ++i) {
            for (unsigned int j = i+1; j < m.shape(1); ++j) {
                float s = 0;
                for (unsigned int k = 0; k < m.shape(0); ++k)
                {
                    s += m(k,i)*m(k,j);
                }
                //BOOST_CHECK_CLOSE(0.f, s, 0.001f);
                if(fabs(s) > 0.001f){
                    std::cout << "error: " << s << std::endl;
                    BOOST_CHECK_EQUAL(true,false);
                }
            }
        }
    }

    cuv::tensor<float,cuv::host_memory_space> m(cuv::extents[2][5]);
    // rows
    orthogonalize_symmetric(m);
    // scalar products between any two rows should now be 0.
    for (unsigned int i = 0; i < m.shape(0); ++i) {
        for (unsigned int j = i+1; j < m.shape(0); ++j) {
            float s = 0;
            for (unsigned int k = 0; k < m.shape(1); ++k)
            {
                s  += m(i,k)*m(j,k);
            }
            //BOOST_CHECK_CLOSE(0.f, s, 0.001f);
            if(fabs(s) > 0.001f){
                std::cout << "error: " << s << std::endl;
                BOOST_CHECK_EQUAL(true,false);
            }
        }
    }
    
}

BOOST_AUTO_TEST_CASE(pairs){
    cuv::tensor<float,cuv::dev_memory_space> m(cuv::extents[16*2][16*2]);
    for (unsigned int i = 0; i < m.shape(0); ++i)
    {
        for (unsigned int j = 0; j < m.shape(1); ++j)
        {
            m(i,j) = drand48();
        }
    }

    // columns
    orthogonalize_pairs(m,true);
    // scalar products between any two rows should now be 0.
    for (unsigned int i = 0; i < m.shape(1)-1; i+=2) {
        int j   = i+1;
        float s = 0;
        for (unsigned int k = 0; k < m.shape(0); ++k)
        {
            s += m(k,i)*m(k,j);
        }
        //BOOST_CHECK_CLOSE(0.f, s, 0.001f);
        if(fabs(s) > 0.001f){
            std::cout << "error: " << s << std::endl;
            BOOST_CHECK_EQUAL(true,false);
        }
    }

    // rows
    orthogonalize_pairs(m);
    // scalar products between any two rows should now be 0.
    for (unsigned int i = 0; i < m.shape(0)-1; i+=2) {
        int j   = i+1;
        float s = 0;
        for (unsigned int k = 0; k < m.shape(1); ++k)
        {
            s  += m(i,k)*m(j,k);
        }
        //BOOST_CHECK_CLOSE(0.f, s, 0.001f);
        if(fabs(s) > 0.001f){
            std::cout << "error: " << s << std::endl;
            BOOST_CHECK_EQUAL(true,false);
        }
    }
    
}
BOOST_AUTO_TEST_SUITE_END()

BOOST_AUTO_TEST_SUITE( pca )
BOOST_AUTO_TEST_CASE(normal){
    typedef cuv::tensor<float,cuv::host_memory_space> tens_t;

    cuvnet::pca_whitening pp;

    tens_t data (10, 3);
    for (unsigned int i = 0; i < data.shape(0); ++i)
    {
        data(i,0) = (nrand()-0.5) + 100;
        data(i,1) = (nrand()-0.5) + 110;
        data(i,2) = (nrand()-0.5) + 120;
    }
    tens_t data2 = data.copy();

    pp.fit_transform(data);
    BOOST_CHECK_EQUAL(data.shape(0),data2.shape(0));
    BOOST_CHECK_EQUAL(data.shape(1),data2.shape(1));

    // check for identity matrix
    const tens_t&  rot = pp.rot();
    const tens_t& rrot = pp.rrot();
    tens_t res(cuv::extents[rot.shape(0)][rot.shape(0)]);
    cuv::prod(res,rot,rrot);
    for(unsigned int i=0;i<rot.shape(0);i++){
        res(i,i)-=1.f;
    }
    BOOST_CHECK_CLOSE(1.f, 1.f+cuv::norm2(res), 0.01f);

    // check for reverse_transform
    pp.reverse_transform(data);
    BOOST_CHECK(cuv::equal_shape(data,data2));
    for(unsigned int i=0;i<data.shape(0);i++)
        for(unsigned int j=0;j<data.shape(1);j++){
            BOOST_CHECK_CLOSE((float)data(i,j), (float)data2(i,j),0.01f);
        }
}

BOOST_AUTO_TEST_CASE(reduced_pca){
    typedef cuv::tensor<float,cuv::host_memory_space> tens_t;

    cuvnet::pca_whitening pp(2,false); // 1 less than data dim

    tens_t data (1000, 3);

    // two basis vectors for dataset
    float p[] = { 2,-1, 1};
    float q[] = { 1, 3, 5};

    for (unsigned int i = 0; i < data.shape(0); ++i)
    {
        float fp = nrand();
        float fq = nrand(); // less variance in Q
        data(i,0) = fp*p[0] + fq*q[0] + 4;
        data(i,1) = fp*p[1] + fq*q[1] + 5;
        data(i,2) = fp*p[2] + fq*q[2] + 6;
    }
    tens_t data2 = data.copy();

    pp.fit_transform(data);
    BOOST_CHECK_EQUAL(data.shape(0),data2.shape(0));
    BOOST_CHECK_EQUAL(data.shape(1),2);

    // check for identity matrix does not make sense (reduced!)

    // check for reverse_transform
    pp.reverse_transform(data);
    BOOST_CHECK(cuv::equal_shape(data,data2));
    for(unsigned int i=0;i<data.shape(0);i++)
        for(unsigned int j=0;j<data.shape(1);j++){
            //std::cout << "i, j = "<<i<<", "<<j<<std::endl;
            BOOST_CHECK_CLOSE(1.f,  1.f + (float)data(i,j) - (float)data2(i,j),0.01f);
        }
}

BOOST_AUTO_TEST_CASE(reduced_whiten){
    typedef cuv::tensor<float,cuv::host_memory_space> tens_t;

    cuvnet::pca_whitening pp(2,true); // 1 less than data dim

    tens_t data (1000, 3);

    // two basis vectors for dataset
    float p[] = { 2,-1, 1};
    float q[] = { 1, 3, 5};

    for (unsigned int i = 0; i < data.shape(0); ++i)
    {
        float fp = nrand();
        float fq = nrand(); // less variance in Q
        data(i,0) = fp*p[0] + fq*q[0] + 4;
        data(i,1) = fp*p[1] + fq*q[1] + 5;
        data(i,2) = fp*p[2] + fq*q[2] + 6;
    }
    tens_t data2 = data.copy();

    pp.fit_transform(data);
    BOOST_CHECK_EQUAL(data.shape(0),data2.shape(0));
    BOOST_CHECK_EQUAL(data.shape(1),2);

    // check for identity matrix  not possible (not lossless)

    // check for reverse_transform
    pp.reverse_transform(data);
    BOOST_CHECK(cuv::equal_shape(data,data2));
    for(unsigned int i=0;i<data.shape(0);i++)
        for(unsigned int j=0;j<data.shape(1);j++){
            //std::cout << "i, j = "<<i<<", "<<j<<std::endl;
            BOOST_CHECK_CLOSE( 1.f,  1 + (float)data(i,j) - (float)data2(i,j),0.01f);
        }
}

BOOST_AUTO_TEST_CASE(normal_zca){
    typedef cuv::tensor<float,cuv::host_memory_space> tens_t;

    cuvnet::pca_whitening pp(-1,true,true); // 1 less than data dim

    tens_t data (1000, 3);

    // two basis vectors for dataset
    float p[] = { 2,-1, 1};
    float q[] = { 1, 3, 5};

    for (unsigned int i = 0; i < data.shape(0); ++i)
    {
        float fp = nrand();
        float fq = nrand(); // less variance in Q
        data(i,0) = fp*p[0] + fq*q[0] + 4;
        data(i,1) = fp*p[1] + fq*q[1] + 5;
        data(i,2) = fp*p[2] + fq*q[2] + 6;
    }
    tens_t data2 = data.copy();

    pp.fit_transform(data);
    BOOST_CHECK_EQUAL(data.shape(0),data2.shape(0));
    BOOST_CHECK_EQUAL(data.shape(1),data2.shape(1));

    // check for identity matrix
    const tens_t&  rot = pp.rot();
    const tens_t& rrot = pp.rrot();
    tens_t res(cuv::extents[rot.shape(0)][rot.shape(0)]);
    cuv::prod(res,rot,rrot);
    for(unsigned int i=0;i<rot.shape(0);i++){
        res(i,i)-=1.f;
    }
    BOOST_CHECK_CLOSE(1.f, 1.f + cuv::norm2(res), 0.01f);

    // check for reverse_transform
    pp.reverse_transform(data);
    BOOST_CHECK(cuv::equal_shape(data,data2));
    for(unsigned int i=0;i<data.shape(0);i++)
        for(unsigned int j=0;j<data.shape(1);j++){
            //std::cout << "i, j = "<<i<<", "<<j<<std::endl;
            BOOST_CHECK_CLOSE( 1.f, 1.f + (float)data(i,j) - (float)data2(i,j),0.01f);
        }
}
BOOST_AUTO_TEST_CASE(reduced_zca){
    typedef cuv::tensor<float,cuv::host_memory_space> tens_t;

    cuvnet::pca_whitening pp(2,true,true); // 1 less than data dim

    tens_t data (1000, 3);

    // two basis vectors for dataset
    float p[] = { 2,-1, 1};
    float q[] = { 1, 3, 5};

    for (unsigned int i = 0; i < data.shape(0); ++i)
    {
        float fp = nrand();
        float fq = nrand(); // less variance in Q
        data(i,0) = fp*p[0] + fq*q[0] + 4;
        data(i,1) = fp*p[1] + fq*q[1] + 5;
        data(i,2) = fp*p[2] + fq*q[2] + 6;
    }
    tens_t data2 = data.copy();

    pp.fit_transform(data);
    BOOST_CHECK_EQUAL(data.shape(0),data2.shape(0));
    BOOST_CHECK_EQUAL(data.shape(1),data2.shape(1));

    // check for identity matrix not possible: not lossless!

    // check for reverse_transform
    pp.reverse_transform(data);
    BOOST_CHECK(cuv::equal_shape(data,data2));
    for(unsigned int i=0;i<data.shape(0);i++)
        for(unsigned int j=0;j<data.shape(1);j++){
            //std::cout << "i, j = "<<i<<", "<<j<<std::endl;
            BOOST_CHECK_CLOSE(1.f, 1.f + (float)data(i,j) - (float)data2(i,j),0.01f);
        }
}
BOOST_AUTO_TEST_SUITE_END()

BOOST_AUTO_TEST_SUITE( pipeline )
BOOST_AUTO_TEST_CASE(simple){
    cuvnet::preprocessing_pipeline<> pp;
    pp.add(new cuvnet::global_min_max_normalize<>());
    pp.add(new cuvnet::zero_sample_mean<>());

    cuv::tensor<float,cuv::host_memory_space> v(cuv::extents[2][2]);
    v(0,0) = 0.f;
    v(0,1) = 200.f;
    v(1,0) = 0.f;
    v(1,1) = 400.f;
    pp.fit_transform(v);
    BOOST_CHECK_CLOSE((float)v(0,0), -0.25f, .01);
    BOOST_CHECK_CLOSE((float)v(0,1),  0.25f, .01);
    BOOST_CHECK_CLOSE((float)v(1,0), -0.5f, .01);
    BOOST_CHECK_CLOSE((float)v(1,1),  0.5f, .01);
}
BOOST_AUTO_TEST_SUITE_END()
