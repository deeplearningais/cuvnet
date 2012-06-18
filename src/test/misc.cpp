#include <vector>
#include <algorithm>
#include <gtest/gtest.h>

#include <tools/argsort.hpp>
#include <tools/preprocess.hpp>
#include <tools/orthonormalization.hpp>

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

TEST(argsort, normal){
    std::vector<float> v(50);
    std::generate(v.begin(),v.end(),drand48);
    std::vector<unsigned int> idx = argsort(v.begin(), v.end());
    EXPECT_EQ(50, idx.size());
    for(unsigned int i=1;i<50;i++){
        EXPECT_LT(v[idx[i-1]] , v[idx[i]]);
    }
}
TEST(argsort, inv){
    std::vector<float> v(50);
    std::generate(v.begin(),v.end(),drand48);
    std::vector<unsigned int> idx = argsort(v.begin(), v.end(), std::greater<float>());
    EXPECT_EQ(50, idx.size());
    for(unsigned int i=1;i<50;i++){
        EXPECT_GT(v[idx[i-1]] , v[idx[i]]);
    }
}
TEST(orthogonalization, everything){
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
                //EXPECT_NEAR(0.f, s, 0.001f);
                if(fabs(s) > 0.001f){
                    std::cout << "error: " << s << std::endl;
                    EXPECT_EQ(true,false);
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
            //EXPECT_NEAR(0.f, s, 0.001f);
            if(fabs(s) > 0.001f){
                std::cout << "error: " << s << std::endl;
                EXPECT_EQ(true,false);
            }
        }
    }
    
}

TEST(orthogonalization, pairs){
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
        //EXPECT_NEAR(0.f, s, 0.001f);
        if(fabs(s) > 0.001f){
            std::cout << "error: " << s << std::endl;
            EXPECT_EQ(true,false);
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
        //EXPECT_NEAR(0.f, s, 0.001f);
        if(fabs(s) > 0.001f){
            std::cout << "error: " << s << std::endl;
            EXPECT_EQ(true,false);
        }
    }
    
}

TEST(pca,normal){
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
    EXPECT_EQ(data.shape(0),data2.shape(0));
    EXPECT_EQ(data.shape(1),data2.shape(1));

    // check for identity matrix
    const tens_t&  rot = pp.rot();
    const tens_t& rrot = pp.rrot();
    tens_t res(cuv::extents[rot.shape(0)][rot.shape(0)]);
    cuv::prod(res,rot,rrot);
    for(unsigned int i=0;i<rot.shape(0);i++){
        res(i,i)-=1.f;
    }
    EXPECT_NEAR(0.f, cuv::norm2(res), 0.01f);

    // check for reverse_transform
    pp.reverse_transform(data);
    EXPECT_TRUE(cuv::equal_shape(data,data2));
    for(unsigned int i=0;i<data.shape(0);i++)
        for(unsigned int j=0;j<data.shape(1);j++){
            EXPECT_NEAR(data(i,j), data2(i,j),0.01f);
        }
}

TEST(pca,reduced_pca){
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
    EXPECT_EQ(data.shape(0),data2.shape(0));
    EXPECT_EQ(data.shape(1),2);

    // check for identity matrix does not make sense (reduced!)

    // check for reverse_transform
    pp.reverse_transform(data);
    EXPECT_TRUE(cuv::equal_shape(data,data2));
    for(unsigned int i=0;i<data.shape(0);i++)
        for(unsigned int j=0;j<data.shape(1);j++){
            //std::cout << "i, j = "<<i<<", "<<j<<std::endl;
            EXPECT_NEAR(data(i,j), data2(i,j),0.01f);
        }
}

TEST(pca,reduced_whiten){
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
    EXPECT_EQ(data.shape(0),data2.shape(0));
    EXPECT_EQ(data.shape(1),2);

    // check for identity matrix  not possible (not lossless)

    // check for reverse_transform
    pp.reverse_transform(data);
    EXPECT_TRUE(cuv::equal_shape(data,data2));
    for(unsigned int i=0;i<data.shape(0);i++)
        for(unsigned int j=0;j<data.shape(1);j++){
            //std::cout << "i, j = "<<i<<", "<<j<<std::endl;
            EXPECT_NEAR(data(i,j), data2(i,j),0.01f);
        }
}

TEST(pca,normal_zca){
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
    EXPECT_EQ(data.shape(0),data2.shape(0));
    EXPECT_EQ(data.shape(1),data2.shape(1));

    // check for identity matrix
    const tens_t&  rot = pp.rot();
    const tens_t& rrot = pp.rrot();
    tens_t res(cuv::extents[rot.shape(0)][rot.shape(0)]);
    cuv::prod(res,rot,rrot);
    for(unsigned int i=0;i<rot.shape(0);i++){
        res(i,i)-=1.f;
    }
    EXPECT_NEAR(0.f, cuv::norm2(res), 0.01f);

    // check for reverse_transform
    pp.reverse_transform(data);
    EXPECT_TRUE(cuv::equal_shape(data,data2));
    for(unsigned int i=0;i<data.shape(0);i++)
        for(unsigned int j=0;j<data.shape(1);j++){
            //std::cout << "i, j = "<<i<<", "<<j<<std::endl;
            EXPECT_NEAR(data(i,j), data2(i,j),0.01f);
        }
}
TEST(pca,reduced_zca){
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
    EXPECT_EQ(data.shape(0),data2.shape(0));
    EXPECT_EQ(data.shape(1),data2.shape(1));

    // check for identity matrix not possible: not lossless!

    // check for reverse_transform
    pp.reverse_transform(data);
    EXPECT_TRUE(cuv::equal_shape(data,data2));
    for(unsigned int i=0;i<data.shape(0);i++)
        for(unsigned int j=0;j<data.shape(1);j++){
            //std::cout << "i, j = "<<i<<", "<<j<<std::endl;
            EXPECT_NEAR(data(i,j), data2(i,j),0.01f);
        }
}

TEST(pipeline, simple){
    cuvnet::preprocessing_pipeline<> pp;
    pp.add(new cuvnet::global_min_max_normalize<>());
    pp.add(new cuvnet::zero_sample_mean<>());

    cuv::tensor<float,cuv::host_memory_space> v(cuv::extents[2][2]);
    v(0,0) = 0.f;
    v(0,1) = 200.f;
    v(1,0) = 0.f;
    v(1,1) = 400.f;
    pp.fit_transform(v);
    EXPECT_NEAR(v(0,0), -0.25f, .01);
    EXPECT_NEAR(v(0,1),  0.25f, .01);
    EXPECT_NEAR(v(1,0), -0.5f, .01);
    EXPECT_NEAR(v(1,1),  0.5f, .01);
}
