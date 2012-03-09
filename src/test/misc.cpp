#include <vector>
#include <algorithm>
#include <gtest/gtest.h>

#include <tools/argsort.hpp>
#include <tools/preprocess.hpp>

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
