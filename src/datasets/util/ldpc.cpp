// low density parity check code example

// for every constraint, generate 
// TODO: 4 
// output variables, with weights resembling the different cases 
// where the sum of the inputs mod 2 is 0

#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/vector.hpp>
#include <boost/numeric/ublas/io.hpp>
#include <boost/numeric/ublas/lu.hpp>
#include <boost/format.hpp>
#include <algorithm>
#include <ctime>
#include <cstdio>
#include <fstream>

#define V(X) #X<<":"<<(X)<<", "
using namespace boost::numeric::ublas;
typedef permutation_matrix<std::size_t> pmatrix;

template<class PM, class MV>
BOOST_UBLAS_INLINE
void swap_cols (const PM &pm, MV &mv) {
    typedef typename PM::size_type size_type;
    typedef typename MV::value_type value_type;

    size_type size = pm.size ();
    for (size_type i = 0; i < size; ++ i) {
        if (i != pm (i))
            column (mv, i).swap (column (mv, pm (i)));
    }
}

template<class V>
V vec_conc(const V& a, const V& b){
    V c(a.size()+b.size());
    vector_range<V>(c, range(0,a.size())) = a;
    vector_range<V>(c, range(a.size(),c.size())) = b;
    return c;
}


#define N_DENSITY 3
#define N_EVEN    4

float binarize(float f){
    return f>0.5f;
}


template<class T>
void printmat(std::string name, const T& m){
    std::string filename = boost::str(boost::format("%s_%dx%d_float32.bin")%name%m.size1()%m.size2());
    std::ofstream of(filename.c_str());
    of.write((char*)(&m(0,0)), sizeof(float)*m.size1()*m.size2());
    std::cout <<name<<": "<< filename << std::endl;
    return;


    for (unsigned int i = 0; i < m.size1(); ++i)
    {
        for (unsigned int j = 0; j < m.size2(); ++j)
        {
            printf("% 3.1f ", (float)m(i,j));
        }
        printf("\n");
    }
    printf("\n");
}

template<class T>
bool eq_row(const T& m, int i, int j){
    bool same = true;
    for(unsigned int k=0; k<m.size2(); k++){
        if(m(i, k) != m(j, k)) {
            same = false;
            break;
        }
    }
    return same;
}
template<class T>
bool eq_col(const T& m, int i, int j){
    bool same = true;
    for(unsigned int k=0; k<m.size1(); k++){
        if(m(k, i) != m(k, j)) {
            same = false;
            break;
        }
    }
    return same;
}


matrix<float> get_constraint_matrix_old(unsigned int n_variables, unsigned int density){
    unsigned int n_constraints = n_variables;
    matrix<float> H(n_variables,n_constraints);
    H = zero_matrix<float>(n_variables,n_constraints);
    int res;
    do{
        for(unsigned int i=0;i<n_variables;i++){
            // every bit in message
            H(i,i) = 1;
            for (unsigned int j = 1; j < density; ++j){
                // every constraint it participates in
                while(true){
                    unsigned int idx = drand48()*n_constraints;
                    if(H(i,idx))
                        continue;
                    H(i,idx) = 1.f;
                    break;
                }
            }
        }

        // check rank
        if(H.size1()>H.size2()){
            matrix<float> A(H.size1(),H.size1());
            noalias(A) = prod(H,trans(H));
            pmatrix pm(A.size1());
            res = lu_factorize(A,pm);
        }else{
            matrix<float> A(H.size2(),H.size2());
            noalias(A) = prod(trans(H),H);
            pmatrix pm(A.size1());
            res = lu_factorize(A,pm);
        }
        std::cout<<"res="<<res<<std::endl;
    }while(res!=0);

    return H;
}

matrix<float> get_constraint_matrix(unsigned int n_variables, unsigned int density){
    matrix<float> H(n_variables,n_variables);
    for(unsigned int v=0; v< n_variables; v++){
        while(true){
            for(unsigned int c=0; c<n_variables; c++)
                H(v, c) = 0;

            for(unsigned int c=0; c<density;){
                int pos = (int)(drand48() * n_variables);
                if(H(v,pos) > 0.5)
                    continue; // already occupied

                if(sum(column(H, pos)) <= density){
                    H(v, pos) = 1;
                    c++;
                }
                // too many in this row already, continue
            }
            //if(sum(row(H, v)) != density)
                //continue;
            bool stop = false;
            for(unsigned int v2=0; v2< n_variables; v2++){
                if(eq_row(H, v, v2)){
                    stop = true;
                    break;
                }
            }
            if(stop == true)
                break;
        }
    }
    return H;
}
matrix<float> get_constraint_matrix_OLDANDBROKEN(unsigned int n_variables, unsigned int density){
    matrix<float> H(n_variables,n_variables);
    assert(n_variables % density == 0);
    for (unsigned int i = 0; i < n_variables/density; ++i)
        for (unsigned int  j = i*density; j < i*density+density; ++j)
            for (unsigned int  k = i*density; k < i*density+density; ++k)
                H(k, j) = 1;
    printmat("H0",H);
    int res=0;
    int maxres = res;
    int cnt=0;
    do{
        do{
            pmatrix pm(n_variables);
            std::random_shuffle(pm.begin(),pm.end());
            swap_rows(pm,H);
            std::random_shuffle(pm.begin(),pm.end());
            swap_cols(pm,H);
        }while(cnt++<10); // start by shuffling 10 times

        // check rank
        matrix<float> A(H.size1(),H.size1());
        noalias(A) = prod(H,trans(H));
        pmatrix pm2(n_variables);
        res = lu_factorize(A,pm2);
        std::cout<<" maxres="<<maxres<<std::flush;
        maxres = std::max(maxres,res);
    }while(res!=n_variables/density);
    return H;
}

float is_even(float f){
    return ((int) (f+0.5)) % 2 == 0;
}

float threshold_and(float f){
    return f>(N_DENSITY-1.5);
}
float threshold_or(float f){
    return f>0.5;
}

int
main(int argc, char **argv)
{
    unsigned int n_variables   = N_DENSITY*5;
    unsigned int n_constraints = n_variables;
    //srand48(time(NULL));
    //srand(time(NULL));
    srand48(42);
    srand(42);

    vector<float> variables(n_variables);
    std::generate(variables.begin(),variables.end(),drand48);
    std::transform(variables.begin(),variables.end(),variables.begin(),binarize);

    matrix<float> H = get_constraint_matrix(n_variables,N_DENSITY);
    vector<float> enc = prod((H) , variables);
    
    std::cout <<"original: "<<variables<<std::endl;
    std::cout <<"encoded : "<<enc<<std::endl;
    vector<float> check(enc.size());
    std::transform(enc.begin(),enc.end(),check.begin(),is_even);
    std::cout <<"check   : "<<check<<std::endl;  // this is what we want the NN to produce. Now design a matrix that does the same.
    
    // for 3 variables participating in 1 constraint, there are 4 cases where their sum is even.
    matrix<float> wand(n_variables, n_constraints * N_EVEN);
    matrix<float> wor(n_constraints*N_EVEN, n_constraints);
    wand = zero_matrix<float>(wand.size1(),wand.size2());
    wor = zero_matrix<float>(wor.size1(),wor.size2());
    for(int c=0;c<n_constraints;c++)
    {
        int cnt = 0;
        for (unsigned int v = 0; v < n_variables; ++v){
            if( H(c,v) == 0.f )
                continue;
            wor(c*N_EVEN+0,c) = 1;
            wor(c*N_EVEN+1,c) = 1;
            wor(c*N_EVEN+2,c) = 1;
            wor(c*N_EVEN+3,c) = 1;
            switch(cnt){
                case 0:
                    wand(v,c*N_EVEN+0) = 1;
                    wand(v,c*N_EVEN+1) = 1;
                    wand(v,c*N_EVEN+2) =-1;
                    wand(v,c*N_EVEN+3) =-1;
                    break;           
                case 1:              
                    wand(v,c*N_EVEN+0) = 1;
                    wand(v,c*N_EVEN+1) =-1;
                    wand(v,c*N_EVEN+2) = 1;
                    wand(v,c*N_EVEN+3) =-1;
                    break;           
                case 2:              
                    wand(v,c*N_EVEN+0) =-1;
                    wand(v,c*N_EVEN+1) = 1;
                    wand(v,c*N_EVEN+2) = 1;
                    wand(v,c*N_EVEN+3) =-1;
                    break;
                default:
                    assert(false);
            }
            cnt++;
        }
    }
    printmat("H",H);
    printmat("wand",wand);
    vector<float> intermed = prod(trans(wand),variables);
    for (unsigned int i = 0; i < intermed.size(); i+=N_EVEN)
    {
        // last constraint needs to sum up to 0
        // if it fails, it is <0
        intermed[i+N_EVEN-1] += N_DENSITY-1; // [-N_DENSITY,0] --> [0,N_DENSITY]
    }
    std::cout << "intermed 0: "<<intermed<<std::endl;
    std::transform(intermed.begin(), intermed.end(), intermed.begin(),threshold_and);
    std::cout << "intermed 1: "<<intermed<<std::endl;

    vector<float> check2 = prod(trans(wor), intermed);
    std::cout << "check2 0: "<<check2<<std::endl;
    std::transform(check2.begin(), check2.end(), check2.begin(),threshold_or);
    std::cout << "check2 1: "<<check2<<", "<<sum(check2)<<std::endl;
    std::cout << "check1 1: "<<check <<", "<<sum(check)<<std::endl;

    intermed  = prod(wor,check2);
    intermed += prod(trans(wand),variables);
    for (unsigned int i = 0; i < intermed.size(); i+=N_EVEN)
        intermed[i+N_EVEN-1] += N_DENSITY-1; // [-N_DENSITY,0] --> [0,N_DENSITY]
    std::transform(intermed.begin(), intermed.end(), intermed.begin(),threshold_and);
    std::cout << "intermed'0: "<<intermed<<std::endl;
    std::cout << "orig      : "<<variables<<std::endl;
    variables += prod(wand,intermed);
    std::transform(variables.begin(), variables.end(), variables.begin(),threshold_or);
    std::cout << "orig'    0: "<<variables<<std::endl;

    printmat("wor",wor);

    // create dataset
    const unsigned int dataset_size = 32768;
    //const unsigned int dataset_size = 60000;
    matrix<float> dataset(dataset_size, n_variables*2); // code + data
    matrix<float> cov = outer_prod(check2,check);
    vector<unsigned int> allvar(pow(2,n_variables));
    for(unsigned int i=0;i<allvar.size();i++)
        allvar[i] = i;
    random_shuffle(allvar.begin(),allvar.end());
    if(allvar.size()<dataset_size){
        std::cerr <<"too large datset size!"<<std::endl;
        exit(1);
    }

    for (int datum = 0; datum < dataset_size; ++datum)
    {
        unsigned int val = allvar[datum];
        for(unsigned int k=0;k<n_variables;k++){
            variables[k] = (val & (0x1 << k)) ? 1 : 0;
        }
        //std::cout << variables<<std::endl;
        //std::generate(variables.begin(),variables.end(),drand48);
        //std::transform(variables.begin(),variables.end(),variables.begin(),binarize);
        enc = prod((H) , variables);
        std::transform(enc.begin(),enc.end(),check.begin(),is_even);

        vector<float> intermed = prod(trans(wand),variables);
        for (unsigned int i = 0; i < intermed.size(); i+=N_EVEN)
            intermed[i+N_EVEN-1] += N_DENSITY-1; // [-N_DENSITY,0] --> [0,N_DENSITY]
        std::transform(intermed.begin(), intermed.end(), intermed.begin(),threshold_and);
        check2 = prod(trans(wor), intermed);
        std::transform(check2.begin(), check2.end(), check2.begin(),threshold_or);
        cov += outer_prod(check2,check);

        //std::cout << "check2 1: "<<check2<<", "<<sum(check2)<<std::endl;
        //std::cout << "check1 1: "<<check <<", "<<sum(check)<<std::endl;
        //std::cout << std::endl;
        assert(norm_2(check2-check) < 0.001);
        row(dataset,datum) = vec_conc(check, variables);

    }

    printmat("ds", dataset);
    
    float mm = -1;
    for(unsigned int i=0;i<cov.size1();i++)
        for(unsigned int j=0;j<cov.size2();j++)
            mm = std::max(cov(i,j),mm);

    cov = cov / mm;

    for(unsigned int i=0;i<cov.size1();i++)
        for(unsigned int j=0;j<cov.size2();j++){
            cov(i,j) -= 0.5f;
            cov(i,j)  = std::max(0.f,cov(i,j));
        }
    cov = cov * 2.f;


    printmat("cov", cov);
    
    return 0;
}
