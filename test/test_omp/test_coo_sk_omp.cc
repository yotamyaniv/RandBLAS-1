#include "../comparison.hh"
#include <gtest/gtest.h>
#include "../test_matmul_cores/linop_common.hh"



using std::vector;
using RandBLAS::OMP;
using RandBLAS::RNGState;
using RandBLAS::SignedInteger;
using RandBLAS::SparseDist;
using RandBLAS::SparseSkOp;
using RandBLAS::Axis;
using RandBLAS::fill_sparse;
using RandBLAS::fill_sparse_unpacked_nosub;
using test::linop_common::random_matrix;
using RandBLAS::DenseDist;
using RandBLAS::sparse_data::coo::apply_coo_left_jki_p11;

// maybe need these:

//using namespace RandBLAS::sparse_data;



class TestCOOSKOMP : public ::testing::Test {
    
    protected:
        std::vector<int64_t> vec_nnzs{(int64_t) 1, (int64_t) 2, (int64_t) 3, (int64_t) 7};    
        std::vector<uint32_t> keys{42, 0, 1};

    virtual void SetUp(){};

    virtual void TearDown(){};

    template <typename T, SignedInteger sint_t>
    void test_coo_sk_times_dense_A(int64_t d, int64_t m, int64_t n, int64_t key_index, int64_t nnz_index, OMP dev){
        // C =  S * A
        //          coo * dense      
        // S generation:

        std::vector<T> C(d * n, 0.0);
        std::vector<T> C_ref(d * n, 0.0);

        using RNG = SparseSkOp<float>::state_t::generator;
        SparseDist D0(d, m, vec_nnzs[nnz_index], Axis::Short);
        SparseSkOp<float, RNG, sint_t> S0(D0, keys[key_index]);
        fill_sparse(S0);
        // convert to COO distribution:
        auto coo_sk = RandBLAS::sparse::coo_view_of_skop(S0);

        std::vector<T> dense_sk(d * m);

        RandBLAS::sparse_data::coo::coo_to_dense(coo_sk, blas::Layout::ColMajor, dense_sk.data());

        // make a dense A matrix of correct shape
        auto A = std::get<0>(random_matrix<T>(m,n, RNGState(1)));
       
        // apply_coo_left_jki_p11 
        apply_coo_left_jki_p11<T,sint_t>(
            1.0,blas::Layout::ColMajor,blas::Layout::ColMajor,d,n, m,
            coo_sk, 0,0,A.data(),m,C.data(),d,dev);

        // do the gemm 
        blas::gemm(
            blas::Layout::ColMajor, blas::Op::NoTrans, blas::Op::NoTrans,        
            d, n, m, 1.0, dense_sk.data(), d, A.data(),  m, 0.0, C_ref.data(),d);
    
        // check approx equality
        test::comparison::buffs_approx_equal(C.data(), C_ref.data(), d * n,
        __PRETTY_FUNCTION__, __FILE__, __LINE__);
    }

};

TEST_F(TestCOOSKOMP, test_coo_sk_times_dense_A) {
    //no offload
    test_coo_sk_times_dense_A<float,int64_t>(50,100,200,0,1, OMP::Host);
    test_coo_sk_times_dense_A<float,int64_t>(50,100,200,1,1, OMP::Host);
    test_coo_sk_times_dense_A<float,int64_t>(50,100,200,2,1, OMP::Host);
    test_coo_sk_times_dense_A<float,int64_t>(50,100,200,2,1, OMP::Host);
    //yes offload
    test_coo_sk_times_dense_A<float,int64_t>(50,100,200,0,1, OMP::Device);
    test_coo_sk_times_dense_A<float,int64_t>(50,100,200,1,1, OMP::Device);
    test_coo_sk_times_dense_A<float,int64_t>(50,100,200,2,1, OMP::Device);
    test_coo_sk_times_dense_A<float,int64_t>(50,100,200,2,1, OMP::Device);
    
    
}


