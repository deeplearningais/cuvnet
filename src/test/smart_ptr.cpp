#include <cmath>
#include <stdexcept>
#include <cstdio>

#include <cuvnet/op.hpp>

#include <boost/test/unit_test.hpp>

using namespace cuvnet;

BOOST_AUTO_TEST_SUITE(smart_ptr)

BOOST_AUTO_TEST_CASE(t_cow_ptr){
	cow_ptr<int> k;
	BOOST_REQUIRE(!k);

	cow_ptr<int> i(new int(4));
	BOOST_REQUIRE_EQUAL(i.cdata(),4);

	// copying ptr should result in copy of data
	cow_ptr<int> j = i;
	BOOST_REQUIRE_EQUAL(&i.cdata(), &j.cdata());

	// writing to ptr should result in copy of data
	*i = 3;
	BOOST_REQUIRE_NE(&i.cdata(), &j.cdata());
	BOOST_REQUIRE_EQUAL(i.cdata(), 3);

	// changing the only ptr to a value should not result in reallocation
	const int* old = &i.cdata();
	*i = 4;
	BOOST_REQUIRE_EQUAL(&i.cdata(), old);
}

BOOST_AUTO_TEST_CASE(t_cow_ptr_unload){
    using namespace cuv;

    typedef tensor<float, host_memory_space> host_t;
    typedef tensor<float, dev_memory_space> dev_t;
    typedef cow_ptr<host_t> host_pt;
    typedef cow_ptr<dev_t> dev_pt;

    {
        dev_pt p = make_cow_ptr<dev_t>(extents[3][4]);
        p.unload_from_dev();
        BOOST_REQUIRE_EQUAL(p.ptr(), (dev_t*)NULL);

        p.ensure_on_dev();
        BOOST_REQUIRE_EQUAL(p->ndim(), 2);
        BOOST_REQUIRE_EQUAL(p->shape(0), 3);
        BOOST_REQUIRE_EQUAL(p->shape(1), 4);
    }

    {
        // unloading host tensor should NOT work
        host_pt p = make_cow_ptr<host_t>(extents[3][4]);
        p.unload_from_dev();
        BOOST_REQUIRE_EQUAL(p->ndim(), 2);
        BOOST_REQUIRE_EQUAL(p->shape(0), 3);
        BOOST_REQUIRE_EQUAL(p->shape(1), 4);
    }

}

BOOST_AUTO_TEST_SUITE_END()
