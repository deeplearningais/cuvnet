#include <stdexcept>
#include <gtest/gtest.h>
#include <glog/logging.h>

#include <cuvnet/op.hpp>

using namespace cuvnet;

TEST(ce_ptr_test, init){
	ce_ptr<int> k;
	assert(!k);

	ce_ptr<int> i = ce_ptr<int>(new int(4));
	ce_ptr<int> j = i;
	
	EXPECT_TRUE(i);
	EXPECT_TRUE(j);
	EXPECT_EQ(i.ptr() , j.ptr());

	void* p = (void*) 4;
	void* q = (void*) 5;
	i.lock(p); // p says he wants to keep the value of i

	EXPECT_TRUE(i.writable(p));
	EXPECT_TRUE(j.writable(p));

	EXPECT_TRUE(!i.writable(q));
	EXPECT_TRUE(!j.writable(q));

	EXPECT_TRUE(i.locked_by(p));
	EXPECT_TRUE(j.locked_by(p));
	EXPECT_TRUE(!i.locked_by(q));
	EXPECT_TRUE(!j.locked_by(q));

	std::cout<<i << " "<< i.ptr()<<std::endl;
	std::cout<<j << " "<< j.ptr()<<std::endl;

	j.lock(q);     // q says he wants to keep the value of j

	EXPECT_TRUE(i.locked_by(q)); // both are locked by q now
	EXPECT_TRUE(j.locked_by(q)); // both are locked by q now

	EXPECT_TRUE(i.locked_by(p)); // both are locked by p still
	EXPECT_TRUE(j.locked_by(p)); // both are locked by p still

	EXPECT_TRUE(!j.writable(p)); // noone can write w/o copying
	EXPECT_TRUE(!j.writable(q)); // noone can write w/o copying

	j.data(q) = 5;          // q wants to change the value of j.
	                        // --> j gets detached
				// --> i is not locked by q anymore
				// --> j is not locked by p anymore
	j.reset(new int(6));
	j.lock(q);

	EXPECT_TRUE(!i.locked_by(q)); 
	EXPECT_TRUE(!j.locked_by(p));
	EXPECT_TRUE(i.locked_by(p)); 
	EXPECT_TRUE(j.locked_by(q));

	EXPECT_TRUE(i.writable(p));
	EXPECT_TRUE(j.writable(q));

	EXPECT_TRUE(!i.writable(q));
	EXPECT_TRUE(!j.writable(p));

	EXPECT_NE(i.ptr() , j.ptr());
	EXPECT_EQ(*i , 4);
	EXPECT_EQ(*j , 6);

	std::cout<<i << " "<< i.ptr()<<std::endl;
	std::cout<<j << " "<< j.ptr()<<std::endl;

}

TEST(op_test,misc){
	typedef boost::shared_ptr<Op> ptr_t;
    ptr_t id = boost::make_shared<Identity>();
    Op::param_collector pc;
    id->visit(pc);
    EXPECT_EQ(pc.plist.size(),0);
}
