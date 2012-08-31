#ifndef __NETWORK_COMMUNICATION_HPP__
#     define __NETWORK_COMMUNICATION_HPP__
#include <cuvnet/common.hpp>

namespace cuvnet
{
    class Op;

    namespace network_communication
    {
        typedef cuv::tensor<float, cuv::host_memory_space> htensor_t;
        struct value_not_found_exception{};

        struct connection;

        class server{
            private:
                boost::shared_ptr<connection> m_impl;
                std::map<std::string, htensor_t> m_merged;
                std::map<std::string, bool>      m_need_push;
            public:
                server(const std::string& url, const std::string& prefix, const std::string key="");
                void pull_merged();
                void push_merged();
                void merge();
                void cleanup();
                void run(unsigned int sleep_sec, int n=-1);
        };

        class client{
            private:
                boost::shared_ptr<connection> m_impl;
                std::string m_id;
            public:
                client(const std::string& url, const std::string& prefix, const std::string key="", const std::string id="");
                htensor_t fetch_merged(const std::string& s);
                void put_for_merging(const std::string& s, htensor_t& m);
        };

        class param_synchronizer{
            private:
                int m_push_steps;
                int m_pull_steps;
                int m_cnt;
                std::vector<Op*> m_ops;
                client& m_client;
            public:
                param_synchronizer(client& clt, int push_steps, int pull_steps, const std::vector<Op*>& ops)
                    : m_push_steps(push_steps)
                      , m_pull_steps(pull_steps)
                      , m_cnt(0)
                      , m_ops(ops)
                      , m_client(clt){}

                void operator()();
        };


    }
}
#endif /* __NETWORK_COMMUNICATION_HPP__ */
