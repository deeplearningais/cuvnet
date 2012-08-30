#ifndef __NETWORK_COMMUNICATION_HPP__
#     define __NETWORK_COMMUNICATION_HPP__
#include <cuvnet/common.hpp>

namespace cuvnet
{
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


    }
}
#endif /* __NETWORK_COMMUNICATION_HPP__ */
