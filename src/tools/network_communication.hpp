#ifndef __NETWORK_COMMUNICATION_HPP__
#     define __NETWORK_COMMUNICATION_HPP__
#include <cuvnet/common.hpp>

namespace cuvnet
{
    class Op;

    /**
     * Communicate between processes working on the same problem via MongoDB.
     */
    namespace network_communication
    {
        typedef cuv::tensor<float, cuv::host_memory_space> htensor_t;
        struct value_not_found_exception{};

        struct connection;

        // TODO:
        // - split up datasets (not necessary if randomized?)
        // - callbacks for gradient descent
        //   - initialize networks with fixed seed
        //   - when acquiring merged params: MITTELN mit aktuellen (sonst tlw umsonst)
        // - only pull when somebody ELSE has pushed... or only merge if 2 have pushed?
        //   nice group-by query in merge to find such things?
        //   still, need to limit merges! e.g. versioning?
        // - disable in validation epochs
        // - update loop for server

        /**
         * Does tasks which cannot be done by clients.
         * The main use currently is to apply weight updates to centrally
         * managed weights.
         */
        class server{
            private:
                boost::shared_ptr<connection> m_impl;      ///< PImpl idiom: DB connection is stored in here.
                std::map<std::string, htensor_t> m_merged; ///< contains merged weight vectors
                std::map<std::string, bool>      m_need_push; ///< the marked weights need pushing to MongoDB
                std::map<std::string, int>       m_versions; ///< a version increased when changed
            public:
                /**
                 * ctor.
                 *
                 * @param url ip of mongodb instance
                 * @param prefix database in mongodb
                 * @param key identifies instances working on the same tasks uniquely.
                 */
                server(const std::string& url, const std::string& prefix, const std::string key="");

                /**
                 * pull information from server (eg after a crash).
                 */
                void pull_merged();

                /**
                 * push data to server if needed.
                 */
                void push_merged();

                /**
                 * merge weight updates in the database into the stored weights.
                 */
                void merge();

                /**
                 * delete all data of the current key.
                 */
                void cleanup();

                /**
                 * run the server in a loop.
                 * @param sleep_msec polling interval (milliseconds)
                 * @param n how often to poll in total (default: forever)
                 */
                void run(unsigned int sleep_msec, int n=-1);
        };

        /**
         * Provides a special-purpose interface for communication between learners.
         *
         * For now this is mainly means storing and retreiving weight updates.
         * It should however also coordinate when to stop learning.
         */
        class client{
            private:
                boost::shared_ptr<connection> m_impl; ///< PImpl idiom.
                std::string m_id; ///< a unique id identifying this client
                std::map<std::string, int> m_versions; ///< the newest version pulled (to avoid pulling twice)
            public:
                /**
                 * ctor.
                 *
                 * @param url ip of mongodb instance
                 * @param prefix database in mongodb
                 * @param key identifies instances working on the same tasks uniquely.
                 */
                client(const std::string& url, const std::string& prefix, const std::string key="", const std::string id="");
                /**
                 * fetch a merged weight vector from the database.
                 * @throw value_not_found_exception if it cannot be found
                 * @param s a name, which should uniquely identify the parameter
                 */
                htensor_t fetch_merged(const std::string& s);
                /**
                 * send a weight update for merging to the database.
                 * @param s a name, which should uniquely identify the parameter
                 * @param delta the change in parameters
                 * @param current_value if there is no `current' value in the database, use this one
                 */
                void put_for_merging(const std::string& s, const htensor_t& delta, const matrix& current_value);
        };

        class param_synchronizer{
            private:
                int m_push_steps;
                int m_pull_steps;
                int m_cnt;
                std::vector<Op*> m_ops;
                client& m_client;
            public:
                typedef void result_type;
                param_synchronizer(client& clt, int push_steps, int pull_steps, const std::vector<Op*>& ops)
                    : m_push_steps(push_steps)
                      , m_pull_steps(pull_steps)
                      , m_cnt(0)
                      , m_ops(ops)
                      , m_client(clt){}

                void operator()(
                        std::map<Op*, cuv::tensor<float, cuv::host_memory_space> >* updates
                        ,unsigned int, unsigned int
                        );
        };


    }
}
#endif /* __NETWORK_COMMUNICATION_HPP__ */
