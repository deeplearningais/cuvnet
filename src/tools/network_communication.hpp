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


        /**
         * a merger merges parameter objects and their deltas.
         * The simplest form simply adds all deltas to the parameter objs.
         * A simple addon could be a momentum term.
         */
        struct merger{
            std::map<std::string, htensor_t> m_merged; ///< contains merged weight vectors
            /**
             * add a previously unknown parameter to the list
             */
            virtual void add_param(const std::string& name, const htensor_t& t);
            /**
             * merge a parameter update into its parameter object.
             */
            virtual void merge(const std::string& name, const htensor_t& delta);
            /**
             * @return whether the named parameter is known
             */
            bool has(const std::string& name);
            /**
             * @return the number of tracked parameters
             */
            inline size_t n_params(){return m_merged.size();}
            /**
             * @return the named parameter 
             */
            htensor_t& operator[](const std::string& name);
        };

        /**
         * merge with momentum.
         */
        struct momentum_merger : public merger{
            float m_momentum;
            std::map<std::string, htensor_t> m_moments; ///< contains the per-variable momentum
            /**
             * ctor.
             * @param momentum the momentum to be used in weight updates
             */
            momentum_merger(float momentum);
            /**
             * merge a parameter update into its parameter object.
             */
            virtual void merge(const std::string& name, const htensor_t& delta);
            /**
             * add a previously unknown parameter to the list
             */
            virtual void add_param(const std::string& name, const htensor_t& t);
        };

        /**
         * merge with adagrad.
         */
        struct adagrad_merger : public merger{
            float m_expwin;
            int m_resetcnt;
            std::map<std::string, unsigned int> m_count;
            std::map<std::string, htensor_t> m_moments; ///< contains the per-variable momentum
            //std::map<std::string, std::list<htensor_t> > m_queue; ///< contains the per-variable momentum
            /**
             * ctor.
             * @param expwin integration constant of window in [0,1]. Larger means more adaptive.
             * @param resetcnt reset squared gradient sum every this many updates
             */
            adagrad_merger(float expwin=0.1, int resetcnt=500);
            /**
             * merge a parameter update into its parameter object.
             */
            virtual void merge(const std::string& name, const htensor_t& delta);
            /**
             * add a previously unknown parameter to the list
             */
            virtual void add_param(const std::string& name, const htensor_t& t);
        };


        // TODO:
        // - split up datasets (not necessary if randomized?)
        // - initialize networks with fixed seed
        // - disable in validation epochs
        // - add "stage" parameter to weights?

        /**
         * Does tasks which cannot be done by clients.
         * The main use currently is to apply weight updates to centrally
         * managed weights.
         */
        class server{
            private:
                boost::shared_ptr<connection> m_impl;      ///< PImpl idiom: DB connection is stored in here.
                //std::map<std::string, htensor_t> m_merged; ///< contains merged weight vectors
                std::map<std::string, bool>      m_need_push; ///< the marked weights need pushing to MongoDB
                std::map<std::string, int>       m_versions; ///< a version increased when changed
                merger*                          m_merger;  ///< merges deltas into params
            public:
                /**
                 * ctor.
                 *
                 * @param url ip of mongodb instance
                 * @param prefix database in mongodb
                 * @param key identifies instances working on the same tasks uniquely.
                 * @param m merger. If Null, set to plain merger
                 */
                server( const std::string& url, const std::string& prefix, const std::string key="", merger* m=NULL);

                inline void set_merger(merger* m){m_merger = m;}

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
                std::map<std::string, int> m_delta_versions; ///< the newest version sent (for sorting merges)
            public:
                /**
                 * ctor.
                 *
                 * @param url ip of mongodb instance
                 * @param prefix database in mongodb
                 * @param key identifies instances working on the same tasks uniquely.
                 * @param id identifies the client uniquely
                 */
                client(const std::string& url, const std::string& prefix, const std::string key="", const std::string id="");
                /**
                 * @return the unique client id
                 */
                inline const std::string& id()const{return m_id;}
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

                /**
                 * send a 'stop' signal to coworkers.
                 *
                 * @param stage stop clients in this stage
                 */
                void send_stop_signal(const std::string& stage);

                /**
                 * determine if a co-worker requested to stop learning
                 *
                 * @param stage only check for stop-signals of the named stage
                 */
                bool got_stop_signal(const std::string& stage);
        };

        class param_synchronizer{
            private:
                int m_push_steps;
                int m_pull_steps;
                int m_push_off;
                int m_last_push;
                int m_pull_off;
                int m_cnt;
                std::vector<Op*> m_ops;
                client& m_client;
                std::string m_stage;
            public:
                typedef void result_type;
                param_synchronizer(const std::string& stage, client& clt, 
                        int push_steps, int pull_steps, 
                        int push_off, int pull_off, 
                        const std::vector<Op*>& ops)
                    : 
                        m_push_steps(push_steps)
                      , m_pull_steps(pull_steps)
                      , m_push_off(push_off)
                      , m_last_push(0)
                      , m_pull_off(pull_off)
                      , m_cnt(0)
                      , m_ops(ops)
                      , m_client(clt)
                      , m_stage(stage)
                {
                }

                void stop_coworkers();
                void test_stop();

                void operator()(
                        std::map<Op*, cuv::tensor<float, cuv::host_memory_space> >* updates
                        ,unsigned int, unsigned int
                        );
        };


    }
}
#endif /* __NETWORK_COMMUNICATION_HPP__ */
