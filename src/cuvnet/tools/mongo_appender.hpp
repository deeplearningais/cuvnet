#include <mdbq/client.hpp>

#include <log4cxx/appenderskeleton.h>
#include <log4cxx/spi/loggingevent.h>

namespace log4cxx
{

    /**
      An appender that appends logging events to a vector.
      */
    class MongoAppender : public AppenderSkeleton
    {
        public:
            DECLARE_LOG4CXX_OBJECT(MongoAppender)
                BEGIN_LOG4CXX_CAST_MAP()
                    LOG4CXX_CAST_ENTRY(MongoAppender)
                    LOG4CXX_CAST_ENTRY_CHAIN(AppenderSkeleton)
                END_LOG4CXX_CAST_MAP()

                MongoAppender();

            void setMDBQClient(mdbq::Client& clt);

            /**
              This method is called by the AppenderSkeleton#doAppend method.
              */
            void append(const spi::LoggingEventPtr& event, log4cxx::helpers::Pool& p);

            void close();

            bool isClosed() const
            { return closed; }

            bool requiresLayout() const
            { return false;   }

        private:
            mdbq::Client* m_clt;
            std::string m_threadname;
    };
    typedef helpers::ObjectPtrT<MongoAppender> MongoAppenderPtr;

    void configureMongoAppender(MongoAppenderPtr& ptr);
}

