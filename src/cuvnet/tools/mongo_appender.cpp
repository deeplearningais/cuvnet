#include "mongo_appender.hpp"
#include <boost/foreach.hpp>
#include <boost/lexical_cast.hpp>
#include <log4cxx/helpers/thread.h>
#include <log4cxx/basicconfigurator.h>
#include <sys/syscall.h>

#include <mongo/bson/bson.h>

using namespace log4cxx;
using namespace log4cxx::helpers;

IMPLEMENT_LOG4CXX_OBJECT(MongoAppender)

void MongoAppender::append(const spi::LoggingEventPtr& event, Pool& /*p*/)
{
    if(!m_clt)
        throw std::runtime_error("MongoAppender: client not set!");
    //const std::string& thread = event->getThreadName();
    std::string thread;
    event->getMDC("threadname", thread);
    if(thread != m_threadname)
        return;

    mongo::BSONObjBuilder ob;
    auto keys = event->getMDCKeySet();
    BOOST_FOREACH(auto k, keys){
        std::string s;
        event->getMDC(k, s);
        ob.append(k, s);
    }
    {
        std::string ndc;
        event->getNDC(ndc);
        ob.append("_ndc", ndc);
    }
    ob.append("_logger", event->getLoggerName());
    const std::string& msg = event->getMessage();
    ob.append("_msg", msg);
    m_clt->log(event->getLevel()->toInt(), ob.obj());
}

void MongoAppender::close()
{
        if (this->closed) {
                return;
        }
        m_clt->checkpoint();
        this->closed = true;
}


MongoAppender::MongoAppender()
: m_clt(NULL)
{
    m_threadname = boost::lexical_cast<std::string>(
        (unsigned long)syscall(SYS_gettid));
    std::cout << "mongo-logging started for threadname: `" << m_threadname << "'" << std::endl;
}

void MongoAppender::setMDBQClient(mdbq::Client& clt) {
    m_clt = &clt;
}

namespace log4cxx
{
void configureMongoAppender(MongoAppenderPtr& ptr){
    LoggerPtr root(Logger::getRootLogger());
    ptr->setThreshold(Level::getDebug());
    root->addAppender(ptr);
}
}
