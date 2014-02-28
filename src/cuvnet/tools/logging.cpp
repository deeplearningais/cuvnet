#include <iostream>
#include <iomanip>
#include "logging.hpp"
#include <boost/lexical_cast.hpp>

#include "log4cxx/logger.h"
#include "log4cxx/basicconfigurator.h"
#include "log4cxx/propertyconfigurator.h"
#include "log4cxx/patternlayout.h"
#include "log4cxx/consoleappender.h"
#include "log4cxx/fileappender.h"
#include "log4cxx/writerappender.h"
#include "log4cxx/xml/xmllayout.h"
#include "log4cxx/helpers/exception.h"
#include "log4cxx/level.h"
#include "log4cxx/ndc.h"
#include "log4cxx/mdc.h"

namespace cuvnet
{
    Logger::Logger(const std::string& fn){
        using namespace log4cxx;
    
        xml::XMLLayoutPtr myXMLLayoutPtr = new xml::XMLLayout();
        //myXMLLayoutPtr->setLocationInfo(true); // __FILE__ and __LINE__
        myXMLLayoutPtr->setProperties(true); // MDC objects
        FileAppenderPtr myXMLFileAppenderPtr = new FileAppender(myXMLLayoutPtr, fn, false, true, 1024);
    
        // OtrosLogViewer Setup: TIMESTAMP LEVEL [THREAD] NDC LOGGER - MESSAGE
        //PatternLayoutPtr myLayoutPtr = new PatternLayout("%d{ISO8601} %-5p [%t] %x %c - %m%n");
        PatternLayoutPtr myLayoutPtr = new PatternLayout("%d{ISO8601} %-5p %x %c - %m%n");
        ConsoleAppenderPtr myAppenderPtr = new ConsoleAppender(myLayoutPtr);
        myAppenderPtr->setThreshold(Level::getInfo());
        //FileAppenderPtr myFileAppenderPtr = new FileAppender(myLayoutPtr, "log.txt", false, true, 1024);

        char* hn = new char[64];
        gethostname(hn, 64);
        new MDC("host", hn);
    
        BasicConfigurator::configure(myAppenderPtr);
        //BasicConfigurator::configure(myFileAppenderPtr);
        BasicConfigurator::configure(myXMLFileAppenderPtr);

        {
            log4cxx::LoggerPtr log(log4cxx::Logger::getLogger("determine_shapes"));
            log->setLevel(Level::getError());
        }
    }

    Tracer::Tracer(const Tracer&){
        // is private and should therefore never be called
    }
    Tracer& Tracer::operator=(const Tracer&){
        // is private and should therefore never be called
        return *this;
    }
    Tracer::Tracer(log4cxx::LoggerPtr& log, const std::string& msg)
        :m_msg(msg)
        ,m_log(log)
    {
        using namespace log4cxx;
        LOG4CXX_DEBUG(log, "ENTRY " << m_msg);
        m_ndc = new NDC(m_msg);
        m_mdc = NULL;
    }
    Tracer::Tracer(log4cxx::LoggerPtr& log, const std::string& msg, const std::string& var, const std::string& id)
        :m_msg(msg)
        ,m_log(log)
    {
        using namespace log4cxx;
        LOG4CXX_DEBUG(log, "ENTRY " << m_msg);
        m_mdc = new MDC(var, id);
        m_ndc = new NDC(m_msg);
    }
    Tracer::~Tracer(){
        m_tim.update();
        log4cxx::MDC mdc_duration(m_msg, boost::lexical_cast<std::string>(1000.f * m_tim.perf()));
        LOG4CXX_DEBUG(m_log, 
                "EXIT " << m_msg << 
                " ms=" << std::setprecision(4) 
                <<  1000.f * m_tim.perf());
        delete m_ndc;
        delete m_mdc;
    }

}
