#ifndef __CUVNET_LOGGING_HPP
#     define __CUVNET_LOGGING_HPP

#include <string>
#include <ostream>
#include <cuv/tools/timing.hpp>
#include <log4cxx/logger.h>

namespace log4cxx
{
    class MDC;
    class NDC;
}
namespace cuvnet
{
    /**
     * Initializes logging via log4cxx
     * @ingroup tools
     */
	class Logger{
		public:
			/**
			 * constructor
			 */
			Logger(const std::string& fn = "log.xml");
	};

#define TRACE(logger, msg) Tracer _trace(logger, msg);
#define TRACE1(logger, msg, var, id) Tracer _trace(logger, msg, var, boost::lexical_cast<std::string>(id)) ;

    /**
     * logs at when instantiated and when destroyed.
     */
    class Tracer{
        private:
            std::string m_msg;
            Timing m_tim;
            log4cxx::MDC* m_mdc;
            log4cxx::NDC* m_ndc;
            log4cxx::LoggerPtr m_log;
            Tracer(const Tracer&);
            Tracer& operator=(const Tracer&);
        public:
            Tracer(log4cxx::LoggerPtr& log, const std::string& msg);
            Tracer(log4cxx::LoggerPtr& log, const std::string& msg, const std::string& var, const std::string& val);
            ~Tracer();
    };
}

#endif /* __CUVNET_LOGGING_HPP */
