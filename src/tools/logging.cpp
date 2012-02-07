#include "logging.hpp"

namespace cuvnet
{
	Logger::Logger(const std::string& url, const std::string& collection)
		:m_collection(collection)
		,m_os(NULL) // dummy
		,m_mode(1)
	{
		m_con.connect(url);
	}
	Logger::Logger(std::ostream& os)
		: m_os(&os)
		, m_mode(0)
	{
	}
	Logger::record::record(mongo::DBClientConnection* con, const std::string* col, std::ostream* os, int level, int mode)
		:m_con(con),m_collection(col), m_os(os), m_level(level), m_cnt(0) , m_mode(mode)
	{
		m_objb.reset(new mongo::BSONObjBuilder());
		m_objb->append("level",level);
	}
	Logger::record::record(const Logger::record::record& o)
		:m_con(o.m_con),m_collection(o.m_collection), m_os(o.m_os), m_level(o.m_level), m_cnt(o.m_cnt) , m_mode(o.m_mode)
	{
		m_objb = const_cast<std::auto_ptr<mongo::BSONObjBuilder>&>(o.m_objb);
	}
	Logger::record::~record(){
		if(m_cnt>0){
			if(m_mode==1)
				m_con->insert(*m_collection, m_objb->obj());
			else
				*m_os << m_objb->obj();
		}
	}
	Logger::record
	Logger::log(int level){
		return record(&m_con,&m_collection,m_os,level,m_mode);
	}
}
