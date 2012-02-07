#ifndef __CUVNET_LOGGING_HPP
#     define __CUVNET_LOGGING_HPP

#include <string>
#include <ostream>
#include <mongo/client/dbclient.h>

namespace cuvnet
{
	class Logger{
		public:
			/**
			 * constructor
			 * 
			 * @param url how to connect to mongodb
			 * @param collection where in db to log 
			 */
			Logger(const std::string& url, const std::string& collection);

			/**
			 * filestream logger
			 * 
			 */
			Logger(std::ostream& os);

			template<class T>
			struct kv{
				kv(const std::string& n, const T& t)
					:k(n),v(t){}
				const std::string& k;
				const T v;
			};
			struct record{
				mongo::DBClientConnection* m_con;
				const std::string* m_collection;
				std::ostream* m_os;
				const int m_level;
				int   m_cnt;
				int   m_mode;
				std::auto_ptr<mongo::BSONObjBuilder> m_objb;
				record(mongo::DBClientConnection* con, const std::string* col, std::ostream* os, int level, int mode);
				~record();
				record(const record& o);
				template<class T>
				record& operator<<(const kv<T>& p){
					m_cnt++;
					*m_objb<<p.k<<p.v;
					return *this;
				}
			};

			/**
			 * use this to start logging a record
			 * e.g. by running
			 *
			 * @code
			 * logger.log(1)<<bson_pair("age",3)
			 *              <<bson_pair("foo","bar");
			 * @endcode
			 */
			record log(int level=4);

		private:
			/// keeps the connection to the db
			mongo::DBClientConnection m_con;

			/// refers to the collection in the db
			const std::string m_collection;

			/// a simple output stream in case db is closed
			std::ostream* m_os;

			/// mode: cout/mongodb
			int m_mode;
	};

	template<class T>
	inline
	Logger::kv<T> bson_pair(const std::string& k, const T& v){ return Logger::kv<T>(k,v); }

	inline
	Logger::kv<std::string> bson_pair(const std::string& k, const char* v){ return Logger::kv<std::string>(k,v); }
}

#endif /* __CUVNET_LOGGING_HPP */
