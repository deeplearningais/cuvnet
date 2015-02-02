#ifndef __CUVNET_PATTERNSET_HPP__
#     define __CUVNET_PATTERNSET_HPP__
#include <vector>
#include <boost/shared_ptr.hpp>
#include <boost/enable_shared_from_this.hpp>

    /// set of patterns which belong together, eg sliding windows over the same image
    /// @ingroup datasets
    template<class Pattern>
        struct pattern_set : public boost::enable_shared_from_this<pattern_set<Pattern> >{
            /// the type of the patterns
            typedef Pattern pattern_type;

            /// patterns which have been generated, but still need to be
            /// processed by the network
            std::vector<boost::shared_ptr<Pattern> > m_todo;

            /// patterns which are currently being processed by the network
            std::vector<boost::weak_ptr<Pattern> > m_processing;

            /// patterns which have been processed by the network
            std::vector<boost::shared_ptr<Pattern> > m_done;

            /// add a pattern to the TODO list
            /// @note this creates a circle, the pattern points to the set and vice versa.
            /// once the pattern is moved to processing, the circle is broken.
            /// just don't keep around large numbers of patternsets you don't intend to process!
            inline void push(boost::shared_ptr<pattern_type> p){
                p->set = this->shared_from_this();
                m_todo.push_back(p);
            }

            /// get a pattern for processing, also moves it from todo to processing internally.
            inline boost::shared_ptr<Pattern> get_for_processing(){
                assert(!m_todo.empty());
                boost::shared_ptr<pattern_type> pat = m_todo.back();
                m_processing.push_back(boost::weak_ptr<pattern_type>(pat));
                m_todo.pop_back();
                return pat;
            }

            struct cmp_weak_strong{
                const boost::shared_ptr<Pattern>& q;
                cmp_weak_strong(const boost::shared_ptr<Pattern>& _q):q(_q){}
                bool operator()(const boost::weak_ptr<Pattern>& p){
                    return p.lock() == q;
                }
            };

            /// tell the pattern_set that the pattern p can be moved from processing to done.
            /// @warning this creates a circle of shared_ptrs which you have to resolve 
            ///          by clearing m_done once everything you're done looking at it!
            inline void notify_processed(boost::shared_ptr<Pattern> p){
                typename std::vector<boost::weak_ptr<Pattern> >::iterator it 
                    = std::find_if(m_processing.begin(), m_processing.end(), cmp_weak_strong(p));
                cuvAssert(it != m_processing.end());
                m_processing.erase(it);
                m_done.push_back(p);  // NOTE this creates a circle again! You have to make sure to clear m_done in your overload.
            }

            /// this is a special marker pattern which can be put in the queue to signal the end of a loop through the dataset.
            inline bool is_end_marker(){
                return m_todo.size() == 0 
                    && m_processing.size() == 0
                    && m_done.size() == 0;
            }
            inline size_t todo_size()const{return m_todo.size();}
            inline size_t processing_size()const{return m_processing.size();}
        };
#endif /* __CUVNET_PATTERNSET_HPP__ */
