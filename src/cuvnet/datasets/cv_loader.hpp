#include <list>
#include <third_party/threadpool/ThreadPool.h>
#include <datasets/dataset.hpp>
#include <cuvnet/common.hpp>
#include "image_queue.hpp"
#include "cv_datasets.hpp"
#include "detection.hpp"

namespace datasets
{
    struct rgb_classification_loader{
        ThreadPool m_pool;
        typedef cuvnet::image_queue<rgb_classification_dataset> queue_t;
        rgb_classification_dataset m_trainset, m_valset;
        cuvnet::image_queue<rgb_classification_dataset> m_trainqueue, m_valqueue;
        std::list<boost::shared_ptr<rgb_classification_dataset::pattern_t> > m_open_list;
        std::list<cuvnet::cv_mode> m_open_list_mode;

        rgb_classification_loader(std::string basename, int n_jobs=0)
        : m_pool(n_jobs)
        , m_trainset(basename + "_train.txt", 224, 1)
        , m_valset(basename + "_val.txt", 224, 5)
        , m_trainqueue(m_pool, m_trainset, 128, 3*128)
        , m_valqueue(m_pool, m_valset, 12, 3*12)
        {
        }
        void load_instance(queue_t& q, int i, cuvnet::matrix& rgb, cuvnet::matrix& klass){
            auto pat = q.pop();
            rgb[cuv::indices[i]] = pat->rgb;
            klass[i] = pat->ground_truth_class;
            m_open_list.push_back(pat);
        }

        void save_instance(const cuvnet::matrix& pred){
            assert(!m_open_list.empty());
            auto pat = m_open_list.front();
            m_open_list.pop_front();
            cuvnet::cv_mode mode = m_open_list_mode.front();
            m_open_list_mode.pop_front();

            pat->predicted_class = pred;

            // dispatch to correct set
            if(mode == cuvnet::CM_TRAIN)
                m_trainset.notify_done(pat);
            else 
                m_valset.notify_done(pat);
        }

        void save_batch(const cuvnet::matrix& pred){
            int batch_size = pred.shape(0);
            for(int i=0; i < batch_size; i++)
                save_instance(pred[cuv::indices[i]]);
        }

        void load_batch(cuvnet::cv_mode mode, cuvnet::matrix& rgb, cuvnet::matrix& tch){
            m_open_list.clear();
            m_open_list_mode.clear();
            int batch_size = rgb.shape(0);
            if(mode == cuvnet::CM_TRAINALL)
                mode = drand48() > (m_trainset.size() / ((float)(m_trainset.size() + m_valset.size())))
                    ? cuvnet::CM_VALID
                    : cuvnet::CM_TRAIN;
            
            for(int i=0; i < batch_size; i++){
                if(mode == cuvnet::CM_TRAIN)
                    load_instance(m_trainqueue, i, rgb, tch);
                else if(mode == cuvnet::CM_VALID || mode == cuvnet::CM_TEST)
                    load_instance(m_valqueue, i, rgb, tch);
                m_open_list_mode.push_back(mode);
            }
            std::cout << "."<<std::flush;
        }
    };

    struct rgb_detection_loader{
        ThreadPool m_pool;
        typedef cuvnet::image_queue<rgb_detection_dataset> queue_t;
        rgb_detection_dataset m_trainset, m_valset;
        cuvnet::image_queue<rgb_detection_dataset> m_trainqueue, m_valqueue;
        std::list<boost::shared_ptr<rgb_detection_dataset::pattern_t> > m_open_list;
        std::list<cuvnet::cv_mode> m_open_list_mode;

        rgb_detection_loader(std::string basename, int n_jobs=0)
        : m_pool(n_jobs)
        , m_trainset(basename + "_train.txt", 224, 1)
        , m_valset(basename + "_val.txt", 224, 5)
        , m_trainqueue(m_pool, m_trainset, 128, 3*128)
        , m_valqueue(m_pool, m_valset, 12, 3*12)
        {
        }
        void load_instance(queue_t& q, int i, cuvnet::matrix& rgb, std::vector<bbox>& tch){
            auto pat = q.pop();
            rgb[cuv::indices[i]] = pat->rgb;
            tch = pat->bboxes;
            m_open_list.push_back(pat);
        }

        void save_instance(const std::vector<bbox>& pred){
            assert(!m_open_list.empty());
            auto pat = m_open_list.front();
            m_open_list.pop_front();
            cuvnet::cv_mode mode = m_open_list_mode.front();
            m_open_list_mode.pop_front();

            pat->predicted_bboxes = pred;

            // dispatch to correct set
            if(mode == cuvnet::CM_TRAIN)
                m_trainset.notify_done(pat);
            else 
                m_valset.notify_done(pat);
        }

        void save_batch(const std::vector<std::vector<bbox> >& pred){
            int batch_size = pred.size();
            for(int i=0; i < batch_size; i++)
                save_instance(pred[i]);
        }

        void load_batch(cuvnet::cv_mode mode, cuvnet::matrix& rgb, std::vector<std::vector<bbox> >& tch){
            m_open_list.clear();
            m_open_list_mode.clear();
            int batch_size = rgb.shape(0);
            if(mode == cuvnet::CM_TRAINALL)
                mode = drand48() > (m_trainset.size() / ((float)(m_trainset.size() + m_valset.size())))
                    ? cuvnet::CM_VALID
                    : cuvnet::CM_TRAIN;

            tch.resize(batch_size);

            for(int i=0; i < batch_size; i++){
                if(mode == cuvnet::CM_TRAIN)
                    load_instance(m_trainqueue, i, rgb, tch[i]);
                else if(mode == cuvnet::CM_VALID || mode == cuvnet::CM_TEST)
                    load_instance(m_valqueue, i, rgb, tch[i]);
                m_open_list_mode.push_back(mode);
            }
            std::cout << "."<<std::flush;
        }

    };

    struct rgbd_detection_loader{
        ThreadPool m_pool;
        typedef cuvnet::image_queue<rgbd_detection_dataset> queue_t;
        rgbd_detection_dataset m_trainset, m_valset;
        cuvnet::image_queue<rgbd_detection_dataset> m_trainqueue, m_valqueue;
        std::list<boost::shared_ptr<rgbd_detection_dataset::pattern_t> > m_open_list;
        std::list<cuvnet::cv_mode> m_open_list_mode;

        rgbd_detection_loader(std::string basename, int n_jobs=0)
        : m_pool(n_jobs)
        , m_trainset(basename + "_train.txt", 224, 1)
        , m_valset(basename + "_val.txt", 224, 5)
        , m_trainqueue(m_pool, m_trainset, 128, 3*128)
        , m_valqueue(m_pool, m_valset, 12, 3*12)
        {
            m_trainset.m_b_train = true;
            m_valset.m_b_train = false;
        }
        void load_instance(queue_t& q, int i, cuvnet::matrix& rgb, cuvnet::matrix& depth, std::vector<bbox>& tch){
            auto pat = q.pop();
            rgb[cuv::indices[i]] = pat->rgb;
            depth[cuv::indices[i][0]] = pat->depth;
            depth[cuv::indices[i][1]] = pat->depth;
            depth[cuv::indices[i][2]] = pat->depth;
            tch = pat->bboxes;
            m_open_list.push_back(pat);
        }

        void save_instance(const std::vector<bbox>& pred){
            assert(!m_open_list.empty());
            auto pat = m_open_list.front();
            m_open_list.pop_front();
            cuvnet::cv_mode mode = m_open_list_mode.front();
            m_open_list_mode.pop_front();

            pat->predicted_bboxes = pred;

            // dispatch to correct set
            if(mode == cuvnet::CM_TRAIN)
                m_trainset.notify_done(pat);
            else 
                m_valset.notify_done(pat);
        }

        void save_batch(const std::vector<std::vector<bbox> >& pred){
            int batch_size = pred.size();
            for(int i=0; i < batch_size; i++)
                save_instance(pred[i]);
        }

        void load_batch(cuvnet::cv_mode mode, cuvnet::matrix& rgb, cuvnet::matrix& depth, std::vector<std::vector<bbox> >& tch){
            m_open_list.clear();
            m_open_list_mode.clear();
            int batch_size = rgb.shape(0);
            if(mode == cuvnet::CM_TRAINALL)
                mode = drand48() > (m_trainset.size() / ((float)(m_trainset.size() + m_valset.size())))
                    ? cuvnet::CM_VALID
                    : cuvnet::CM_TRAIN;

            tch.resize(batch_size);

            for(int i=0; i < batch_size; i++){
                if(mode == cuvnet::CM_TRAIN)
                    load_instance(m_trainqueue, i, rgb, depth, tch[i]);
                else if(mode == cuvnet::CM_VALID || mode == cuvnet::CM_TEST)
                    load_instance(m_valqueue, i, rgb, depth, tch[i]);
                
                // DEBUG
                //cuv::fill_rnd_uniform(rgb);
                
                m_open_list_mode.push_back(mode);
            }
            std::cout << "d"<<std::flush;
        }

    };


}
