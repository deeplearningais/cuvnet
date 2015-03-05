#include <cuvnet/tools/logging.hpp>
#include "bounding_box_matching.hpp"
#include "../../third_party/graph/bipartite_matching.h"

namespace
{
    log4cxx::LoggerPtr g_log(log4cxx::Logger::getLogger("bboxmatch")); 
}

#define DISABLE_HACK 1

namespace cuvnet
{
    /// calculates log( exp(0) + exp(z) )
    inline double logsumexp_0(double z){
        double tmp = std::max((double)0, z);
        return std::log(std::exp(z-tmp) + std::exp(0-tmp)) + tmp;
    }
    /// calculates 1 / (1 + exp(-z) )
    inline double sigmoid(double z){
        if (z >= 0)
            return           1 / (1 + std::exp(-z));
        else
            return std::exp(z) / (1 + std::exp( z));
    }

    std::vector<std::vector<int> > optimal_matching(
            const std::vector<std::vector<datasets::rotated_rect> >& means, 
            const std::vector<std::vector<datasets::bbox> >& output,
            const float alpha, 
            const std::vector<std::vector<datasets::bbox> >& teach){
        unsigned int bs = teach.size();
        unsigned int K = means.size();
        unsigned int C = output[0].size() / K; // number of classes in output
        
        std::vector<std::vector<int> > matching(bs);
        for (unsigned int b = 0; b < bs; b++) {
            matching[b].resize(C*K, -1);
            for (unsigned int c = 0; c < C; ++c)
            {
                typedef boost::adjacency_list<boost::vecS, boost::vecS, boost::bidirectionalS,
                        boost::property<boost::vertex_name_t, std::string>,
                        boost::property<boost::edge_weight_t,
                        double> > Graph;

                unsigned int teach_boxes = teach[b].size();

                Graph pg(K + teach_boxes);

                /*
                 *boost::property_map<Graph,
                 *    boost::vertex_name_t>::type
                 *        vm= boost::get(boost::vertex_name, pg);
                 */
                boost::property_map<Graph,
                    boost::edge_weight_t>::type
                        ew= boost::get(boost::edge_weight, pg);

                for (unsigned int k = 0; k < K; k++) {
                    for (unsigned int t = 0; t < teach_boxes; t++) {
                        // edge weights are the costs for a given node couple and consists of
                        //  - distance between kmean center and teacher
                        //  - confidence for 
                        // make costs negative, because graph is solved for a maximum weighted matching
                        // add offset for node index due to prediction boxes
                        if (teach[b][t].klass == (int)c) {
                            ew[boost::add_edge(k, K+t, pg).first] = 
                                - std::pow(datasets::rotated_rect::l2dist(means[c][k], teach[b][t].rect), 2)
                                + (output[b][k].confidence > 0 || DISABLE_HACK) * output[b][K*c+k].confidence / alpha;
                                //- logsumexp_0(-conf[b][k])   // these two are equivalent to conf[b][k]
                                //+ logsumexp_0( conf[b][k]);
                        }
                    }
                }

                std::vector<std::pair<int,int> > out
                    = get_maximum_weight_bipartite_matching(pg, K,
                            boost::get(boost::vertex_index, pg),
                            boost::get(boost::edge_weight, pg));

                for (unsigned int i=0; i < out.size(); ++i) {
                    // remove node index offset
                    matching[b][K*c+out[i].first] = out[i].second - K;
                }
            }
        }

        return matching;
    }

    std::pair<float, float> BoundingBoxMatching::loss_terms(){
        float f_match = 0;
        double f_conf = 0;
        unsigned int bs = m_output_bbox.size();

        for (unsigned int b = 0; b < bs; b++) {
            unsigned int foc = 0;
            bool use_foco = false; // use first object class only
            if(m_first_object_class_only && m_teach[b].size() > 0){
                use_foco = true;
                foc = m_teach[b][0].klass;
            }
            //std::vector<bool> matched(m_n_predictions * m_n_klasses);
            for (unsigned int c = 0; c < m_n_klasses; ++c)
            {
                if(use_foco && foc != c)
                    continue;
                for (unsigned int k = 0; k < m_n_predictions; k++) {
                    unsigned int idx = c * m_n_predictions + k;
                    int i_m = m_matching[b][idx];
                    if(i_m >= 0) {
                        f_match += (m_output_bbox[b][idx].confidence > 0|| DISABLE_HACK) * std::pow(datasets::rotated_rect::l2dist(
                                    m_output_bbox[b][idx].rect, m_teach[b][i_m].rect), 2);
                        f_conf += logsumexp_0(-m_output_bbox[b][idx].confidence);  // log of sigmoid
                    } else {
                        f_conf += logsumexp_0( m_output_bbox[b][idx].confidence);  // log of (1-sigmoid)
                    }
                    assert(std::isfinite(f_match));
                    assert(std::isfinite(f_conf));
                }
            }
        }
        f_match *= 1./2;
        return std::make_pair(f_match, (float)f_conf);
    }

    void BoundingBoxMatching::fprop(){
        // prediction of network
        param_t::element_type& p0  = *m_params[0];
        result_t::element_type& r0 = *m_results[0];
       
        // - transfer prediction and confidence to host
        // - calculate output of absolute bboxes
        // - find optimal matching between kmeans and teacher bboxes
        // - calculate loss of bboxes and confidence
       
        //cuv::tensor<float, cuv::host_memory_space> 
        prediction = p0.value.data();

        unsigned int bs = prediction.shape(0);
        prediction.reshape({bs, m_n_klasses, m_n_predictions, 5});
    
        {
            //auto tmp_p = prediction;
            //tmp_p.reshape(bs * m_n_klasses * m_n_predictions, 5);

            //cuv::tensor<float, cuv::host_memory_space> _min(cuv::extents[5]);
            //cuv::reduce_to_row(_min, tmp_p, cuv::RF_MIN);

            //cuv::tensor<float, cuv::host_memory_space> _avg(cuv::extents[5]);
            //cuv::reduce_to_row(_avg, tmp_p, cuv::RF_MEAN);
            
            //cuv::tensor<float, cuv::host_memory_space> _max(cuv::extents[5]);
            //cuv::reduce_to_row(_max, tmp_p, cuv::RF_MAX);

            //float box_min = std::min(std::min(std::min(_min(0), _min(1)), _min(2)), _min(3));
            //float box_max = std::max(std::max(std::max(_max(0), _max(1)), _max(2)), _max(3));
            //float box_avg = (_avg(0) + _avg(1) + _avg(2) + _avg(3)) / 4.;
            //float con_min = _min(4);
            //float con_max = _max(4);
            //float con_avg = _avg(4);

            //LOG4CXX_WARN(g_log, 
            //        "box prediction, min:"<< box_min <<
            //        " , max:"<< box_max<<
            //        " , mean:"<< box_avg);
            //LOG4CXX_WARN(g_log, 
            //        "conf prediction, min:"<< con_min <<
            //        " , max:"<< con_max<<
            //        " , mean:"<< con_avg);
            
            //LOG4CXX_WARN(g_log, "teacher size:"<<m_teach.size());
            /*
            int cnt = 0;
            for(auto bv : m_teach){
                LOG4CXX_WARN(g_log, 
                        "bv size :" << cnt << " -- " << bv.size()
                        );
                for(auto b : bv){
                    LOG4CXX_WARN(g_log, 
                            "(" << b.rect.x << ", " <<
                            b.rect.y << ", " <<
                            b.rect.h << ", " <<
                            b.rect.w << ")"
                            );
                }
            }
            cnt ++;
            */
        }


        m_output_bbox.resize(bs);
        int n_bboxes = 0;
        for (unsigned int b = 0; b < bs; b++) {
            m_output_bbox[b].resize(m_n_predictions * m_n_klasses);
            for(unsigned int c = 0; c < m_n_klasses; c++){
                n_bboxes += m_teach[b].size();
                for (unsigned int k = 0; k < m_n_predictions; k++) {
                    //m_output_bbox[b][k].x = prediction(b, k, 0) + m_typical_bboxes[k].x;
                    //m_output_bbox[b][k].y = prediction(b, k, 1) + m_typical_bboxes[k].y;
                    //m_output_bbox[b][k].h = prediction(b, k, 2) + m_typical_bboxes[k].h;
                    //m_output_bbox[b][k].w = prediction(b, k, 3) + m_typical_bboxes[k].w;
                    m_output_bbox[b][c*m_n_predictions + k].rect.x = prediction(b, c, k, 0) * m_typical_bboxes[c][k].w + m_typical_bboxes[c][k].x;
                    m_output_bbox[b][c*m_n_predictions + k].rect.y = prediction(b, c, k, 1) * m_typical_bboxes[c][k].h + m_typical_bboxes[c][k].y;
                    m_output_bbox[b][c*m_n_predictions + k].rect.h = exp((float)prediction(b, c, k, 2)) * m_typical_bboxes[c][k].h;
                    m_output_bbox[b][c*m_n_predictions + k].rect.w = exp((float)prediction(b, c, k, 3)) * m_typical_bboxes[c][k].w;

                    m_output_bbox[b][c*m_n_predictions + k].confidence = prediction(b, c, k, 4);
                    m_output_bbox[b][c*m_n_predictions + k].klass = c;

                    assert(std::isfinite(std::pow(m_output_bbox[b][c*m_n_predictions + k].rect.x, 2)));
                    assert(std::isfinite(std::pow(m_output_bbox[b][c*m_n_predictions + k].rect.y, 2)));
                    assert(std::isfinite(std::pow(m_output_bbox[b][c*m_n_predictions + k].rect.h, 2)));
                    assert(std::isfinite(std::pow(m_output_bbox[b][c*m_n_predictions + k].rect.w, 2)));
                }
            }
        }

        // find optimal matching between kmeans center of bboxes and teacher bboxes
        // wrt to loss function
        m_matching = optimal_matching(m_typical_bboxes, m_output_bbox, m_alpha, m_teach);

        boost::tie(m_f_match, m_f_conf) = loss_terms(); 
        
        //LOG4CXX_WARN(g_log, "loss terms: " << m_f_match / bs << " " << m_f_conf / bs / m_alpha);
        
        m_loss = m_f_match / bs + m_f_conf / bs / m_alpha;
        
        if(r0.can_overwrite_directly()){
            (*r0.overwrite_or_add_value())[0] = m_loss;
        }
        else if(r0.can_add_directly()){
            (*r0.overwrite_or_add_value())[0] += m_loss;
        }else{
            // reallocate *sigh*
            value_ptr v(new value_type(r0.shape, value_ptr::s_allocator));
            v.data()[0] = m_loss;
            r0.push(v);
        }
        p0.value.reset();
    }
    
    void BoundingBoxMatching::bprop(){
        param_t::element_type&  p0 = *m_params[0];
        result_t::element_type& r0 = *m_results[0];

        float delta = r0.delta.data()[0];
        unsigned int bs = m_output_bbox.size();

        m_delta_matching.resize(bs);
        m_delta_conf.resize(bs);
        for (unsigned int b = 0; b < bs; b++) {
            m_delta_matching[b].resize(m_n_predictions * m_n_klasses);
            m_delta_conf[b].resize(m_n_predictions * m_n_klasses);

            unsigned int foc = 0;
            bool use_foco = false; // use first object class only
            if(m_first_object_class_only && m_teach[b].size() > 0){
                use_foco = true;
                foc = m_teach[b][0].klass;
            }

            for (unsigned int c = 0; c < m_n_klasses; ++c)
            {
                for (unsigned int k = 0; k < m_n_predictions; k++) {
                    unsigned int idx = c * m_n_predictions + k;

                    int i_m = m_matching[b][idx];

                    if (i_m >= 0 && (!use_foco || c == foc)) { // means matching 
                        double fact = 1./bs * (m_output_bbox[b][idx].confidence > 0|| DISABLE_HACK);

                        //m_delta_matching[b][idx].x = fact * (m_output_bbox[b][idx].x - m_teach[b][i_m].rect.x);
                        //m_delta_matching[b][idx].y = fact * (m_output_bbox[b][idx].y - m_teach[b][i_m].rect.y);

                        //m_delta_matching[b][idx].h = fact * (m_output_bbox[b][idx].h - m_teach[b][i_m].rect.h);
                        //m_delta_matching[b][idx].w = fact * (m_output_bbox[b][idx].w - m_teach[b][i_m].rect.w);

                        m_delta_matching[b][idx].x = fact * (m_output_bbox[b][idx].rect.x - m_teach[b][i_m].rect.x) * m_typical_bboxes[c][k].w;
                        m_delta_matching[b][idx].y = fact * (m_output_bbox[b][idx].rect.y - m_teach[b][i_m].rect.y) * m_typical_bboxes[c][k].h;

                        m_delta_matching[b][idx].h = fact * (m_output_bbox[b][idx].rect.h - m_teach[b][i_m].rect.h) * exp((float)prediction(b,c,k,2)) * m_typical_bboxes[c][k].h;
                        m_delta_matching[b][idx].w = fact * (m_output_bbox[b][idx].rect.w - m_teach[b][i_m].rect.w) * exp((float)prediction(b,c,k,3)) * m_typical_bboxes[c][k].w;

                        //m_delta_matching[b][k] = (m_output_bbox[b][k] - m_teach[b][i_m].rect).scale_like_vec();

                        m_delta_conf[b][idx] = - sigmoid(-m_output_bbox[b][idx].confidence) / bs / m_alpha;
                    } else {
                        m_delta_matching[b][idx].x = 0;
                        m_delta_matching[b][idx].y = 0;
                        m_delta_matching[b][idx].h = 0;
                        m_delta_matching[b][idx].w = 0;

                        if(!use_foco || c == foc)
                            m_delta_conf[b][idx] =   sigmoid( m_output_bbox[b][idx].confidence) / bs / m_alpha;
                        else
                            m_delta_conf[b][idx] =   0.f;

                        assert(std::isfinite(m_delta_matching[b][idx].x));
                        assert(std::isfinite(m_delta_matching[b][idx].y));
                        assert(std::isfinite(m_delta_matching[b][idx].h));
                        assert(std::isfinite(m_delta_matching[b][idx].w));
                        assert(std::isfinite(m_delta_conf[b][idx]));
                    }
                }
            }
        }

        cuv::tensor<float, cuv::host_memory_space> grad;
        grad.resize({bs, m_n_klasses, m_n_predictions, 5});
        for (unsigned int b = 0; b < bs; b++) {
            for (unsigned int c = 0; c < m_n_klasses; ++c)
            {
                for (unsigned int k = 0; k < m_n_predictions; k++) {
                    grad(b,c, k,0) = m_delta_matching[b][c*m_n_predictions +k].x;
                    grad(b,c, k,1) = m_delta_matching[b][c*m_n_predictions +k].y;
                    grad(b,c, k,2) = m_delta_matching[b][c*m_n_predictions +k].h;
                    grad(b,c, k,3) = m_delta_matching[b][c*m_n_predictions +k].w;
                    grad(b,c, k,4) = m_delta_conf[b][c*m_n_predictions +k];
                }
            }
        }
        grad.reshape(p0.shape);
        grad *= delta;

        {
            //auto tmp_p = grad;
            //tmp_p.reshape(bs * m_n_klasses * m_n_predictions, 5);

            //cuv::tensor<float, cuv::host_memory_space> _min(cuv::extents[5]);
            //cuv::reduce_to_row(_min, tmp_p, cuv::RF_MIN);

            //cuv::tensor<float, cuv::host_memory_space> _avg(cuv::extents[5]);
            //cuv::reduce_to_row(_avg, tmp_p, cuv::RF_MEAN);
            
            //cuv::tensor<float, cuv::host_memory_space> _max(cuv::extents[5]);
            //cuv::reduce_to_row(_max, tmp_p, cuv::RF_MAX);

            //float box_min = std::min(std::min(std::min(_min(0), _min(1)), _min(2)), _min(3));
            //float box_max = std::max(std::max(std::max(_max(0), _max(1)), _max(2)), _max(3));
            //float box_avg = (_avg(0) + _avg(1) + _avg(2) + _avg(3)) / 4.;
            //float con_min = _min(4);
            //float con_max = _max(4);
            //float con_avg = _avg(4);

            //LOG4CXX_WARN(g_log, 
            //        "box grad, min:"<< box_min <<
            //        " , max:"<< box_max<<
            //        " , mean:"<< box_avg);
            //LOG4CXX_WARN(g_log, 
            //        "conf grad, min:"<< con_min <<
            //        " , max:"<< con_max<<
            //        " , mean:"<< con_avg);
            
            /*
            LOG4CXX_WARN(g_log, 
                    "grad min:"<<cuv::minimum(grad)<<
                    " grad max:"<<cuv::maximum(grad)<<
                    " grad mean:"<<cuv::mean(grad)
                    );
            */
        }

        if(p0.can_overwrite_directly()){
            value_type& v = p0.overwrite_or_add_value().data();
            v = grad;
        }else if(p0.can_add_directly()){
            value_type v(grad);
            p0.overwrite_or_add_value().data() += v;
        }else{
            value_ptr v(new value_type(grad));
            p0.push(v);
        }
    }

    void BoundingBoxMatching::_determine_shapes(){
        m_results[0]->shape.resize(1);
        m_results[0]->shape[0] = 1;

        // assertions for input dimensions 
        cuvAssert(m_params[0]->shape.size() == 2);
        cuvAssert(m_params[0]->shape[1] % (m_n_klasses * 5) == 0);
        cuvAssert(m_params[0]->shape[1] / (m_n_klasses * 5) == m_typical_bboxes[0].size());

        m_n_predictions = m_params[0]->shape[1] / (5 * m_n_klasses);
    }
}
