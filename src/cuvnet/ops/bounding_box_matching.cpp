#include <cuvnet/tools/logging.hpp>
#include "bounding_box_matching.hpp"
#include "../../third_party/graph/bipartite_matching.h"

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
            const std::vector<BoundingBoxMatching::bbox>& means, 
            const std::vector<std::vector<float> >& conf,
            const float alpha, 
            const std::vector<std::vector<BoundingBoxMatching::bbox> >& teach){
        unsigned int bs = teach.size();
        unsigned int K = means.size();
        
        std::vector<std::vector<int> > matching(bs);
        for (unsigned int b = 0; b < bs; b++) {
            matching[b].resize(K, -1);
            typedef boost::adjacency_list<boost::vecS, boost::vecS, boost::bidirectionalS,
                    boost::property<boost::vertex_name_t, std::string>,
                    boost::property<boost::edge_weight_t,
                    double> > Graph;

            unsigned int teach_boxes = teach[b].size();

            Graph pg(K + teach_boxes);

            boost::property_map<Graph,
                boost::vertex_name_t>::type
                    vm= boost::get(boost::vertex_name, pg);
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
                    ew[boost::add_edge(k, K+t, pg).first] = 
                        - alpha * BoundingBoxMatching::bbox::l2dist(means[k], teach[b][t])
                        - -logsumexp_0(-conf[b][k]);
                }
            }

            std::vector<std::pair<int,int> > out
                = get_maximum_weight_bipartite_matching(pg, K,
                        boost::get(boost::vertex_index, pg),
                        boost::get(boost::edge_weight, pg));
            
            for (unsigned int i=0; i < out.size(); ++i) {
                // remove node index offset
                matching[b][out[i].first] = out[i].second - K;
            }
        }

        return matching;
    }

    std::pair<float, float> BoundingBoxMatching::loss_terms(){
        float f_match = 0;
        double f_conf = 0;
        unsigned int bs = m_output_bbox.size();

        for (unsigned int b = 0; b < bs; b++) {
            std::vector<bool> matched(m_K);
            for (unsigned int k = 0; k < m_K; k++) {
                int i_m = m_matching[b][k];
                if(i_m >= 0) {
                    f_match += std::pow(BoundingBoxMatching::bbox::l2dist(
                            m_typical_bboxes[k] + m_output_bbox[b][k], m_teach_bbox[b][i_m]), 2);
                    f_conf += logsumexp_0(-m_output_conf[b][k]);  // log of sigmoid
                } else {
                    f_conf += logsumexp_0( m_output_conf[b][k]);  // log of (1-sigmoid)
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
       
        cuv::tensor<float, cuv::host_memory_space> prediction = p0.value.data();
        //cuv::tensor<float, cuv::host_memory_space> confidence = p1.value.data();

        unsigned int bs = prediction.shape(0);
        m_K = prediction.shape(1);
        
        m_output_bbox.resize(bs);
        m_output_conf.resize(bs);
        for (unsigned int b = 0; b < bs; b++) {
            m_output_bbox[b].resize(m_K);
            m_output_conf[b].resize(m_K);
            for (unsigned int k = 0; k < m_K; k++) {
                m_output_bbox[b][k].x_min = prediction(b, k, 0) + m_typical_bboxes[k].x_min;
                m_output_bbox[b][k].y_min = prediction(b, k, 1) + m_typical_bboxes[k].y_min;
                m_output_bbox[b][k].x_max = prediction(b, k, 2) + m_typical_bboxes[k].x_max;
                m_output_bbox[b][k].y_max = prediction(b, k, 3) + m_typical_bboxes[k].y_max;

                m_output_conf[b][k] = prediction(b, k, 4);
            }
        }

        // find optimal matching between kmeans center of bboxes and teacher bboxes
        m_matching = optimal_matching(m_typical_bboxes, m_output_conf, m_alpha, m_teach_bbox);

        boost::tie(m_f_match, m_f_conf) = loss_terms(); 
        float loss = m_alpha * m_f_match + m_f_conf;

        if(r0.can_overwrite_directly()){
            (*r0.overwrite_or_add_value())[0] = loss;
        }
        else if(r0.can_add_directly()){
            (*r0.overwrite_or_add_value())[0] += loss;
        }else{
            // reallocate *sigh*
            value_ptr v(new value_type(r0.shape, value_ptr::s_allocator));
            v.data()[0] = loss;
            r0.push(v);
        }
        p0.value.reset();
    }
    
    void BoundingBoxMatching::bprop(){
        param_t::element_type&  p0 = *m_params[0];
        result_t::element_type& r0 = *m_results[0];

        float delta = r0.delta.data()[0];
        unsigned int bs = m_output_bbox.size();

        std::vector<std::vector<bbox> > delta_matching(bs);
        std::vector<std::vector<float> > delta_conf(bs);
        for (unsigned int b = 0; b < bs; b++) {
            delta_matching[b].resize(m_K);
            delta_conf[b].resize(m_K);
            for (unsigned int k = 0; k < m_K; k++) {
                int i_m = m_matching[b][k];

                if (i_m >= 0) { // means matching 
                    delta_matching[b][k] = 
                        (m_typical_bboxes[k] + m_output_bbox[b][k]
                         - m_teach_bbox[b][i_m]).scale_like_vec(m_alpha);
                
                    delta_conf[b][k] = - sigmoid(-m_output_conf[b][k]);
                } else {
                    delta_conf[b][k] =   sigmoid( m_output_conf[b][k]);
                }
            }
        }

        cuv::tensor<float, cuv::host_memory_space> grad;
        grad.resize(p0.shape);
        for (unsigned int b = 0; b < bs; b++) {
            for (unsigned int k = 0; k < m_K; k++) {
                grad(b,k,0) = delta_matching[b][k].x_min;
                grad(b,k,1) = delta_matching[b][k].y_min;
                grad(b,k,2) = delta_matching[b][k].x_max;
                grad(b,k,3) = delta_matching[b][k].y_max;
                grad(b,k,4) = delta_conf[b][k];
            }
        }
        grad *= delta;

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
        cuvAssert(m_params[0]->shape.size() == 3);
        cuvAssert(m_params[0]->shape[2] == 5);
        cuvAssert(m_params[0]->shape[1] == m_typical_bboxes.size());
    }
}
