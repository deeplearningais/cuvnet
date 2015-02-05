// source: http://www.salilab.org/~drussel/bipartite_matching.html

#ifndef DSR_BIPARTITE_MATCHING_H
#define DSR_BIPARTITE_MATCHING_H

#include <vector>
#include <limits>
#include <boost/graph/adjacency_list.hpp>

namespace bipartite_matching_detail {
  //template <class ItE, class ItW, class OItP>
  typedef std::pair<int,int> Edge;
  typedef double Weight;
  typedef std::vector<Edge > Edges;
  typedef Edges Out;
  typedef std::vector<double> Weights;
  typedef std::vector<bool> Prime;
  typedef std::pair<int, Weight> WeightedEdge;
  typedef std::vector<WeightedEdge> Neighbors;
  typedef std::vector<WeightedEdge> Distances;
  typedef std::vector<Neighbors> UAdjacency;
  typedef std::vector<WeightedEdge> VAdjacency;

  inline Edge make_edge(int u, int v){
    return Edge(u,v);
  }

 
  inline void flip_edge(int u, int v, UAdjacency &ua, VAdjacency &va){
    // add edge u,v to the matching
    for (unsigned int i=0; i< ua[u].size(); ++i){
      if (ua[u][i].first==v){
        Weight w= ua[u][i].second;
	  
        ua[u].erase(ua[u].begin() + i);
        assert(va[v].first == -1);
        va[v]=WeightedEdge(u, w);
        return;
      }
    }
    assert(0);
  }

  inline void flip_edge(int v, int u, VAdjacency &va, UAdjacency &ua){
    // remove edge u,v from the matching
    assert(va[v].first == u);
    va[v].first = -1;
    ua[u].push_back(WeightedEdge(v, va[v].second));
    va[v].second= Weight(-1);
  }

  inline void relax(int u,
                    Weight udist,
                    const Neighbors &uadj,
                    Distances &vdists){
    for (unsigned int i=0; i< uadj.size(); ++i){
      int v= uadj[i].first;
      // negative since we flip that edge to out of the set
      Weight dist= udist - uadj[i].second;
      if (vdists[v].second > dist) {
        vdists[v].first= u;
        vdists[v].second= dist;
      }
    }
  }
 
  inline void relax(int v,
                    Weight vdist,
                    const WeightedEdge &vadj,
                    Distances &udists){
    if (vadj.first == -1) return; // there is no incoming edge
    int u= vadj.first;
    Weight ndist = vdist + vadj.second;
    if (ndist < udists[u].second){
      udists[u] = WeightedEdge(v, ndist);
    }
  }


  inline void find_augmenting_path(const Prime &up, const Prime &vp, 
                                   const UAdjacency& uadj, const VAdjacency &vadj,
                                   std::back_insert_iterator<Edges > out){
    if (up.empty() || vp.empty()) return;
    Distances u_dists(uadj.size());
    Distances v_dists(vadj.size());
    for (unsigned int i=0; i< u_dists.size(); ++i){
      if (!up[i]){
        u_dists[i]=WeightedEdge(-1, std::numeric_limits<Weight>::max());
      } else {
        u_dists[i]=WeightedEdge(-1, 0);
      }
    }
    for (unsigned int i=0; i< v_dists.size(); ++i){
      v_dists[i]=WeightedEdge(-1, std::numeric_limits<Weight>::max());
    }

    for (unsigned int i=0; i< uadj.size() + vadj.size()-1; ++i){
      for (unsigned int j=0; j< u_dists.size(); ++j){
        relax(j, u_dists[j].second, uadj[j], v_dists);
      }
      for (unsigned int j=0; j< v_dists.size(); ++j){
        relax(j, v_dists[j].second, vadj[j], u_dists);
      }
    }

    // extract path
    Weight mind= std::numeric_limits<Weight>::max();
    int minv(-1);
    for (unsigned int v=0; v< vp.size(); ++v){
      if (vp[v] && v_dists[v].second < mind){
        mind= v_dists[v].second;
        minv= v;
      }
    }
    if (minv==-1){
      // we did not succeed in connected the two, so we are done.
      return;
    } else {
      int curu= v_dists[minv].first;
      std::pair<int,int> ret(curu, minv);
      *out= ret;
      ++out;
      while (u_dists[curu].first != -1){
        int pred= u_dists[curu].first;
        curu= v_dists[pred].first;
        std::pair<int,int> ret(curu, pred);
        *out= ret;
        ++out;
      }
    }
  }

  inline void compute(UAdjacency &uadj,
                      VAdjacency &vadj,
                      Out& out){
    // make U,V prime
    Prime up(uadj.size(), true);
    Prime vp(vadj.size(), true);
    unsigned int remaining_ind= std::min(uadj.size(), vadj.size());

    while (remaining_ind != 0) {
      //std::cout << "remaining " << remaining_ind << std::endl;
      std::vector<std::pair<int,int> > path;
      // this returns the path in reverse order,
      // it does not matter currently since the length is always odd
      find_augmenting_path(up, vp, uadj, vadj, std::back_inserter(path));
      /*std::cout << "path is [";
      for (unsigned int i=0; i< path.size(); ++i) {
        std::cout << " (" << path[i].first << ", " << path[i].second << ")";
      }
      std::cout << "]" << std::endl;*/
      if (!path.empty()){
        /*std::cout << "Found path " << path.size() << ": ";
          for (unsigned int i=path.size()-1; i< path.size(); --i){
          std::cout << "u" << path[i].first << " v" << path[i].second << " ";
          }
          std::cout << std::endl;*/

        for (unsigned int i=0; i+1< path.size(); ++i){
          // flip all edges out of match.Looks strange because the path is returned backwards
          flip_edge(path[i+1].second, path[i].first, vadj, uadj);
        }
        for (unsigned int i=0; i< path.size(); ++i){
          // flip all edges into of match
          flip_edge(path[i].first, path[i].second, uadj, vadj);
        }
        vp[path.front().second]=false;
        up[path.back().first]=false;
        --remaining_ind;

        /*std::cout << "Matching is ";
          for (unsigned int i=0; i< vadj.size(); ++i){
          V v(i);
          if (vadj[i].first != -1){
          std::cout <<  "(" << vadj[v].first<< "-" << v << ") ";
          }
          }
          std::cout << std::endl;*/
      } else {
        break;
      }
    }

    // extract matching
    for (unsigned int i=0; i< vadj.size(); ++i){
      int v(i);
      if (vadj[i].first != -1){
        out.push_back( make_edge(vadj[v].first, v) );
      }
    }
  }


};



/** Compute the maximum weight bipartite matching on a bipartite
    graph.

    \param[in] g A boost edge and vertex list bipartite graph
    \param[in] na Vertices [0,na) must be in the first set.
    \param[in] im The vertex index property map for the graph.
    \param[in] wm The edge weight property map for the graph.

    \return a list of pairs of vertex indexes for the edges
    connecting the two sets. In each pair the first is in
    [0,na) and the second in [na,boost::num_vertices(g)).
*/
template <class BoostGraph, class IndexMap, class WeightMap>
inline std::vector<std::pair< int, int> >
get_maximum_weight_bipartite_matching(const BoostGraph &g,
                                      unsigned int na,
                                      const IndexMap &im,
                                      const WeightMap &wm) {
  std::vector<std::pair<int, int> > out;
  // copy graph
  typedef typename boost::graph_traits<BoostGraph>::edge_iterator EI;
  std::pair<EI, EI> edges=boost::edges(g);


  typename bipartite_matching_detail::UAdjacency uadj(na);
  typename bipartite_matching_detail::VAdjacency vadj(boost::num_vertices(g)-na,
                                                      bipartite_matching_detail::WeightedEdge(-1,
                                                                                              std::numeric_limits<double>::max()));
  for (; edges.first != edges.second; ++edges.first){
    int s= im[boost::source(*edges.first, g)];
    int t= im[boost::target(*edges.first, g)];
    if (s> t) std::swap(s,t);
    double w= wm[*edges.first];


    uadj[s].push_back(bipartite_matching_detail::WeightedEdge(t-na, w));
  }

  bipartite_matching_detail::compute(uadj, vadj, out);
  typename std::vector<typename std::pair<int, int> >
    ret(out.size());
  for (unsigned int i=0; i< out.size(); ++i) {
    // assume vertices are indices
    ret[i]= std::make_pair(out[i].first, out[i].second+na);
  }
  return ret;
}

#endif // DSR_BIPARTITE_MATCHING_H
