#ifndef __GRAPHVIZ_NODE_HPP__
#     define __GRAPHVIZ_NODE_HPP__
#include <string>

namespace cuvnet
{
	namespace detail
	{
		struct graphviz_node{
			std::string label;
			std::string shape;
			std::string group;
			std::string color;
			std::string fillcolor;
			std::string style;
            float penwidth;
            graphviz_node():penwidth(1.f){}
		};
	}
}
#endif /* __GRAPHVIZ_NODE_HPP__ */
