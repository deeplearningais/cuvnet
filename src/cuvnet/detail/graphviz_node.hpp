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
			std::string color;
			std::string fillcolor;
			std::string style;
		};
	}
}
#endif /* __GRAPHVIZ_NODE_HPP__ */
