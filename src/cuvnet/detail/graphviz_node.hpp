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
		};
	}
}
#endif /* __GRAPHVIZ_NODE_HPP__ */
