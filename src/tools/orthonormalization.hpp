#include <cuv.hpp>

void orthogonalize_symmetric(cuv::tensor<float, cuv::host_memory_space>& m, bool columns=false);
void orthogonalize_symmetric(cuv::tensor<float, cuv::dev_memory_space>& m, bool columns=false);
void orthogonalize_pairs(cuv::tensor<float, cuv::dev_memory_space>& m, bool columns=false);
