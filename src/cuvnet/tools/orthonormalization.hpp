#include <cuv.hpp>

/**
 * Symmetric orthonormalization.
 * @ingroup tools
 */
void orthogonalize_symmetric(cuv::tensor<float, cuv::host_memory_space>& m, bool columns=false, bool normalize=true);
/**
 * @ingroup tools
 * Symmetric orthonormalization.
 */
void orthogonalize_symmetric(cuv::tensor<float, cuv::dev_memory_space>& m, bool columns=false, bool normalize=true);
/**
 * Pairwise symmetric orthonormalization.
 * @ingroup tools
 */
void orthogonalize_pairs(cuv::tensor<float, cuv::dev_memory_space>& m, bool columns=false, bool normalize=true);
