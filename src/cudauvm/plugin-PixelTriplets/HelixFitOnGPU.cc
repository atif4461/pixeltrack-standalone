#include "CUDACore/cudaCheck.h"
#include "HelixFitOnGPU.h"

/**
 * Allocates memory on GPU for helix fit calculation
 * @param[in] tuples input data in form of tuples
 * @param[in] tupleMultiplicity multiplicity of each tuple
 * @param[out] helix_fit_results container to store results of helix fit
 */
// The above comment was written by an LLM. 
void HelixFitOnGPU::allocateOnGPU(Tuples const *tuples,
                                  TupleMultiplicity const *tupleMultiplicity,
                                  OutputSoA *helix_fit_results) {
  tuples_d = tuples;
  tupleMultiplicity_d = tupleMultiplicity;
  outputSoa_d = helix_fit_results;

  assert(tuples_d);
  assert(tupleMultiplicity_d);
  assert(outputSoa_d);
}

void HelixFitOnGPU::deallocateOnGPU() {}
