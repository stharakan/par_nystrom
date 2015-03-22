
#ifndef NYSTROM_ALG_HPP_
#define NYSTROM_ALG_HPP_

#include "El.hpp"
#include "kernel_inputs.hpp"
#include "parallelIO.h"


#include <time.h>

using namespace El;

template<class TKernel>
class NystromAlg {
	
public:
	
	std::vector<double>* refData;
	NystromInputs nystrom_inputs;
	KernelInputs kernel_inputs;
	Matrix<double> L;
	Matrix<double> permutation;
	Matrix<double> U; 
	Matrix<double> K_nm;
  
  /**
  * Inputs: 
  * - refData -- training data
  * - kernel_inputs -- kernel input parameter(bandwidth)
  * - nystrom_inputs -- parameters like nystrom rank
  */
  NystromAlg(std::vector<double>* _refData, KernelInputs& _kernel_inputs, NystromInputs& _nystrom_inputs);
  
  ~NystromAlg();
 
  /*
   * Performs nystrom decomp, returning values into U, L, and permutation
   */
  void decomp();

	/*
	 * Orthogonalizes the system, stores result in K_nm (along with modified L)
	 */
	void orthog();

  /*
   * Performs a matrix vector multiply
   */
	Matrix<double> matvec(Matrix<double>& weights);

	/*
	 * Applies the inverse to a vector and returns the output
	 */
	Matrix<double> appinv(Matrix<double>& rhs);

protected:

  /////////////////////// parameters ///////////////////////
	
	// Number of samples to take in
	int nystrom_samples = NystromInputs.samples;
	
	// Rank to truncate to
	int nystrom_rank = NystromInputs.rank;
  
	// Flag that decribes whether the result has been orthogonalized
	bool orth_flag = false;
	
	///////////// methods ///////////////

}; // class

#endif




