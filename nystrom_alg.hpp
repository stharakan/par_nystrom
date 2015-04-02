
#ifndef NYSTROM_ALG_HPP_
#define NYSTROM_ALG_HPP_


#include "El.hpp"
#include "kernel_inputs.hpp"
#include "nystrom_utils.hpp"
#include "gaussKernel.hpp"
#include "clustering.h"
//#include "parallelIO.h"

#include <iostream>
#include <time.h>
#include <vector>
#include <math.h>

using namespace El;


//template<class TKernel>
class NystromAlg {
	
public:
	// Pointer to training data	
	DistMatrix<double>* refData;
	
	// Defines the nystrom parameters
	NystromInputs nystrom_inputs;
	
	// Defines the kernel parameters
	KernelInputs kernel_inputs;

	// Matrix which holds the spectrum (decreasing order)
	DistMatrix<double,VR,STAR> L;

	// U matrix for the sample points (truncated to reduced rank)
	DistMatrix<double> U; 

	// Kernel matrix between refData and samples / Orthogonal U
	DistMatrix<double> K_nm;
  
  /**
  * Inputs: 
  * - refData -- training data
  * - kernel_inputs -- kernel input parameter
  * - nystrom_inputs -- parameters like nystrom rank
  */
  NystromAlg(DistMatrix<double>* _refData, KernelInputs& _kernel_inputs, NystromInputs& _nystrom_inputs,Grid* _g, GaussKernel _gKernel); 
  
  ~NystromAlg(); //TODO -- need to free non-matrix variables
 
  /*
   * Performs nystrom decomp, returning values into U, L, and permutation
   */
  void decomp(); 

	/*
	 * Orthogonalizes the system, stores result in K_nm (along with modified L)
	 */
	void orthog(); 

  /*
   * Performs a matrix vector multiply. User must create weight
	 * vector and allocate space for the result of the multiply
   */
	void matvec(DistMatrix<double,VR,STAR>& weights,DistMatrix<double,VR,STAR>& out); 

	/*
	 * Applies the inverse to a vector and returns the output. As
	 * with matvec, user must create rhs and allocate memory for 
	 * the result
	 */
	void appinv(DistMatrix<double,VR,STAR>& rhs, DistMatrix<double,VR,STAR>& x); 

	/*
	 * Computes the average matvec error and time to compute it 
	 * over the number of runs specified in runs
	 */
	void matvec_errors(std::vector<int> testIdx,int runs,double& avg_err,double& avg_time); 

private:

	// Grid for Elemental
	Grid* g;

	// Flag that describes whether the result has been decomposed
	bool dcmp_flag;
	
	// Flag that describes whether the result has been orthogonalized
	bool orth_flag;

	// Describes total number of training points
	int ntrain;

	// Describes dimension
	int dim;

	// Nystrom inputs: record them here for ease
	int nystrom_rank;
	int nystrom_samples;

	// Pointers to data
	DistMatrix<double>* ptrX;
	DistMatrix<double>* ptrY;

	// Kernel
	GaussKernel gKernel;

	// Make a class for this later
	void kernelCompute(Matrix<double>& sources, Matrix<double>& targets, Matrix<double>& kernel);

	// Store all the indices we'll need
	std::vector<int> s_idx,l_idx,d_idx,dummy_idx;

}; // class

#endif




