
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
	
	// Matrix which holds the spectrum after orth (decreasing order)
	DistMatrix<double,VR,STAR> D;
	
	// U matrix for the sample points (truncated to reduced rank)
	DistMatrix<double,MC,MR> U; 

	// Kernel matrix between refData and samples / Orthogonal U
	DistMatrix<double,MC,MR> K_nm;
  
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
   * Performs a matrix vector multiply with targets different from sources. 
	 * User must create weight vector, pass in the data points of the new 
	 * targets and allocate space for the result of the multiply
   */
	void matvec(DistMatrix<double>* Xtest, DistMatrix<double,VR,STAR>& weights,DistMatrix<double,VR,STAR>& out); 
	
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

	/*
	 * Performs regression on a given test data set/label combination.
	 * Reports both classification error and regression (l2) error
	 */
	void regress_test(DistMatrix<double>* Xtest,DistMatrix<double,VR,STAR>* Ytest,double& class_err,double& reg_err,bool exact);

	/*
	 * Calculatess both class correct and regression error given two arbitrary vectors.
	 * Decision point is naturally set to 0, can add more detail for this later
	 */
	void calc_errors(DistMatrix<double,VR,STAR>& Ytest, DistMatrix<double,VR,STAR>& Yguess, double& class_corr, double& reg_err);

private:

	// Grid for Elemental
	Grid* g;

	// Flag that describes whether the result has been decomposed
	bool dcmp_flag;
	
	// Flag that describes whether the result has been orthogonalized
	bool orth_flag;

	// Flag that describes whether we need to sample
	bool samp_flag;

	// Describes total number of training points
	int ntrain;

	// Describes dimension
	int dim;

	// Nystrom inputs: record them here for ease
	int nystrom_rank;
	int nystrom_samples;

	// Pointers to data
	DistMatrix<double>* ptrX;
	DistMatrix<double,VR,STAR>* ptrY;

	// Kernel
	GaussKernel gKernel;

	// Store all the indices we'll need
	std::vector<int> s_idx,l_idx,d_idx,dummy_idx;

	// Store the sample index
	std::vector<Int> smpIdx;

}; // class

#endif




