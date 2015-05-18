
#ifndef NYSTROM_ALG_HPP_
#define NYSTROM_ALG_HPP_


#include "El.hpp"
#include "kernel_inputs.hpp"
#include "nystrom_utils.hpp"
#include "gaussKernel.hpp"
#include "clustering.h"
//#include "parallelIO.h"
#include <numeric>
#include <iostream>
#include <time.h>
#include <vector>
#include <math.h>


using namespace El;


//template<class TKernel>
class NystromAlg {
	
public:
	// Pointer to training data	
	DistMatrix<double>* ptrX;
	
	// Pointer to training labs	
	DistMatrix<double,VR,STAR>* ptrY;
	
	// Kernel
	GaussKernel gKernel;
	
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
	NystromAlg(DistMatrix<double>* _ptrX, int _samp, int _rank, GaussKernel _gKernel, DistMatrix<double,VR,STAR>* _ptrY = NULL);
  
  /**
  * Inputs: 
  * - refData -- training data
  * - kernel_inputs -- kernel input parameter
  * - nystrom_inputs -- parameters like nystrom rank
  */
	NystromAlg(DistMatrix<double>* _ptrX, double _h, int _samp, int _rank = 0, DistMatrix<double,VR,STAR>* _ptrY = NULL);
	
	~NystromAlg();
 
  /*
   * Performs nystrom decomp, returning values into U, L, and permutation
	 * If the flag is true
   */
  void decomp(bool do_orth = false);

	/*
	 * Orthogonalizes the system by calling either os_orthog or qr_orthog based 
	 * on do_qr flag (false => oneshot, true => qr).
	 * Since these are competing algs, once one is called, the other cannot be
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
   * Performs a matvec for oneshot method User must create weight
	 * vector and allocate space for the result of the multiply
   */
	//void os_matvec(DistMatrix<double,VR,STAR>& weights,DistMatrix<double,VR,STAR>& out); //TODO delete this function
	
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
	void regress_test(DistMatrix<double>* Xtest,DistMatrix<double,VR,STAR>* Ytest,std::vector<int> testIdx, double& class_err,double& reg_err,bool exact); 

	/*
	 * Calculates both class correct and regression error given two arbitrary vectors.
	 * Decision point is naturally set to 0, can add more detail for this later
	 */
	void calc_errors(DistMatrix<double,VR,STAR>& Ytest, DistMatrix<double,VR,STAR>& Yguess, double& class_corr, double& reg_err);

	/*
	 * Sets the method to qr instead of one shot
	 */
	void force_qr(){do_qr = true;}

	/* 
	 * Returns dimension index
	 */
	std::vector<int> get_d(){ return d_idx;}

	/*
	 * Returns sample indices
	 */
	std::vector<int> get_smp(){ return smpIdx;}



private:

	// Grid for Elemental
	const Grid* g;

	// Flag that describes whether the result has been decomposed
	bool dcmp_flag = false;
	
	// Flag that describes whether result has been orthogonalized (with oneshot)
	bool orth_flag = false;
	
	// Flag that describes whether we will do qr or not (default is not -- oneshot instead)
	bool do_qr = false;

	// Flag that describes whether we need to sample
	bool samp_flag;

	// Describes total number of training points
	int ntrain;

	// Describes dimension
	int dim;

	// Nystrom inputs: record them here for ease
	int nystrom_rank;
	int nystrom_samples;

	// Store all the indices we'll need
	std::vector<int> s_idx,l_idx,d_idx,dummy_idx;

	// Store the sample index
	std::vector<int> smpIdx;
	
	/*
	 * Orthogonalizes the system, and returns the result in K_nm, so K_nm = K_nm U \
	 * for fast multiplies. The approximation is then given by so that 
	 * K \approx K_nm D K_mn^T, with K_nm is orthogonal.
	 */
	void os_orthog(); 
	
	/*
	 * Orthogonalizes the system, stores result in K_nm (along with modified L)
	 */
	void qr_orthog(); 

}; // class

#endif




