
#ifndef NYSTROM_ALG_HPP_
#define NYSTROM_ALG_HPP_


#include "El.hpp"
#include "kernel_inputs.hpp"
#include "nystrom_utils.hpp"
//#include "parallelIO.h"

#include <iostream>
#include <time.h>

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

	// Reordering permutation
	DistMatrix<double,VR,STAR> permute;

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
  NystromAlg(DistMatrix<double>* _refData, KernelInputs& _kernel_inputs, NystromInputs& _nystrom_inputs,Grid* _g); 
  
  ~NystromAlg(); //TODO -- need to free non-matrix variables
 
  /*
   * Performs nystrom decomp, returning values into U, L, and permutation
   */
  void decomp(); //TODO

	/*
	 * Orthogonalizes the system, stores result in K_nm (along with modified L)
	 */
	void orthog(); //TODO

  /*
   * Performs a matrix vector multiply. User must create weight
	 * vector and allocate space for the result of the multiply
   */
	//DistMatrix<double> matvec(DistMatrix<double>& weights); //TODO
	void matvec(DistMatrix<double>& weights); //TODO

	/*
	 * Applies the inverse to a vector and returns the output. As
	 * with matvec, user must create rhs and allocate memory for 
	 * the result
	 */
	//DistMatrix<double> appinv(DistMatrix<double>& rhs); //TODO
	void appinv(DistMatrix<double>& rhs); //TODO

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
	
}; // class

NystromAlg::NystromAlg(DistMatrix<double>* _refData, KernelInputs& _kernel_inputs, NystromInputs& _nystrom_inputs,Grid* _g):
	refData(_refData),
	kernel_inputs(_kernel_inputs),
	nystrom_inputs(_nystrom_inputs),
	g(_g)
{
	
	ntrain          = refData->Height();
	dim             = refData->Width();
	dcmp_flag       = false;
	orth_flag       = false;
	nystrom_rank    = nystrom_inputs.rank;
	nystrom_samples = nystrom_inputs.samples;


	L.SetGrid(*g);
	permute.SetGrid(*g);
	U.SetGrid(*g);
	K_nm.SetGrid(*g);

	// Allocate memory or at least try to
	try{
		L.Resize(nystrom_rank,1);
		permute.Resize(nystrom_rank,1);
		U.Resize(nystrom_samples,nystrom_rank);
		K_nm.Resize(ntrain,nystrom_samples);
	}
	catch(exception& e){ ReportException(e); }
};

NystromAlg::~NystromAlg(){
	// Matrices
	L.Empty();
	permute.Empty();
	U.Empty();
	K_nm.Empty();

	// Options
	// Parameters
	// Flags
}
	
void NystromAlg::decomp(){
		if (!dcmp_flag){
			// Random sample of size nystrom_samples
			//std::vector<Int> smpIdx();	
			
			// Fill with kernel values
			//Matrix<double> K_mm(nystrom_samples,nystrom_samples,*g);


			// Take SVD of subsampled matrix
			//Matrix<double> dummy(*g);
			//Matrix<double> s(*g);
			//SVD(K_mm,dummy,s);

			//TODO Do we need to do this?
			// Sort singular values, store permutation
			

			// Compute K_nm
			
			dcmp_flag = true;
		}
		else{
			std::cout << "Decomposition performed!" << std::endl;
		}
}

void NystromAlg::orthog(){



}

void NystromAlg::matvec(DistMatrix<double>& weights){

	
	
}

void NystromAlg::appinv(DistMatrix<double>& rhs){



}

#endif




