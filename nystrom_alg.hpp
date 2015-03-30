
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

	// Reordering permutation
	DistMatrix<int,VR,STAR> permute; //TODO is this needed?

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
	void matvec(DistMatrix<double>& weights,DistMatrix<double>& out); 

	/*
	 * Applies the inverse to a vector and returns the output. As
	 * with matvec, user must create rhs and allocate memory for 
	 * the result
	 */
	void appinv(DistMatrix<double>& rhs, DistMatrix<double>& x); 

	/*
	 * Computes the Gaussian kernel from a set of points when compared to itself.
	 * The inputs are specified in KernelInputs
	 */
	void self_GaussKernel(DistMatrix<double>& data, DistMatrix<double>& K); 


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

NystromAlg::NystromAlg(DistMatrix<double>* _refData, KernelInputs& _kernel_inputs, NystromInputs& _nystrom_inputs,Grid* _g, GaussKernel _gKernel):
	refData(_refData),
	kernel_inputs(_kernel_inputs),
	nystrom_inputs(_nystrom_inputs),
	g(_g),
	gKernel(_gKernel)
{
	ptrX = refData;
	ptrY = refData;
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
	
	l_idx.resize(nystrom_samples);
	s_idx.resize(nystrom_rank);
	d_idx.resize(dim);
	dummy_idx.resize(1);
	dummy_idx[1] = 0;
	for(int i=0;i<nystrom_samples;i++){ l_idx[i] = i;} //TODO omp this loop
	for(int i=0;i<nystrom_rank;i++){ s_idx[i] = i;} //TODO omp this loop
	for(int i=0;i<dim;i++){ d_idx[i] = i;} //TODO omp this loop

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
			std::vector<Int> smpIdx(nystrom_samples);	
	
			//TODO Share among processes?
			if(mpi::WorldRank() == 0){
				std::vector<int> _smpIdx;
				randperm(nystrom_samples,ntrain,_smpIdx);
				std::cout << "Sample idx" << std::endl;
				for (int i=0;i<nystrom_samples;i++){
					std::cout << _smpIdx[i] << std::endl;
					smpIdx[i] = _smpIdx[i];
				}
				_smpIdx.clear();
			}

			//Send vector to everybody else
			mpi::Broadcast(&smpIdx[0], nystrom_samples, 0, mpi::COMM_WORLD);
			
			//int A[4] = {1,0,0,1};
			//std::vector<double> A_vec(A,A+4);
			//if(mpi::WorldRank() ==0){
			//	for (int i=0; i<A_vec.size();i++){
			//		std::cout <<A_vec[i]<<std::endl;
			//	}
			//}
			
			// Sample from data
			DistMatrix<double> Xsub(*g);
			if(mpi::WorldRank() ==0){
				std::cout << "Full idx (dim)" <<std::endl;
				for(int i =0;i<d_idx.size();i++){
					std::cout << d_idx[i] <<std::endl;
				}
			}
			GetSubmatrix(*ptrX,smpIdx,d_idx,Xsub);
			Print(Xsub,"X_mn");	
			// Fill K_mm with kernel values 
			DistMatrix<double> K_mm(nystrom_samples,nystrom_samples,*g);
			gKernel.SelfKernel(Xsub, K_mm);
			Print(K_mm,"K_mm");
			//fill with rands for now //TODO take out
			//Uniform(K_mm,nystrom_samples,nystrom_samples,0.25,0.25);
			//DistMatrix<double> mmCopy(*g);
			//FillDiagonal(K_mm,0.5);
			//Copy(K_mm,mmCopy);
			//AdjointAxpy(1.0,mmCopy,K_mm);
			//mmCopy.Empty();
			
			// Take Eigendecomp of subsampled matrix
			auto mmCopy(K_mm);
			DistMatrix<double> Umm(*g);
			DistMatrix<double> Lmm(*g);
			HermitianEig(UPPER,mmCopy,Lmm,Umm,DESCENDING);
			mmCopy.Empty();

			// This is just because we cant read yet //TODO take out
			//double minL = Min(Lmm).value;
			//if(minL < 0.0){
			//	DistMatrix<double> Lcorr(nystrom_samples,1,*g);
			//	Fill(Lcorr,1.01);
			//	Axpy(-minL, Lcorr, Lmm);
			//}
			
			// Truncate
			GetSubmatrix(Umm,l_idx,s_idx,U);
			GetSubmatrix(Lmm,s_idx,dummy_idx,L);
			Umm.Empty();
			Lmm.Empty();

			// Compute K_nm
			gKernel.Kernel(*ptrX,Xsub,K_nm);	
			
			dcmp_flag = true;
		}
		else{
			if(mpi::WorldRank() == 0){std::cout << "Decomposition already performed!" << std::endl;}
		}
}

void NystromAlg::orthog(){
	// If we haven't done decomp, do that now
	if(!dcmp_flag){this->decomp();}

	// Check if we've already orthogonalized
	if(!orth_flag){
		K_nm.EmptyData(); //Don't need this until later
		
		//Form the large U (KU below)
		DistMatrix<double> KU(ntrain,nystrom_rank,*g);
		Fill(KU,0.0);
		Uniform(K_nm,ntrain,nystrom_samples,0.5,0.5);
		Gemm(NORMAL,NORMAL,1.0,K_nm,U,1.0,KU);

		//Take the QR
		DistMatrix<double,MD,STAR> t(*g);
		DistMatrix<double,MD,STAR> d(*g);
		QR(KU,t,d);

		//Form R L R^T
		DistMatrix<double> R(*g);
		GetSubmatrix(KU,s_idx,s_idx,R); 
		MakeTrapezoidal(UPPER,R);
		auto elem_sqrt = [](double x){return (sqrt(x));};//TODO check L is nonneg before calling this?
		EntrywiseMap(L,function<double(double)>(elem_sqrt));
		DiagonalScaleTrapezoid(RIGHT,UPPER,NORMAL,L,R); // Now R_curr = R_true L^(1/2) 
		Trtrmm(UPPER,R); // Now R_curr = R_old R_old^T  = R_true L R_true

		// Eigendecompose that product
		auto RLRt(R);
		DistMatrix<double> smallQ(*g);
		DistMatrix<double> newL(*g);
		HermitianEig(UPPER,R,newL,smallQ,DESCENDING);
		
		// Restrict Q
		DistMatrix<double> Q_nm(*g);
		Identity(Q_nm,ntrain,nystrom_rank);
		qr::ApplyQ(LEFT,NORMAL,KU,t,d,Q_nm);
		KU.Empty();
		t.Empty();
		d.Empty();
		
		// Load combined Q into K_nm
		K_nm.Resize(ntrain,nystrom_rank);
		Fill(K_nm,0.0);
		Gemm(NORMAL,NORMAL, 1.0,Q_nm,smallQ, 1.0,K_nm);
		smallQ.Empty();
		Q_nm.Empty();
		
		//Load new L into L
		L = newL;
		newL.Empty();

		// Let everyone else know, also don't need U anymore --> dont know if 
		// emptying it helps
		orth_flag = true;
		//U.Empty();
	}
	else{
		if(mpi::WorldRank() == 0){std::cout << "Already orthogonalized!"<<std::endl;}
	}

}

void NystromAlg::matvec(DistMatrix<double>& weights, DistMatrix<double>& out){
	
	// IF we have orthogonalized, out = K_nm L K_nm^T weights
	// IF we have only decomped,  out = K_nm U L U^T K_nm^T weights
	// Either way, can do Kw = K_nm^T * w
	DistMatrix<double, VR, STAR> Kw(ntrain,1,*g);
	Fill(Kw,0.0);
	Gemv(TRANSPOSE,1.0,K_nm,weights,1.0,Kw);
	
	// Apply either just L, or K_mm = U L U^T, store output in Kw
	if(!orth_flag){
		DistMatrix<double,VR, STAR> dummy(nystrom_rank,1,*g);
		Fill(dummy,0.0);
		Gemv(TRANSPOSE,1.0,U,Kw,1.0,dummy);
		Fill(Kw,0.0);
		DiagonalScale(RIGHT,NORMAL,L,dummy);
		Gemv(NORMAL,1.0,U,dummy,1.0,Kw);
		dummy.EmptyData();
	}
	else{
		DiagonalScale(RIGHT,NORMAL,L,Kw);
	}
	
	// Set up output vector properly
	out.Resize(ntrain,1);
	Fill(out,0.0);

	// Finish by applying K_nm
	Gemv(NORMAL,1.0,K_nm,Kw,1.0,out);
}

void NystromAlg::appinv(DistMatrix<double>& rhs, DistMatrix<double>& x){
	// Make sure it is orthogonalized
	if(!orth_flag){
		if(mpi::WorldRank() == 0){
			std::cout << "Need to orthogonalize first .." << std::endl;
		}
		this->orthog();
	}
	
	// Kapprox = K_nm L K_nm^T, so just need to invert diag
	// since K_nm is orthogonal
	// 
	DistMatrix<double,VR,STAR> Kw(nystrom_rank,1,*g);
	Fill(Kw,0.0);
	Gemv(TRANSPOSE, 1.0,K_nm,rhs, 1.0,Kw);

	// Scale by inv diag
	DiagonalSolve(RIGHT,NORMAL,L,Kw);

	// Finish multiply, load into x
	x.Resize(nystrom_rank,1);
	Fill(x,0.0);
	Gemv(TRANSPOSE, 1.0,K_nm,Kw, 1.0,x);
	
	// Free the dummy vector
	Kw.Empty();
}

#endif




