
#include "nystrom_alg.hpp"
#include <string>

using namespace El;
using std::string;

NystromAlg::NystromAlg(DistMatrix<double>* _ptrX, int _samp, int _rank, GaussKernel _gKernel, DistMatrix<double,VC,STAR>* _ptrY):
	gKernel(_gKernel)
{
	// Constants
	int proc = mpi::WorldRank();
	ptrX = _ptrX;
	ptrY = _ptrY;
	g = & (ptrX->Grid());
	dim             = _ptrX->Height();
	ntrain          = _ptrX->Width(); 
	nystrom_rank    = _rank;
	nystrom_samples = _samp;
	trunc_flag = nystrom_rank != nystrom_samples;

	// Indexes and ranges
	d_idx.resize(dim);
	dummy_idx.resize(1);
	dummy_idx[0] = 0;
	for(int i=0;i<dim;i++){ d_idx[i] = i;} //TODO omp this loop?
	dummy_rng = Range<int>(0,1);
	s_rng = Range<int>(0,nystrom_rank);
	l_rng = Range<int>(0,nystrom_samples);

	// Set grids
	L.SetGrid(*g);
	D.SetGrid(*g);
	U.SetGrid(*g);
	K_nm.SetGrid(*g);
	
	// Allocate memory or at least try to
	D.Resize(nystrom_rank,1);
	Fill(D,0.0);
	L.Resize(nystrom_rank,1);
	Fill(L,0.0);
	U.Resize(nystrom_samples,nystrom_rank);
	Fill(U,0.0);
	K_nm.Resize(ntrain,nystrom_samples);
};


// Gaussian kernel specific 
NystromAlg::NystromAlg(DistMatrix<double>* _ptrX, double _h, int _samp, int _rank, DistMatrix<double,VC,STAR>* _ptrY)
{
	// Data 
	int proc = mpi::WorldRank();
	ptrX = _ptrX;
	ptrY = _ptrY;
	g = & (ptrX->Grid());
	dim             = _ptrX->Height();
	ntrain          = _ptrX->Width(); 

	// Sampling params
	nystrom_samples = _samp;
	if(!_rank){
		nystrom_rank = _rank;
	}
	else{
		nystrom_rank = nystrom_samples;
	}
	trunc_flag = nystrom_rank != nystrom_samples;

	// Kernel
	gKernel.setParams(_h,g);

	// Set grids
	L.SetGrid(*g);
	D.SetGrid(*g);
	U.SetGrid(*g);
	K_nm.SetGrid(*g);
	
	// Indices and Ranges
	d_idx.resize(dim);
	dummy_idx.resize(1);
	dummy_idx[0] = 0;
	for(int i=0;i<dim;i++){ d_idx[i] = i;} //TODO omp this loop?
	dummy_rng = Range<int>(0,1);
	s_rng = Range<int>(0,nystrom_rank);
	l_rng = Range<int>(0,nystrom_samples);

	// Allocate
	D.Resize(nystrom_rank,1);
	Fill(D,0.0);
	L.Resize(nystrom_rank,1);
	Fill(L,0.0);
	U.Resize(nystrom_samples,nystrom_rank);
	Fill(U,0.0);
	K_nm.Resize(ntrain,nystrom_samples);
};

NystromAlg::~NystromAlg(){
	// Matrices
	L.Empty();
	U.Empty();
	K_nm.Empty();
	D.Empty();

	// Options
	// Parameters
	// Flags
}
	
void NystromAlg::decomp(bool do_orth){
		if (!dcmp_flag){
			// Random sample of size nystrom_samples
			if (samp_flag){
				smpIdx.resize(nystrom_samples);	

				//TODO Share among processes?
				if(mpi::WorldRank() == 0){
					randperm(nystrom_samples,ntrain,smpIdx);
					//std::cout << "Sample idx" << std::endl;
					for (int i=0;i<nystrom_samples;i++){
						//smpIdx[i] = i;
						//std::cout << smpIdx[i] << std::endl;
					}
					std::sort(smpIdx.begin(),smpIdx.end());
				}

				//Send vector to everybody else
				mpi::Broadcast(&smpIdx[0], nystrom_samples, 0, mpi::COMM_WORLD);
			}
			
			// Sample from data
			//if(mpi::WorldRank() == 0){std::cout << "sample"<<std::endl;}
			DistMatrix<double> Xsub(*g);
			GetSubmatrix(*ptrX,d_idx,smpIdx,Xsub); 
			//Print(Xsub,"X_sub");	
			
			// Fill K_mm with kernel values 
			//if(mpi::WorldRank() == 0){std::cout << "small kernel"<<std::endl;}
			DistMatrix<double> K_mm(nystrom_samples,nystrom_samples,*g);
			gKernel.SelfKernel(Xsub, K_mm);
			//Print(K_mm,"K_mm");

			// Take Eigendecomp of subsampled matrix
			//if(mpi::WorldRank() == 0){std::cout << "Eig" << std::endl;}
			auto mmCopy(K_mm);
			DistMatrix<double> Umm(*g);
			DistMatrix<double,VC,STAR> Lmm(*g);
			HermitianEig(UPPER,mmCopy,Lmm,Umm,DESCENDING);
			mmCopy.Empty();

			// Truncate
			//if(mpi::WorldRank() == 0){std::cout <<"truncating" <<std::endl;}
			DiagonalSolve(RIGHT,NORMAL,Lmm,Umm);
			if (trunc_flag){ // need to take a subsample
				// get submatrices
				GetSubmatrix(Umm,l_rng,s_rng,U);
				GetSubmatrix(Lmm,s_rng,dummy_rng,L);
				//DistMatrix<double> Id(*g);
				//Identity(Id, nystrom_samples, nystrom_rank);
				//Gemm(NORMAL,NORMAL, 1.0,Umm,Id, 0.0,U);
				//Gemv(TRANSPOSE, 1.0,Id,Lmm, 0.0,L);
			}
			else{
				U = Umm;
				L = Lmm;
			}
			Umm.Empty();
			Lmm.Empty();

			// Compute K_nm
			//if(mpi::WorldRank() == 0){std::cout << "Gen large kernel" << std::endl;}
			gKernel.Kernel(*ptrX,Xsub,K_nm);	


			dcmp_flag = true;
		}
		else{
			if(mpi::WorldRank() == 0){std::cout << "Decomposition already performed!" << std::endl;}
		}

		// Run orthogonalization if required
		if(do_orth){
			this->orthog();
		}
}

void NystromAlg::orthog(){
	// Check to see that we have decomped
	if(!dcmp_flag){this->decomp();}

	// Run appropriate orthogonalization, if needed
	if(!orth_flag){
		if(do_qr){
			this->qr_orthog();
		}
		else{
			this->os_orthog();
		}
	}
	else{
		if(mpi::WorldRank==0){std::cout<< "Orthogonalization already done" <<std::endl;}
	}
}	

void NystromAlg::os_orthog(){

	// Functions needed for later
	auto elem_sqrt = [](double x){return (sqrt(x));};
	auto elem_32rt = [](double x){return (x * sqrt(x));};
	bool print = mpi::WorldRank() == -1;

	// Allocate extra mem -- A, U_os, L_os; class vars -- V, S
	if(print){std::cout << "Allocating mem"<<std::endl;}
	DistMatrix<double> A(*g);
	DistMatrix<double> B(*g);
	auto U_os(U);
	auto Ucopy(U);
	auto L_os(L);

	U.Empty();
	U.Resize(nystrom_samples,nystrom_samples);
	Fill(U,0.0);
	auto Ahalf(U);

	// Pick out parts of K_nm we want (load into A and B)
	//for(int i=0; i<ntrain; i++){full_idx[i]=i;}
	std::vector<int> full_idx(ntrain);
	int n_oth = ntrain - nystrom_samples;
	std::vector<int> oth_idx(n_oth);
	std::iota(full_idx.begin(),full_idx.end(),0);
	std::set_difference(full_idx.begin(),full_idx.end(),smpIdx.begin(),smpIdx.end(),oth_idx.begin());

	// Pick out parts of the kernel matrix (currently recalculating //TODO)
	if(0){
		GetSubmatrix(K_nm,oth_idx,smpIdx,B); //K_(n-m),m 
		GetSubmatrix(K_nm,smpIdx,smpIdx,A); //K_mm
	}else{
		DistMatrix<double> Xoth(dim,n_oth,*g);
		DistMatrix<double> Xsub(dim,nystrom_samples,*g);
		B.Resize(n_oth,nystrom_samples);
		Fill(B,0.0);
		A.Resize(nystrom_samples,nystrom_samples);
		Fill(A,0.0);

		GetSubmatrix(*ptrX,d_idx,oth_idx,Xoth);
		GetSubmatrix(*ptrX,d_idx,smpIdx,Xsub);

		// Compute kernels (symmetric if we can)
		gKernel.Kernel(Xoth,Xsub,B);
		if(trunc_flag){
			gKernel.Kernel(Xsub,Xsub,A);
		}else{
			gKernel.SelfKernel(Xsub,A);
		}
		Xoth.Empty();
		Xsub.Empty();
	}

	// First make B^T B
	Syrk(UPPER,ADJOINT,1.0,B, 0.0,U); 
	B.Empty();

	// Form Ahalf
	EntrywiseMap(L_os,function<double(double)>(elem_32rt));// Ahalf = K_mm^-1/2 
	DiagonalScale(RIGHT,NORMAL, L_os, U_os); // U_os = U L^-1/2 //TODO change this to Syrk (half comp)
	Gemm(NORMAL,TRANSPOSE, 1.0,U_os,Ucopy, 0.0,Ahalf); // Ahalf = U_os * U^T
	Ucopy.Empty();
	
	// If we truncated, need to adjust to preserve orthogonality
	if(trunc_flag){
		// Form Kmm^-1/2 (A^2 + B^T B) K_mm^-1/2
		Gemm(NORMAL,NORMAL, 1.0,A,A, 1.0, U); //Inner matrix
		Fill(A,0.0); // Don't need A anymore
	}
	
	// Form Kmm^-1/2 B^T B K_mm^-1/2 (inner matrix is changed if trunc)
	B.Resize(nystrom_samples,nystrom_samples); //  B = V * Ahalf; empty V
	Fill(B,0.0);
	Symm(LEFT,UPPER, 1.0,U,Ahalf, 0.0, B); 
	U.Empty();
			
	// A = K_mm + Kmm^-1/2 * B^T B K_mm^-1/2
	Gemm(NORMAL,NORMAL, 1.0,Ahalf,B, 1.0, A); 
	B.Empty();
	

	if(print){std::cout << "eig"<<std::endl;}
	// Eigendecompose A into U_os L_os
	HermitianEig(UPPER,A,L_os,U_os,DESCENDING);
	A.Empty();

	// Form D = L_os
	if(print){std::cout << "form s"<<std::endl;}
	D = L_os;

	// Form V = A^-1/2 U_os L_os^-1/2
	if(print){std::cout << "form v"<<std::endl;}
	U.Resize(nystrom_samples,nystrom_samples);
	Fill(U,0.0);
	EntrywiseMap(L_os,function<double(double)>(elem_sqrt)); //U_os = U_os * L^-1/2
	DiagonalSolve(RIGHT,NORMAL,L_os,U_os);
	Gemm(NORMAL,NORMAL, 1.0,Ahalf,U_os, 0.0,U); // U = Ahalf * U_os

	// Load K_nm for fast multiplies so K_nm = K_nm * U
	auto Kcopy(K_nm);
	K_nm.Resize(ntrain,nystrom_samples);
	Fill(K_nm,0.0);
	Gemm(NORMAL,NORMAL, 1.0,Kcopy,U, 0.0,K_nm);

	// Let everyone know
	orth_flag = true;

}

void NystromAlg::qr_orthog(){
	//Form the large U (KU below)
	DistMatrix<double> KU(ntrain,nystrom_rank,*g);
	Fill(KU,0.0);
	Gemm(NORMAL,NORMAL,1.0,K_nm,U,1.0,KU);
	K_nm.EmptyData(); //Don't need this until later

	//Take the QR
	DistMatrix<double,MD,STAR> t(*g);
	DistMatrix<double,MD,STAR> d(*g);
	QR(KU,t,d);

	// Form and restrict large Q
	DistMatrix<double> Q_nm(*g);
	Identity(Q_nm,ntrain,nystrom_rank);
	qr::ApplyQ(LEFT,NORMAL,KU,t,d,Q_nm);

	//Form R
	DistMatrix<double> R(*g);
	GetSubmatrix(KU,s_rng,s_rng,R); //TODO Make this more efficient
	//KU.Resize(t.Height(),KU.Width());
	//auto R(KU);
	KU.Empty();
	t.Empty();
	d.Empty();
	MakeTrapezoidal(UPPER,R);

	// Form R L R^T
	auto elem_sqrt = [](double x){return (sqrt(x));};
	EntrywiseMap(L,function<double(double)>(elem_sqrt));
	DiagonalScaleTrapezoid(RIGHT,UPPER,NORMAL,L,R); // Now R_curr = R_true L^(1/2) 
	Trtrmm(UPPER,R); // Now R_curr = R_old R_old^T  = R_true L R_true

	// Eigendecompose that product
	auto RLRt(R);
	DistMatrix<double> smallQ(*g);
	DistMatrix<double> newL(*g);
	HermitianEig(UPPER,R,newL,smallQ,DESCENDING);
	R.Empty();

	// Load combined Q into K_nm
	K_nm.Resize(ntrain,nystrom_rank);
	Fill(K_nm,0.0);
	Gemm(NORMAL,NORMAL, 1.0,Q_nm,smallQ, 1.0,K_nm);
	smallQ.Empty();
	Q_nm.Empty();

	//Load new L into D
	D = newL;
	newL.Empty();

	// Let everyone else know, (save U for potential multiplies)
	orth_flag = true;

}

void NystromAlg::matvec(DistMatrix<double>* Xtest, DistMatrix<double,VC,STAR>& weights, DistMatrix<double,VC,STAR>& out){
	// IF we have only decomped,  out = K_(test)m U L U^T K_nm^T weights
	// IF we have orthogonalized, out = K_(test)m U D K_nm^T weights
	// Either way we must compute K_(test)m
	int testpts = Xtest->Width();


	// Compute K_nm * w, may or may not include small U factor
	DistMatrix<double, VC, STAR> Kw(nystrom_samples,1,*g);
	Fill(Kw,0.0);
	Gemv(TRANSPOSE,1.0,K_nm,weights,1.0,Kw);

	DistMatrix<double,VC,STAR> dummy(*g);
	
	// Do either multiply by D, or L * U^T
	if( !(orth_flag) ){
		// Compute U L U^T * Kw
		dummy.Resize(nystrom_rank,1);
		Fill(dummy,0.0);
		Gemv(TRANSPOSE,1.0,U,Kw,1.0,dummy);
 
		DiagonalScale(LEFT,NORMAL,L,dummy);
	}
	else{
		// Scale by D
		dummy = Kw;
		DiagonalScale(LEFT,NORMAL,D,dummy);
		
	}

	// Multiply by U (same in either case)
	Kw.Resize(U.Height(),1);
	Fill(Kw,0.0);
	Gemv(NORMAL,1.0,U,dummy,1.0,Kw);
	dummy.EmptyData();
	
	// Get submatrix according to smpIdx
	DistMatrix<double> Xsub(*g);
	GetSubmatrix(*ptrX,d_idx,smpIdx,Xsub);

	
	// Form kernel both decomps need
	DistMatrix<double> K_tm(*g);
	K_tm.Resize(testpts,nystrom_samples);
	gKernel.Kernel(*Xtest,Xsub,K_tm);

	// Finish by applying K_tm
	out.Resize(testpts,1);
	Fill(out,0.0);
	Gemv(NORMAL,1.0,K_tm,Kw,1.0,out);
}

void NystromAlg::matvec(DistMatrix<double,VC,STAR>& weights, DistMatrix<double,VC,STAR>& out){
	// IF we have orthogonalized, out = K_nm D K_nm^T weights
	// IF we have only decomped,  out = K_nm U L U^T K_nm^T weights
	// Either way, can do Kw = K_nm^T * w
	
	DistMatrix<double, VC, STAR> Kw(K_nm.Width(),1,*g);
	Fill(Kw,0.0);
	Gemv(TRANSPOSE,1.0,K_nm,weights,1.0,Kw);


	// Apply either just D, or K_mm = U L U^T, store output in Kw
	if( !(orth_flag) ){
		DistMatrix<double,VC, STAR> dummy(nystrom_rank,1,*g);
		Fill(dummy,0.0);
		
		Gemv(TRANSPOSE,1.0,U,Kw,1.0,dummy);

		DiagonalScale(LEFT,NORMAL,L,dummy);

		Fill(Kw,0.0);
		Gemv(NORMAL,1.0,U,dummy,1.0,Kw);
		dummy.Empty();
	}
	else{
		DiagonalScale(LEFT,NORMAL,D,Kw);
	}
	
	// Set up output vector properly
	out.Resize(ntrain,1);
	Fill(out,0.0);

	// Finish by applying K_nm
	Gemv(NORMAL,1.0,K_nm,Kw,1.0,out);
}

void NystromAlg::appinv(DistMatrix<double,VC,STAR>& rhs, DistMatrix<double,VC,STAR>& x,int r){
	// Make sure it is orthogonalized
	if(!orth_flag){
		if(mpi::WorldRank() == 0){std::cout << "Need to orthogonalize first .." << std::endl;}
		this->orthog();
	}
	
	bool print = mpi::WorldRank() == -1;

	// Kapprox = K_nm D K_nm^T, so just need to invert diag
	// since K_nm is orthogonal
	DistMatrix<double,VC,STAR> Kw(nystrom_rank,1,*g);
	Fill(Kw,0.0);
	Gemv(TRANSPOSE, 1.0,K_nm,rhs, 1.0,Kw);

	if( r && r < nystrom_samples){
		auto rKw(Kw);
		auto rD(D);

		// TODO use getsubmatrix for safety
		rD.Resize(r,1);
		rKw.Resize(r,1);

		DiagonalSolve(LEFT,NORMAL,rD,rKw);
		rD.Empty();

		Fill(Kw,0.0);
		DistMatrix<double> I(*g);
		Identity(I,nystrom_samples,r); //TODO make better
		Gemv(NORMAL, 1.0,I,rKw, 0.0,Kw);
	}else{
		
		// Scale by inv diag
		DiagonalSolve(LEFT,NORMAL,D,Kw);
	}
	
	// Finish multiply, load into x
	x.Resize(ntrain,1);
	Fill(x,0.0);
	Gemv(NORMAL, 1.0,K_nm,Kw, 1.0,x);

	// Free the dummy vector
	Kw.Empty();

}

double NystromAlg::nullspace(DistMatrix<double,VC,STAR>& weights, DistMatrix<double,VC,STAR>& null_vec, int r){
	// Make sure it is orthogonalized
	if(!orth_flag){
		if(mpi::WorldRank() == 0){std::cout << "Need to orthogonalize first .." << std::endl;}
		this->orthog();
	}

	// Make sure output is appropriately sized, initialize
	null_vec.Resize(ntrain,1);
	Fill(null_vec,0.0);
	DistMatrix<double,VC,STAR> dummy(*g);

	// Calculate Ur^T Ur w
	// First check if we need to restrict to r eigenvectors
	if( r && r < nystrom_samples){
		dummy.Resize(r,1);
		Fill(dummy,0.0);
		Gemv(TRANSPOSE,1.0,K_nm,weights, 0.0,dummy);
		
		DistMatrix<double,VC,STAR> dummy2(nystrom_samples,1,*g);
		Fill(dummy2,0.0);
		DistMatrix<double> I(*g);
		Identity(I,nystrom_samples,r); //TODO make better
		Gemv(NORMAL, 1.0,I,dummy, 0.0,dummy2);
		I.Empty();
		dummy.Empty();

		Gemv(NORMAL, 1.0,K_nm,dummy2, 0.0, null_vec);
		dummy2.Empty();
	}
	else{
		dummy.Resize(nystrom_samples,1);	
		Fill(dummy,0.0);
		Gemv(TRANSPOSE,1.0,K_nm,weights, 0.0,dummy);

		Gemv(NORMAL, 1.0,K_nm,dummy, 0.0, null_vec);
		dummy.Empty();
	}
	
	return FrobeniusNorm(null_vec);
}


void NystromAlg::matvec_errors(std::vector<double>& loc_weights,std::vector<int> testIdx,int runs,double& avg_err,double& avg_time){
	// Initialize all the stuff we need
	double tot_err = 0.0;
	double tot_time = 0.0;
	int testSize = testIdx.size();
	int nlocal = loc_weights.size();
	DistMatrix<double,VC,STAR> vec(ntrain,1,*g);
	DistMatrix<double,VC,STAR> err(ntrain,1,*g);
	DistMatrix<double,VC,STAR> err_sub(testSize,1,*g);

	/*
	// Get offset
	//int offset = mpi::GlobalRank() * nlocal; // Completely equal
	int comms = mpi::Size(mpi::COMM_WORLD);
	std::vector<int> sends(comms,nlocal);
	std::vector<int> recvs(comms);
	std::vector<int> offsets(comms);
	mpi::AllToAll(&sends[0],1,&recvs[0],1,mpi::COMM_WORLD);
	std::partial_sum(sends.begin(),sends.end(),offsets.begin());
	int rank = mpi::WorldRank() - 1;
	int offset;
	if(rank==0){
		offset = 0;
	}else{
		offset = offsets[rank];
	}
	
	// Populate vec -- might be slow
	for(int i=0;i<nlocal;i++){
		vec.Set(offset + i, 1,loc_weights[i]);
	}	*/

	// Make el vector from local vecs
	make_vec_from_loc(loc_weights,vec);

	// Form true kernel for given sample idx
	DistMatrix<double> Xsub(*g);
	GetSubmatrix(*ptrX,d_idx,testIdx,Xsub);
	DistMatrix<double> K(testSize,ntrain,*g);
	gKernel.Kernel(Xsub,*ptrX,K);

	// Approximate kernel-vec into ans, time
	Fill(err,0.0);
	Fill(err_sub,0.0);
	Uniform(vec,ntrain,1);

	double start = mpi::Time();
	//if (mpi::WorldRank() == 0) {std::cout << "Approx matvec" <<std::endl;}
	this->matvec(vec,err);
	avg_time = mpi::Time() - start;
	GetSubmatrix(err,testIdx,dummy_idx,err_sub);

	// Find exact kernel-vec, subtract from ans
	//if (mpi::WorldRank() == 0) {std::cout << "Exact matvec" <<std::endl;}
	Gemv(NORMAL, -1.0,K,vec, 1.0,err_sub);
	double abs_err = FrobeniusNorm(err_sub);

	Fill(err_sub,0.0);
	Gemv(NORMAL, 1.0,K,vec, 0.0,err_sub);
	double base_norm = FrobeniusNorm(err_sub);
	// Compute relative error
	avg_err = abs_err /  base_norm;


	// Empty and finalize data
	vec.Empty();
	err.Empty();
}

void NystromAlg::matvec_errors(std::vector<int> testIdx,int runs,double& avg_err,double& avg_time){
	// Initialize all the stuff we need
	double tot_err = 0.0;
	double tot_time = 0.0;
	int testSize = testIdx.size();
	DistMatrix<double,VC,STAR> vec(*g);
	DistMatrix<double,VC,STAR> err(ntrain,1,*g);
	DistMatrix<double,VC,STAR> err_sub(testSize,1,*g);

	// Form true kernel for given sample idx
	DistMatrix<double> Xsub(*g);
	GetSubmatrix(*ptrX,d_idx,testIdx,Xsub);
	DistMatrix<double> K(testSize,ntrain,*g);
	gKernel.Kernel(Xsub,*ptrX,K);

	// Do the runs
	for(int run=0;run<runs;run++){
		// Approximate kernel-vec into ans, time
		Fill(err,0.0);
		Fill(err_sub,0.0);
		Uniform(vec,ntrain,1);

		double start = mpi::Time();
		//if (mpi::WorldRank() == 0) {std::cout << "Approx matvec" <<std::endl;}
		this->matvec(vec,err);
		tot_time += mpi::Time() - start;
		GetSubmatrix(err,testIdx,dummy_idx,err_sub);
		
		// Find exact kernel-vec, subtract from ans
		//if (mpi::WorldRank() == 0) {std::cout << "Exact matvec" <<std::endl;}
		Gemv(NORMAL, -1.0,K,vec, 1.0,err_sub);
		double abs_err = FrobeniusNorm(err_sub);

		Fill(err_sub,0.0);
		Gemv(NORMAL, 1.0,K,vec, 0.0,err_sub);
		double base_norm = FrobeniusNorm(err_sub);
		// Compute relative error
		double rel_err = abs_err /  base_norm;
		tot_err += rel_err;

	}
	
	// Empty and finalize data
	vec.Empty();
	err.Empty();
	avg_err = tot_err/runs;
	avg_time = tot_time/runs;
}

void NystromAlg::regress_test(DistMatrix<double>* Xtest,DistMatrix<double,VC,STAR>* Ytest,std::vector<int> testIdx,double& class_corr,double& reg_err, bool exact){
	// Make sure it is orthogonalized
	if(!orth_flag){
		if(mpi::WorldRank() == 0){std::cout << "Need to orthogonalize first .." << std::endl;}
		this->orthog();
	}
	
	// Take subset
	int testpts = testIdx.size();
	DistMatrix<double> Xtsub(*g);
	DistMatrix<double,VC,STAR> Ytsub(*g);
	GetSubmatrix(*Xtest,d_idx,testIdx,Xtsub);
	GetSubmatrix(*Ytest,testIdx,dummy_idx,Ytsub);

	// Find weights
	DistMatrix<double,VC,STAR> weight_vec(*g);
	this->appinv(*ptrY,weight_vec);
	DistMatrix<double,VC,STAR> Yguess(*g);
	Yguess.Resize(testpts,1);

	// If exact, only run on testIdx, else approximate for all w/multiply
	if(exact){
		// Form K_tn
		DistMatrix<double> K_tn(*g);
		K_tn.Resize(testpts,ntrain);
		gKernel.Kernel(Xtsub,*ptrX,K_tn);

		// Multiply K_tn w
		Fill(Yguess,0.0);
		Gemv(NORMAL, 1.0,K_tn,weight_vec, 1.0,Yguess);
	}
	else{
		// Multiply ~ K_tn w
		this->matvec(&Xtsub,weight_vec,Yguess);
	}

	// Test errors
	this->calc_errors(Ytsub,Yguess,class_corr,reg_err);
}

void NystromAlg::calc_errors(DistMatrix<double,VC,STAR>& Ytest, DistMatrix<double,VC,STAR>& Yguess, double& class_corr, double& reg_err){
	// Class err
	Int pts = Ytest.Height();
	auto elem_sign = [](double x){return (std::copysign(0.5,x));};
	auto y1(Ytest);
	EntrywiseMap(y1,function<double(double)>(elem_sign));
	auto y2(Yguess);
	EntrywiseMap(y2,function<double(double)>(elem_sign));
	Axpy(1.0,y2,y1);
	auto elem_sqr = [](double x){return (x*x);};
	EntrywiseMap(y1,function<double(double)>(elem_sqr));
	Fill(y2,1.0);
	class_corr = Dot(y1,y2)/pts;
	y2.Empty();

	// Regression err
	double base_norm = FrobeniusNorm(Ytest);
	y1 = Yguess;
	Axpy(-1.0,Ytest,y1);
	reg_err = FrobeniusNorm(y1)/base_norm;

}

