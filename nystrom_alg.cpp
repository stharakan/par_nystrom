
#include "nystrom_alg.hpp"
#include <string>

using namespace El;
using std::string;

NystromAlg::NystromAlg(DistMatrix<double>* _ptrX, DistMatrix<double,VR,STAR>* _ptrY, NystromInputs& _nystrom_inputs,Grid* _g, GaussKernel _gKernel):
	nystrom_inputs(_nystrom_inputs),
	g(_g),
	gKernel(_gKernel)
{
	int proc = mpi::WorldRank();
	
	ptrX = _ptrX;
	ptrY = _ptrY;
	//DistMatrix<double,VR,STAR> dummy(1,1,*_g);
	//ptrY = &dummy; //TODO change this
	dim             = _ptrX->Height();
	ntrain          = _ptrX->Width(); 
	dcmp_flag       = false;
	orth_flag       = false;
	os_flag         = false;
	nystrom_rank    = _nystrom_inputs.rank;
	nystrom_samples = _nystrom_inputs.samples;
	samp_flag = nystrom_rank != nystrom_samples;


	S.SetGrid(*_g);
	V.SetGrid(*_g);
	L.SetGrid(*_g);
	D.SetGrid(*_g);
	U.SetGrid(*_g);
	K_nm.SetGrid(*_g);
	
	//l_idx.resize(nystrom_samples);
	//s_idx.resize(nystrom_rank); //TODO Uncomment if we are orthogonalizing
	d_idx.resize(dim);
	dummy_idx.resize(1);
	dummy_idx[0] = 0;
	s_idx.resize(nystrom_rank);
	l_idx.resize(nystrom_samples);
	for(int i=0;i<nystrom_samples;i++){ l_idx[i] = i;} //TODO omp this loop?
	for(int i=0;i<nystrom_rank;i++){ s_idx[i] = i;} //TODO omp this loop?
	for(int i=0;i<dim;i++){ d_idx[i] = i;} //TODO omp this loop?

	// Allocate memory or at least try to
	//if(mpi::WorldRank() ==0){
	//	std::cout<< "Proc: " << proc  << " allocating memory..."<<std::endl;
	//	std::cout<< "Proc: " << proc  << " rank: " << nystrom_rank <<std::endl;
	//	std::cout<< "Proc: " << proc  << " samp: " << nystrom_samples <<std::endl;
	//	std::cout<< "Proc: " << proc  << " dim: " << dim <<std::endl;
	//	std::cout<< "Proc: " << proc  << " ntrain: " << ntrain <<std::endl;
	//}
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
	
	
void NystromAlg::decomp(){
		if (!dcmp_flag){
			// Random sample of size nystrom_samples
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
			DistMatrix<double,VR,STAR> Lmm(*g);
			HermitianEig(UPPER,mmCopy,Lmm,Umm,DESCENDING);
			mmCopy.Empty();

			// Truncate
			//if(mpi::WorldRank() == 0){std::cout <<"truncating" <<std::endl;}
			DiagonalSolve(RIGHT,NORMAL,Lmm,Umm);
			if (samp_flag){ // need to take a subsample
				//GetSubmatrix(Umm,l_idx,s_idx,U);
				//GetSubmatrix(Lmm,s_idx,dummy_idx,L);
				DistMatrix<double> Id(*g);
				Identity(Id, nystrom_samples, nystrom_rank);
				Gemm(NORMAL,NORMAL, 1.0,Umm,Id, 0.0,U);
				Gemv(TRANSPOSE, 1.0,Id,Lmm, 0.0,L);
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
}

void NystromAlg::oneshot(){
	// Runs the first couple steps of the alg 
	if(!dcmp_flag){this->decomp();}

	// Check if we ran os already
	if(!os_flag){
		auto elem_sqrt = [](double x){return (sqrt(x));};
		auto elem_32rt = [](double x){return (x * sqrt(x));};
		bool print = mpi::WorldRank() == -1;

		// Allocate extra mem -- A, U_os, L_os; class vars -- V, S
		if(print){std::cout << "Allocating mem"<<std::endl;}
		DistMatrix<double> A(*g);
		DistMatrix<double> B(*g);
		auto U_os(U);
		auto L_os(L);
		//DiagonalScale(RIGHT,NORMAL,L_os,U_os);

		V.Resize(nystrom_samples,nystrom_samples);
		Fill(V,0.0);
		auto Ahalf(V);
		S.Resize(nystrom_samples,1);
		Fill(S,0.0);
		std::vector<int> full_idx(ntrain);
		int n_oth = ntrain - nystrom_samples;
		std::vector<int> oth_idx(n_oth);

		// Pick out parts of K_nm we want (load into A and B)
		//for(int i=0; i<ntrain; i++){full_idx[i]=i;}
		if(print){std::cout << "set difference"<<std::endl;}
		std::iota(full_idx.begin(),full_idx.end(),0);
		std::set_difference(full_idx.begin(),full_idx.end(),smpIdx.begin(),smpIdx.end(),oth_idx.begin());
		if(print){std::cout << "submatrix a"<<std::endl;}
		mpi::Barrier(mpi::COMM_WORLD);
		if(print){std::cout << "submatrix b"<<std::endl;}
		
		if(0){
			GetSubmatrix(K_nm,oth_idx,smpIdx,B); //K_(n-m),m //TODO mem!!!
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
			
			gKernel.Kernel(Xoth,Xsub,B);
			gKernel.SelfKernel(Xsub,A);
			Xoth.Empty();
			Xsub.Empty();
		}

		// Form A = K_mm + K_mm^-1/2 B^T B K_mm^-1/2
		mpi::Barrier(mpi::COMM_WORLD);
		if(print){std::cout << "MAke new a"<<std::endl;}
		Syrk(UPPER,ADJOINT,1.0,B, 0.0,V); // V = B^T B; empty B
		B.Empty();
		EntrywiseMap(L_os,function<double(double)>(elem_32rt));// Ahalf = K_mm^-1/2 
		DiagonalScale(RIGHT,NORMAL, L_os, U_os); // U_os = U L^-1/2 //TODO change this to Herk (half comp)
		Gemm(NORMAL,TRANSPOSE, 1.0,U_os,U, 0.0,Ahalf); // Ahalf = U_os * U^T
		B.Resize(nystrom_samples,nystrom_samples); //  B = V * Ahalf; empty V
		Fill(B,0.0);
		Symm(LEFT,UPPER, 1.0,V,Ahalf, 0.0, B); 
		V.Empty();
		Gemm(NORMAL,NORMAL, 1.0,Ahalf,B, 1.0, A); //  A = A + Ahalf * B; empty B
		B.Empty();

		if(print){std::cout << "eig"<<std::endl;}
		// Eigendecompose A into U_os L_os
		V = A;
		HermitianEig(UPPER,V,L_os,U_os,DESCENDING);
		V.Empty();

		// Form S = L_os
		if(print){std::cout << "form s"<<std::endl;}
		S = L_os;

		// Form V = A^-1/2 U_os L_os^-1/2
		if(print){std::cout << "form v"<<std::endl;}
		V.Resize(nystrom_samples,nystrom_samples);
		Fill(V,0.0);
		EntrywiseMap(L_os,function<double(double)>(elem_sqrt)); //U_os = U_os * L^-1/2
		DiagonalSolve(RIGHT,NORMAL,L_os,U_os);
		Gemm(NORMAL,NORMAL, 1.0,Ahalf,U_os, 0.0,V); // V = Ahalf * U_os

		os_flag = true;
	}
	else{
		if(mpi::WorldRank==0){std::cout<< "One shot already done" <<std::endl;}
	}

}

void NystromAlg::orthog(){
	// If we haven't done decomp, do that now
	if(!dcmp_flag){this->decomp();}

	// Check if we've already orthogonalized
	if(!orth_flag){
		
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
		GetSubmatrix(KU,s_idx,s_idx,R); //TODO Make this more efficient
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
	else{
		if(mpi::WorldRank() == 0){std::cout << "Already orthogonalized!"<<std::endl;}
	}

}

void NystromAlg::matvec(DistMatrix<double>* Xtest, DistMatrix<double,VR,STAR>& weights, DistMatrix<double,VR,STAR>& out){
	// IF we have only decomped,  out = K_(test)m U L U^T K_nm^T weights
	// IF we have orthogonalized, need to recompute K_nm
	// Either way we must compute K_(test)m
	int testpts = Xtest->Width();

	// Get submatrix according to smpIdx
	DistMatrix<double> Xsub(*g);
	GetSubmatrix(*ptrX,d_idx,smpIdx,Xsub);

	// Compute K_nm * w
	DistMatrix<double, VR, STAR> Kw(nystrom_samples,1,*g);
	Fill(Kw,0.0);
	if(orth_flag){
		// Compute K_nm since we threw it away --> HUGE MEMORY COST //TODO?
		DistMatrix<double> K_mn(*g);
		K_mn.Resize(nystrom_samples,ntrain);
		gKernel.Kernel(Xsub,*ptrX,K_mn);

		Gemv(NORMAL,1.0,K_mn,weights,1.0,Kw);
		K_mn.Empty();
	}
	else{
		Gemv(TRANSPOSE,1.0,K_nm,weights,1.0,Kw);
	}

	// Compute U L U^T * Kw
	DistMatrix<double,VR, STAR> dummy(nystrom_rank,1,*g);
	Fill(dummy,0.0);
	Gemv(TRANSPOSE,1.0,U,Kw,1.0,dummy);

	DiagonalScale(LEFT,NORMAL,L,dummy);

	Fill(Kw,0.0);
	Gemv(NORMAL,1.0,U,dummy,1.0,Kw);
	dummy.EmptyData();

	// Form kernel both decomps need
	DistMatrix<double> K_tm(*g);
	K_tm.Resize(testpts,nystrom_samples);
	gKernel.Kernel(*Xtest,Xsub,K_tm);

	// Finish by applying K_tm
	out.Resize(testpts,1);
	Fill(out,0.0);
	Gemv(NORMAL,1.0,K_tm,Kw,1.0,out);
}

void NystromAlg::os_matvec(DistMatrix<double,VR,STAR>& weights, DistMatrix<double,VR,STAR>& out){
	// Assume oneshot, so  out = K_nm V S V^T K_nm^T weights
	
	if(orth_flag){
		if(mpi::WorldRank() == 0){std::cout << "ERROR: Cannot run orthog and os, K-nm is overwritten" <<std::endl;}
		return;
	}
	if(!os_flag){
		if(mpi::WorldRank() == 0){std::cout << "Need to run one shot before multiply " <<std::endl;}
		this->oneshot();
	}	

	DistMatrix<double, VR, STAR> Kw(K_nm.Width(),1,*g);
	Fill(Kw,0.0);
	Gemv(TRANSPOSE,1.0,K_nm,weights,1.0,Kw);

	DistMatrix<double,VR, STAR> dummy(V.Width(),1,*g);
	Fill(dummy,0.0);

	Gemv(TRANSPOSE,1.0,V,Kw,1.0,dummy);

	DiagonalScale(LEFT,NORMAL,S,dummy);

	Fill(Kw,0.0);
	Gemv(NORMAL,1.0,V,dummy,1.0,Kw);
	dummy.Empty();

	// Set up output vector properly
	out.Resize(ntrain,1);
	Fill(out,0.0);

	// Finish by applying K_nm
	Gemv(NORMAL,1.0,K_nm,Kw,1.0,out);
}

void NystromAlg::matvec(DistMatrix<double,VR,STAR>& weights, DistMatrix<double,VR,STAR>& out){
	// IF we have orthogonalized, out = K_nm L K_nm^T weights
	// IF we have only decomped,  out = K_nm U L U^T K_nm^T weights
	// Either way, can do Kw = K_nm^T * w
	
	DistMatrix<double, VR, STAR> Kw(K_nm.Width(),1,*g);
	Fill(Kw,0.0);
	Gemv(TRANSPOSE,1.0,K_nm,weights,1.0,Kw);


	// Apply either just L, or K_mm = U L U^T, store output in Kw
	if(!orth_flag){
		DistMatrix<double,VR, STAR> dummy(nystrom_rank,1,*g);
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

void NystromAlg::appinv(DistMatrix<double,VR,STAR>& rhs, DistMatrix<double,VR,STAR>& x,bool method){
	// Make sure it is orthogonalized
	if(method){
		if(!orth_flag){
			if(mpi::WorldRank() == 0){
				std::cout << "Need to orthogonalize first .." << std::endl;
			}
			this->orthog();
		}
	}else{
		if(!os_flag){
			if(mpi::WorldRank() == 0){
				std::cout << "Need to run oneshot first .." << std::endl;
			}
			this->oneshot();
		}
	}
	bool print = mpi::WorldRank() == -1;

	// Kapprox = K_nm D K_nm^T, so just need to invert diag
	// since K_nm is orthogonal
	if(method){
		DistMatrix<double,VR,STAR> Kw(nystrom_rank,1,*g);
		Fill(Kw,0.0);
		Gemv(TRANSPOSE, 1.0,K_nm,rhs, 1.0,Kw);

		// Scale by inv diag
		DiagonalSolve(LEFT,NORMAL,D,Kw);

		// Finish multiply, load into x
		x.Resize(ntrain,1);
		Fill(x,0.0);
		Gemv(NORMAL, 1.0,K_nm,Kw, 1.0,x);

		// Free the dummy vector
		Kw.Empty();
	}
	else{
		// Do K_nm^T
		if(print){std::cout << "here" << std::endl;}
		DistMatrix<double, VR, STAR> Kw(K_nm.Width(),1,*g);
		Fill(Kw,0.0);
		Gemv(TRANSPOSE,1.0,K_nm,rhs,1.0,Kw);

		// Do V^T
		if(print){std::cout << "Vt mult" << std::endl;}
		DistMatrix<double,VR, STAR> dummy(V.Width(),1,*g);
		Fill(dummy,0.0);
		Gemv(TRANSPOSE,1.0,V,Kw,1.0,dummy);

		// Solve S
		if(print){std::cout << "S solve" << std::endl;}
		DiagonalSolve(LEFT,NORMAL,S,dummy);

		// DO V
		if(print){std::cout << "V mult" << std::endl;}
		Fill(Kw,0.0);
		Gemv(NORMAL,1.0,V,dummy,1.0,Kw);
		dummy.Empty();

		// Set up output vector properly
		if(print){std::cout << "K_nm mult" << std::endl;}
		x.Resize(ntrain,1);
		Fill(x,0.0);

		// Finish by applying K_nm
		Gemv(NORMAL,1.0,K_nm,Kw,1.0,x);
	}

}

void NystromAlg::matvec_errors(std::vector<int> testIdx,int runs,double& avg_err,double& avg_time,bool method){
	// Initialize all the stuff we need
	double tot_err = 0.0;
	double tot_time = 0.0;
	int testSize = testIdx.size();
	DistMatrix<double,VR,STAR> vec(*g);
	DistMatrix<double,VR,STAR> err(ntrain,1,*g);
	DistMatrix<double,VR,STAR> err_sub(testSize,1,*g);

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
		if(method){
			this->matvec(vec,err);
		}else{
			this->os_matvec(vec,err);
		}
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

void NystromAlg::regress_test(DistMatrix<double>* Xtest,DistMatrix<double,VR,STAR>* Ytest,std::vector<int> testIdx,double& class_corr,double& reg_err, bool exact,bool method){
	if(method){
		if(!orth_flag){
			if(mpi::WorldRank()==0){std::cout << "Orthogonalizing first ..." <<std::endl;}
			this->orthog();
		}
	}else{
		if(!os_flag){
			if(mpi::WorldRank()==0){std::cout << "Running one shot first ..." <<std::endl;}
			this->oneshot();
		}
		if(!exact){
			if(mpi::WorldRank()==0){std::cout << "Cannot run oneshot and !exact yet!!" <<std::endl;}
			return;
		}
	}
	// Take subset
	int testpts = testIdx.size();
	DistMatrix<double> Xtsub(*g);
	DistMatrix<double,VR,STAR> Ytsub(*g);
	GetSubmatrix(*Xtest,d_idx,testIdx,Xtsub);
	GetSubmatrix(*Ytest,testIdx,dummy_idx,Ytsub);

	// Find weights
	DistMatrix<double,VR,STAR> weight_vec(*g);
	this->appinv(*ptrY,weight_vec,method);
	DistMatrix<double,VR,STAR> Yguess(*g);
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

void NystromAlg::calc_errors(DistMatrix<double,VR,STAR>& Ytest, DistMatrix<double,VR,STAR>& Yguess, double& class_corr, double& reg_err){
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
