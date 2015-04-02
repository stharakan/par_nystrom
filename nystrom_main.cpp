
#include "nystrom_alg.hpp"
#include <string>

using namespace El;
using std::string;

NystromAlg::NystromAlg(DistMatrix<double>* _refData, KernelInputs& _kernel_inputs, NystromInputs& _nystrom_inputs,Grid* _g, GaussKernel _gKernel):
	refData(_refData),
	kernel_inputs(_kernel_inputs),
	nystrom_inputs(_nystrom_inputs),
	g(_g),
	gKernel(_gKernel)
{
	ptrX = refData;
	ptrY = refData;
	dim             = refData->Height();
	ntrain          = refData->Width(); 
	dcmp_flag       = false;
	orth_flag       = false;
	nystrom_rank    = nystrom_inputs.rank;
	nystrom_samples = nystrom_inputs.samples;

	L.SetGrid(*g);
	U.SetGrid(*g);
	K_nm.SetGrid(*g);
	
	l_idx.resize(nystrom_samples);
	s_idx.resize(nystrom_rank);
	d_idx.resize(dim);
	dummy_idx.resize(1);
	dummy_idx[0] = 0;
	for(int i=0;i<nystrom_samples;i++){ l_idx[i] = i;} //TODO omp this loop
	for(int i=0;i<nystrom_rank;i++){ s_idx[i] = i;} //TODO omp this loop
	for(int i=0;i<dim;i++){ d_idx[i] = i;} //TODO omp this loop

	// Allocate memory or at least try to
	try{
		L.Resize(nystrom_rank,1);
		U.Resize(nystrom_samples,nystrom_rank);
		K_nm.Resize(ntrain,nystrom_samples);
	}
	catch(exception& e){ ReportException(e); }
};

NystromAlg::~NystromAlg(){
	// Matrices
	L.Empty();
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
				//std::cout << "Sample idx" << std::endl;
				for (int i=0;i<nystrom_samples;i++){
					smpIdx[i] = _smpIdx[i];
					//std::cout << _smpIdx[i] << std::endl;
				}
				_smpIdx.clear();
			}

			//Send vector to everybody else
			mpi::Broadcast(&smpIdx[0], nystrom_samples, 0, mpi::COMM_WORLD);
			
			// Sample from data
			DistMatrix<double> Xsub(*g);
			GetSubmatrix(*ptrX,d_idx,smpIdx,Xsub); 
			//Print(Xsub,"X_mn");	
			
			// Fill K_mm with kernel values 
			DistMatrix<double> K_mm(nystrom_samples,nystrom_samples,*g);
			gKernel.SelfKernel(Xsub, K_mm);
			//Print(K_mm,"K_mm");
			
			// Take Eigendecomp of subsampled matrix
			auto mmCopy(K_mm);
			DistMatrix<double> Umm(*g);
			DistMatrix<double> Lmm(*g);
			HermitianEig(UPPER,mmCopy,Lmm,Umm,DESCENDING);
			mmCopy.Empty();

			// Truncate
			DiagonalSolve(RIGHT,NORMAL,Lmm,Umm);
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
		
		//Form the large U (KU below)
		DistMatrix<double> KU(ntrain,nystrom_rank,*g);
		Fill(KU,0.0);
		Gemm(NORMAL,NORMAL,1.0,K_nm,U,1.0,KU);
		K_nm.EmptyData(); //Don't need this until later

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
		Fill(Kw,0.0);

		DiagonalScale(LEFT,NORMAL,L,dummy);

		Gemv(NORMAL,1.0,U,dummy,1.0,Kw);
		dummy.EmptyData();
	}
	else{
		DiagonalScale(LEFT,NORMAL,L,Kw);
	}
	
	// Set up output vector properly
	out.Resize(ntrain,1);
	Fill(out,0.0);

	// Finish by applying K_nm
	Gemv(NORMAL,1.0,K_nm,Kw,1.0,out);
}

void NystromAlg::appinv(DistMatrix<double,VR,STAR>& rhs, DistMatrix<double,VR,STAR>& x){
	// Make sure it is orthogonalized
	if(!orth_flag){
		if(mpi::WorldRank() == 0){
			std::cout << "Need to orthogonalize first .." << std::endl;
		}
		this->orthog();
	}
	
	// Kapprox = K_nm L K_nm^T, so just need to invert diag
	// since K_nm is orthogonal
	DistMatrix<double,VR,STAR> Kw(nystrom_rank,1,*g);
	Fill(Kw,0.0);
	Gemv(TRANSPOSE, 1.0,K_nm,rhs, 1.0,Kw);

	// Scale by inv diag
	DiagonalSolve(LEFT,NORMAL,L,Kw);

	// Finish multiply, load into x
	x.Resize(ntrain,1);
	Fill(x,0.0);
	Gemv(NORMAL, 1.0,K_nm,Kw, 1.0,x);
	
	// Free the dummy vector
	Kw.Empty();
}

void NystromAlg::matvec_errors(std::vector<int> testIdx,int runs,double& avg_err,double& avg_time){
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
		this->matvec(vec,err);
		tot_time += mpi::Time() - start;
		GetSubmatrix(err,testIdx,dummy_idx,err_sub);
		
		// Find exact kernel-vec, subtract from ans
		//if (mpi::WorldRank() == 0) {std::cout << "Exact matvec" <<std::endl;}
		Gemv(NORMAL, -1.0,K,vec, 1.0,err_sub);
		double abs_err = FrobeniusNorm(err_sub);
	
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

int main(int argc, char* argv []){
	Initialize(argc,argv);
	
	int mpicomms = mpi::Size(mpi::COMM_WORLD);
	if (mpi::WorldRank() == 0){
		std::cout << "Creating grid on " << mpicomms << " mpi tasks"<<std::endl;
	}
	Grid grid(mpi::COMM_WORLD);

	// ----- INPUTS  ------ //
	// Training data (need this)
	const string datadir   = Input<string>("--dir","data directory" );
	const string trdataloc = Input<string>("--trdata","training data file");
	const string trlabloc  = Input<string>("--trlabs","training labels file");
	const Int ntrain       = Input<Int>("--ntrain","# of training pts");
	const Int dim          = Input<Int>("--dim","dimension of data");

	// Kernel param
	const double sigma     = Input("--sigma","kernel bandwidth", 1.0);

	// Nystrom params
	int nyst_rank          = Input("--rank","Nystrom rank",min(1024,ntrain));
	int nyst_samp          = Input("--samp","Nystrom rank",min(nyst_rank*2,ntrain));
	
	// Error comp
	const int test_pts     = Input("--testpts","# of testing points",min(1000, ntrain));

	// Test data (can be null)
	const string tedataloc = Input("--tedata","test data file","");
	const string telabloc  = Input("--telabs","test labels file","");
	const Int ntest        = Input("--ntest","# of training pts",0);
	
	// Finish up with inputs
	ProcessInput();
	PrintInputReport();

	// Get proc id in case we need it later
	const int proc = mpi::WorldRank();

	// ---- COMPUTATION ---- ///
	try{
		//std::cout << "Loading data" <<std::endl;
		// Read data
		DistMatrix<double> Xtrain(dim,ntrain,grid);
		string trdata = datadir;
		trdata.append(trdataloc);
		Read(Xtrain,trdata,BINARY_FLAT);

		// Read labels
		DistMatrix<double,VR,STAR> Ytrain(ntrain,1,grid);
		string trlab = datadir;
		trlab.append(trlabloc);
		Read(Ytrain,trlab,BINARY_FLAT);

		// Do we need to load test data?
		if(tedataloc.compare("") != 0 && telabloc.compare("") != 0){
			// Read data
			DistMatrix<double> Xtest(dim,ntest,grid); 
			string tedata = datadir;
			tedata.append(tedataloc);
			Read(Xtest,tedata,BINARY_FLAT);

			// Read labels
			DistMatrix<double,VR,STAR> Ytest(ntest,1,grid);
			string telab = datadir;
			telab.append(telabloc);
			Read(Ytest,telab,BINARY_FLAT);
		}
		
		
		//std::cout << "Loading kernel params" <<std::endl;
		KernelInputs kernel_inputs;
		kernel_inputs.bandwidth = sigma;

		//std::cout <<"Loading nystrom params" <<std::endl;
		NystromInputs nystrom_inputs(nyst_rank,nyst_samp);
  
		mpi::Barrier(mpi::COMM_WORLD);
		//std::cout << "Making kernel class" <<std::endl;
		GaussKernel gKern(kernel_inputs, &grid);

		//std::cout << "Initializing NystromAlg obj" <<std::endl;
		NystromAlg nyst(&Xtrain,kernel_inputs,nystrom_inputs,&grid, gKern);

		//std::cout << "Running decomp" <<std::endl;
		nyst.decomp();

		//std::cout << "Running orthog" <<std::endl;
		nyst.orthog();

		DistMatrix<double,VR,STAR> test_vec(grid);
		Uniform(test_vec,ntrain,1);
		DistMatrix<double,VR,STAR> ans(grid);
		
		//std::cout << "Testing multiply" <<std::endl;
		nyst.matvec(test_vec,ans);

		//std::cout << "Testing appinv" << std::endl;
		ans.Empty();
		nyst.appinv(test_vec,ans);

		// Kernel testing
		/*
		DistMatrix<double> A(10,10,grid);
		DistMatrix<double> K(10,10,grid);
		std::vector<int> tot_idx(10);
		for(int i=0;i<10;i++){
			tot_idx[i] = i;
			for(int j=0;j<10;j++){
				A.Set(i,j,(double) (i*j));
			}
		}
		
		gKern.SelfKernel(A,K);
		Print(K);
		std::vector<int> curr_idx = {0,3,6};
		DistMatrix<double> Asub(10,3,grid);
		DistMatrix<double> Ksub(3,10,grid);
		GetSubmatrix(A,tot_idx,curr_idx,Asub);
		gKern.Kernel(Asub,A,Ksub);
		Print(Ksub);	
		*/	
		
		double avg_err;
		double avg_time;
		// Form testIdx
		std::vector<int> testIdx(test_pts);
		double step = ((double)ntrain)/((double) test_pts);
		for(int i=0;i<test_pts;i++){
			int currIdx = (int)(i * step);
			testIdx[i] = currIdx;
		}
		

		nyst.matvec_errors(testIdx,10,avg_err,avg_time);
		if(mpi::WorldRank() == 0){std::cout << "Relative Err: " << avg_err << std::endl;}
	
	}
	catch(exception& e){ ReportException(e); }
	Finalize();

	return 0;
}

