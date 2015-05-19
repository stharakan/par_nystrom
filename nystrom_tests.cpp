
#include "nystrom_alg.hpp"
//#include "nystrom_utils.hpp"
#include <string>
#include <vector>

using namespace El;
using std::string;

/*
 * Tests orthog by checking the value of L*L-1
 */
double test_nystl(NystromAlg& nyst, double & cond){
	const Grid& g = nyst.K_nm.Grid();
	int height = nyst.L.Height();

	// Get conditioning
	double sig1 = nyst.L.Get(0,0);
	double sigh = nyst.L.Get(height-1,0);
	//if(mpi::WorldRank() ==0){std::cout<< sig1<<" vs "<<sigh<<std::endl;}
	cond = sig1/sigh;

	// Test action instead of L L ^-1

	// Make eye
	/*
	DistMatrix<double,VR,STAR> ones(height,1,g);
	Fill(ones,1.0);

	// Do multiply
	auto testL(nyst.L);
	DiagonalSolve(LEFT,NORMAL,nyst.L,testL);

	// Get norm
	Axpy(-1.0,ones,testL);
	double d_err = FrobeniusNorm(testL);
	return d_err/sqrt(height);
	*/

	// Test action instead of L L ^-1
	DistMatrix<double,VR,STAR> vec(g);
	Uniform(vec,height,1);
	auto ref(vec);
	double base_norm = FrobeniusNorm(vec);
	DiagonalSolve(LEFT,NORMAL,nyst.L,vec);
	DiagonalScale(LEFT,NORMAL,nyst.L,vec);
	Axpy(-1.0,vec,ref);
	double err = FrobeniusNorm(ref);
	return err/base_norm;

}

/*
 * Tests nyst by checking if U^T U = I
 * should be called before orthogonalization 
 */
double test_nystu(NystromAlg& nyst){
	const Grid& g = nyst.U.Grid();

	// Make copy
	auto V(nyst.U);
	auto S(nyst.L);
	int w = V.Width();
	DiagonalScale(RIGHT,NORMAL,S,V);

	// Make identity and multiply
	DistMatrix<double> I(g);
	Identity(I,w,w);

	// Do multiply and subtract
	Herk(UPPER,ADJOINT,1.0,V, -1.0,I);

	// Get norm
	double err = HermitianFrobeniusNorm(UPPER,I);
	return err/sqrt(w);
}

/*
 * Tests nyst by checking if ULU^T*K=I
 * should be called before orthogonalization 
 */
double test_nysteig(NystromAlg& nyst){
	const Grid& g = nyst.U.Grid();

	// Make copy
	auto V(nyst.U);
	auto S(nyst.L);
	int h = V.Height();

	// Make kernel matrix
	DistMatrix<double> K(h,h,g);
	DistMatrix<double> Xsub(g);
	GetSubmatrix(*(nyst.ptrX),nyst.get_d(),nyst.get_smp(),Xsub);
	nyst.gKernel.SelfKernel(Xsub,K);
	auto Kstar(K);
	double base_norm = HermitianFrobeniusNorm(UPPER,K);

	// Do multiply and subtract
	DiagonalScale(RIGHT,NORMAL,S,V);
	auto elem_sqrt = [](double x){return (sqrt(x));};
	EntrywiseMap(S,function<double(double)>(elem_sqrt));
	DiagonalScale(RIGHT,NORMAL,S,V);
	Herk(UPPER,NORMAL,1.0,V, -1.0,K);

	// Make identity and multiply
	//DistMatrix<double> I(g);
	//Identity(I,h,h);
	//Gemm(NORMAL,NORMAL,1.0,Kstar,K, -1.0,I);

	// Get norm
	double err = HermitianFrobeniusNorm(UPPER,K);
	return err/base_norm;
}

/*
 * Tests orthog by checking the value of D*D-1
 */
double test_orthd(NystromAlg& nyst,double & cond){
	const Grid& g = nyst.K_nm.Grid();

	// Make eye
	int height = nyst.D.Height();
	DistMatrix<double,VR,STAR> ones(height,1,g);
	Fill(ones,1.0);

	// Do multiply
	auto testD(nyst.D);
	DiagonalSolve(LEFT,NORMAL,nyst.D,testD);
	
	double sig1 = nyst.D.Get(0,0);
	double sigh = nyst.D.Get(height-1,0);
	//if(mpi::WorldRank() ==0){std::cout<< sig1<<" vs "<<sigh<<std::endl;}
	cond = sig1/sigh;
	
	// Test on a vec
	//DistMatrix<double,VR,STAR> test_vec(g);
	//Uniform(test_vec,height,1);
	//auto test2(test_vec);
	//DiagonalScale(LEFT,NORMAL,testD,test_vec);
	//Axpy(-1.0,test_vec,test2);
	//double d_err_vec = FrobeniusNorm(test2)/FrobeniusNorm(test_vec);

	// Get norm
	Axpy(-1.0,ones,testD);
	double d_err = FrobeniusNorm(testD);

	return d_err/sqrt(height);
}

/*
 * Tests orthog by seeing how orthogonal the matrices are.
 * Computes K_nm' * K_nm and compares it to the identity
 */
double test_ortheye(NystromAlg& nyst){
	const Grid& g = nyst.K_nm.Grid();

	DistMatrix<double> I(g);
	int width;

	width = nyst.K_nm.Width();
	Identity(I,width,width);
	Herk(UPPER,ADJOINT,1.0,nyst.K_nm, -1.0,I);

	// Get norm
	double eye_err = HermitianFrobeniusNorm(UPPER,I);
	return eye_err/sqrt(width);
}

/**
 * Tests orthog by applying a weight vector to the multiply with 
 * and without orthog. Since these are both approximations
 * the result should be the same
 */
double test_orthmv(NystromAlg& nyst,double& max_regress_time){
	// Initialize and orthogonalize
	const Grid& g = nyst.K_nm.Grid();
	DistMatrix<double,VR,STAR> vec1(g);
	DistMatrix<double,VR,STAR> out1(g);
	DistMatrix<double,VR,STAR> out2(g);
	Uniform(vec1,nyst.K_nm.Height(),1);
	
	// Do matvec (no orth)
	nyst.matvec(vec1,out1);

	// Do matvec (orth)
	double r_time,max_r_time;
	double start = mpi::Time();
	nyst.orthog();
	r_time = mpi::Time() - start;
	mpi::Reduce(&r_time,&max_r_time,1,mpi::MAX,0,mpi::COMM_WORLD);
	max_regress_time=max_r_time;
	nyst.matvec(vec1,out2);
	auto dummy_vec(out2);

	// Get norm
	Axpy(-1.0,out1,dummy_vec);
	double err = FrobeniusNorm(dummy_vec);
	double base_norm = FrobeniusNorm(out1);

	return err/base_norm;
}

/**
 * Tests app_inv by applying a weight vector to the matrix 
 * and then applying app_inv. Since these are both approximations
 * the result should be the same as the input
 */
double test_appinv(NystromAlg& nyst,int r = 0){
	// Initialize
	const Grid& g = nyst.K_nm.Grid();
	DistMatrix<double,VR,STAR> x(g);
	DistMatrix<double,VR,STAR> xp(g);
	DistMatrix<double,VR,STAR> y(g);

	// Put dummy_vec in correct space (now in x)
	int height = nyst.K_nm.Height();
	int width = nyst.K_nm.Width();
	Uniform(xp,width,1);
	x.Resize(height,1);
	Fill(x,0.0);
	Gemv(NORMAL, 1.0,nyst.K_nm,xp, 1.0,x);

	// Apply forward multiply into y
	nyst.matvec(x,y);
	xp.Resize(height,1);

	// Apply inverse into xp
	nyst.appinv(y,xp,r);
	double test = FrobeniusNorm(xp);

	// Get norm
	Axpy(-1.0,x,xp);
	double err = FrobeniusNorm(xp);
	double base_norm = FrobeniusNorm(x);


	return err/base_norm;
}

/**
 * Tests null_space by applying a weight vector to the matrix 
 * and then applying null_space. Since the new vec is in the space,
 * null_space should return ~0 for all the spectrum
 */
double test_nullspace(NystromAlg& nyst,int r = 0){
	// Initialize
	const Grid& g = nyst.K_nm.Grid();
	DistMatrix<double,VR,STAR> x(g);
	DistMatrix<double,VR,STAR> xp(g);
	DistMatrix<double,VR,STAR> y(g);

	// Put dummy_vec in correct space (now in x)
	int height = nyst.K_nm.Height();
	int width = nyst.K_nm.Width();
	Uniform(xp,width,1);
	x.Resize(height,1);
	Fill(x,0.0);
	Gemv(NORMAL, 1.0,nyst.K_nm,xp, 1.0,x);

	// Apply nullspace to xp with whole spectrum
	double err  = nyst.nullspace(x,xp,r);
	return err;
}

/**
 * Makes the test index set and returns it in testIdx
 * User should initialize testIdx to the appropriate length,
 * generally 1000
 */
void make_testIdx(std::vector<int>& testIdx, int ntrain){

	int test_pts =  testIdx.size();
	double step = ((double)ntrain)/((double) test_pts);

	for(int i=0;i<test_pts;i++){
		int currIdx = (int)(i * step);
		testIdx[i] = currIdx;
	}
}


int main(int argc, char* argv []){
	// Initialize mpi
	Initialize(argc,argv);

	int mpicomms = mpi::Size(mpi::COMM_WORLD);
	if (mpi::WorldRank() == 0){
		std::cout << "Creating grid on " << mpicomms << " mpi tasks"<<std::endl;
	}
	const Grid grid(mpi::COMM_WORLD);
	
	//////////////////////////////
	// ---- Process inputs ---- //
	//////////////////////////////

	// Training data (need this)
	const string datadir   = Input<string>("--dir","data directory" );
	const string trdataloc = Input<string>("--trdata","training data file");
	const string trlabloc  = Input<string>("--trlabs","training labels file","");
	const Int ntrain       = Input<Int>("--ntrain","# of training pts");
	const Int dim          = Input<Int>("--dim","dimension of data");

	// Kernel param
	const double sigma     = Input("--sigma","kernel bandwidth", 1.0);

	// Nystrom params
	const int nyst_rank          = Input("--rank","Nystrom rank",min(1024,ntrain));
	const int nyst_samp          = Input("--samp","Nystrom samp",min(2*nyst_rank,ntrain));

	// Error comp
	const int test_pts     = Input("--testpts","# of testing points",min(1000, ntrain));

	// Test data (can be null)
	const string tedataloc = Input("--tedata","test data file","");
	const string telabloc  = Input("--telabs","test labels file","");
	const Int ntest        = Input("--ntest","# of training pts",1000);

	// Set flags to do things if we want
	const bool regression  = Input("--rr","do regression?",false);
	const bool do_exact    = Input("--ex","compute reg mv exactly?",true);
	const bool do_rtests   = Input("--tr","do regression tests?",false);
	const bool do_ntests   = Input("--tn","do nystrom tests?",false);
	const bool do_qr       = Input("--qr","do one shot tests?",false);

	// Finish up with inputs
	ProcessInput();
	PrintInputReport();

	// Get proc id in case we need it later, initialize
	const int proc = mpi::WorldRank();
	double start;
	bool test_data = false;
	bool trn_labs = false;
	std::vector<int> testIdx(test_pts);
	//////////////////////////////


	//////////////////////////////
	// ---- Read train set ---- //
	//////////////////////////////
	mpi::Barrier(mpi::COMM_WORLD);
	start = mpi::Time();
	DistMatrix<double> Xtrain(dim,ntrain,grid);
	string trdata = datadir;
	trdata.append(trdataloc);
	Read(Xtrain,trdata,BINARY_FLAT);
/*
	std::vector<Int> smpIdx(nyst_samp);
	std::vector<Int> d_idx(dim);
	for(int i=0;i<nyst_samp;i++){smpIdx[i]=i;}
	for(int i=0;i<dim;i++){d_idx[i]=i;}
	DistMatrix<double> Xsub(grid);
	GetSubmatrix(Xtrain,d_idx,smpIdx,Xsub); 
	Print(Xsub,"X_sub -- read");	
*/
	// Read labels
	DistMatrix<double,VR,STAR> Ytrain(grid);
	if(trlabloc.compare("") != 0){
		Ytrain.Resize(ntrain,1);
		string trlab = datadir;
		trlab.append(trlabloc);
		Read(Ytrain,trlab,BINARY_FLAT);
		trn_labs = true;
	}

	// Get timing
	double train_read_time = mpi::Time() - start;
	double max_train_read_time;
	mpi::Reduce(&train_read_time,&max_train_read_time,1,mpi::MAX,0,mpi::COMM_WORLD);
	if(proc==0){std::cout << "Train data read time : " << max_train_read_time <<std::endl;}
	//////////////////////////////


	//////////////////////////////
	// ---- Read test data ---- //
	//////////////////////////////
	mpi::Barrier(mpi::COMM_WORLD);
	DistMatrix<double> Xtest(grid);
	DistMatrix<double,VR,STAR> Ytest(grid);
	if(tedataloc.compare("") != 0 && telabloc.compare("") != 0){
		// Set test_data to true
		test_data = true;
		double test_read_time;

		// Read data
		start = mpi::Time();
		Xtest.Resize(dim,ntest); 
		string tedata = datadir;
		tedata.append(tedataloc);
		Read(Xtest,tedata,BINARY_FLAT);

		// Read labels
		Ytest.Resize(ntest,1);
		string telab = datadir;
		telab.append(telabloc);
		Read(Ytest,telab,BINARY_FLAT);
		test_read_time = mpi::Time() - start;
	
		// Get timing
		double max_test_read_time;
		mpi::Reduce(&test_read_time,&max_test_read_time,1,mpi::MAX,0,mpi::COMM_WORLD);
		if(proc==0){std::cout << "Test data read time  : " << max_test_read_time <<std::endl;}
	}
	else{
		if(regression){
			if(proc==0){std::cout << "Error: no test data to run regression" << std::endl;}
			return -1;
		}
	}
	//////////////////////////////
	

	//////////////////////////////
	// ---- Initialization ---- //
	//////////////////////////////
	mpi::Barrier(mpi::COMM_WORLD);
	start = mpi::Time();

	// Kernel inputs
	KernelInputs kernel_inputs;
	kernel_inputs.bandwidth = sigma;

	// Nystrom inputs
	NystromInputs nystrom_inputs(nyst_rank,nyst_samp);

	// Actual kernel
	GaussKernel gKern(sigma, &grid);
	
	// Initialize alg
	//NystromAlg nyst(&Xtrain,nyst_samp,nyst_rank,gKern,&Ytrain);
	NystromAlg nyst(&Xtrain,sigma,nyst_samp,nyst_rank,&Ytrain);
	double init_time = mpi::Time() - start;

	// Test qr to compare
	NystromAlg nyst_qr(&Xtrain,sigma,nyst_samp,nyst_rank,&Ytrain);
	nyst_qr.force_qr();
	
	// Get timing
	double max_init_time;
	mpi::Reduce(&init_time,&max_init_time,1,mpi::MAX,0,mpi::COMM_WORLD);
	if(proc==0){std::cout << "Initialization time  : " << max_init_time <<std::endl;}
	//////////////////////////////


	//////////////////////////////
	// ----- Decomposition ---- //
	//////////////////////////////
	mpi::Barrier(mpi::COMM_WORLD);
	start = mpi::Time();	
	nyst.decomp();
	double decomp_time = mpi::Time() - start;

	// Get timing 
	double max_decomp_time;
	mpi::Reduce(&decomp_time,&max_decomp_time,1,mpi::MAX,0,mpi::COMM_WORLD);
	if(proc==0){std::cout << "Decomposition time   : " << max_decomp_time <<std::endl;}
	//////////////////////////////


	//////////////////////////////
	// ----- Matvec tests ----- //
	//////////////////////////////
	mpi::Barrier(mpi::COMM_WORLD);
	double avg_mv_err;
	double avg_mv_time;
	make_testIdx(testIdx,ntrain);
	nyst.matvec_errors(testIdx,10,avg_mv_err,avg_mv_time);

	// Get timing
	double max_avg_mv_time;
	mpi::Reduce(&avg_mv_time,&max_avg_mv_time,1,mpi::MAX,0,mpi::COMM_WORLD);
	if(proc==0){std::cout << "Matvec time          : " << max_avg_mv_time <<std::endl;}
	if(proc==0){std::cout << "Relative error       : " << avg_mv_err << std::endl;}

	//////////////////////////////
	
	
	//////////////////////////////
	// ----- Decomp tests ----- //
	//////////////////////////////
	if(do_ntests){
		if(proc==0){std::cout << std::endl << "Running nyst tests ... " <<std::endl;}

		// Test for conditioning of diag L
		double cond;
		double nyst_ll_err = test_nystl(nyst, cond);
		if(proc==0){std::cout << "Error from nyst ll^-1: " << nyst_ll_err <<std::endl;}
		if(proc==0){std::cout << "Conditioning of L    : " << cond <<std::endl;}
		
		// Test viability of eigendecomp in nyst
		double nyst_utu_err = test_nystu(nyst);
		if(proc==0){std::cout << "Error from nyst uTu  : " << nyst_utu_err <<std::endl;}
		
		// Test viability of eigendecomp in nyst
		double nyst_eig_err = test_nysteig(nyst);
		if(proc==0){std::cout << "Error from nyst eig  : " << nyst_eig_err <<std::endl;}
	}
	//////////////////////////////


	//////////////////////////////
	// ----- Oneshot tests ---- //
	//////////////////////////////
	// Store things to test against later: w_os
	DistMatrix<double,VR,STAR> w_os(ntest,1,grid);
	if(do_rtests){
		// Run oneshot
		if(proc==0){std::cout << std::endl <<"Running one shot ... " <<std::endl;}
		
		// Test multiply before and after orth
		double os_time;
		double orth_mv_err = test_orthmv(nyst,os_time);
		if(proc==0){std::cout << "1shot orth time      : " << os_time <<std::endl;}
		if(proc==0){std::cout << "1shot vs decomp err  : " << orth_mv_err <<std::endl;}

		// Test mv errors
		double avg_osmv_err,avg_osmv_time;
		nyst.matvec_errors(testIdx,10,avg_osmv_err,avg_osmv_time);

	  // Get timing
	  double max_avg_osmv_time;
	  mpi::Reduce(&avg_osmv_time,&max_avg_osmv_time,1,mpi::MAX,0,mpi::COMM_WORLD);
	  if(proc==0){std::cout << "1shot matvec time    : " << max_avg_osmv_time <<std::endl;}
	  if(proc==0){std::cout << "1shot rel error      : " << avg_osmv_err << std::endl;}

		// Test orthogonality
		double orth_err = test_ortheye(nyst);
		if(proc==0){std::cout << "How close to orth?   : " << orth_err << std::endl;}
		
		// Test for conditioning of diag D
		double cond;
		double diag_err = test_orthd(nyst,cond);
		if(proc==0){std::cout << "Conditioning of D    : " << cond << std::endl;}
		
		// Test appinv
		double inv_err = test_appinv(nyst);
		if(proc==0){std::cout << "Appinv err (1)       : " << inv_err <<std::endl;}
		inv_err = test_appinv(nyst,nyst_samp/2);
		if(proc==0){std::cout << "Appinv err (0.5)     : " << inv_err <<std::endl;}
		inv_err = test_appinv(nyst,nyst_samp/4);
		if(proc==0){std::cout << "Appinv err (0.25)    : " << inv_err <<std::endl;}

		// Get w_os if regresion
		if(regression){
			// Pick out subset we will test on of both Xtest, Ytest
			make_testIdx(testIdx,ntest);
			
			// get w_os
			nyst.appinv(Ytrain,w_os);
		
			// test regression errors
			double class_corr,err_l2;
			nyst.regress_test(&Xtest,&Ytest,testIdx,class_corr,err_l2,do_exact);
			if(proc==0){std::cout << "L2 error (1shot)     : " << err_l2 <<std::endl;}
			if(proc==0){std::cout << "Class corr (1shot)   : " << class_corr <<std::endl;}
		}
	}
	
	//////////////////////////////
	
	//////////////////////////////
	// ------- Qr tests ------- //
	//////////////////////////////
	DistMatrix<double,VR,STAR> w_qr(ntest,1,grid);
	double orthog_time = 0.0;
	if(do_qr){
		if(proc==0){std::cout << std::endl<<"Running qr tests ... " <<std::endl;}
		
		// Do decomp
		nyst_qr.set_smp(nyst.get_smp());
		nyst_qr.decomp();	

		// Test multiply before and after orth
		double orth_mv_err = test_orthmv(nyst_qr,orthog_time);
		if(proc==0){std::cout << "Qr orth time         : " << orthog_time <<std::endl;}
		if(proc==0){std::cout << "Qr vs decomp err     : " << orth_mv_err <<std::endl;}
		
		// Test mv errors
		double avg_qrmv_err,avg_qrmv_time;
		nyst_qr.matvec_errors(testIdx,10,avg_qrmv_err,avg_qrmv_time);

	  // Get timing
	  double max_avg_qrmv_time;
	  mpi::Reduce(&avg_qrmv_time,&max_avg_qrmv_time,1,mpi::MAX,0,mpi::COMM_WORLD);
	  if(proc==0){std::cout << "Qr matvec time       : " << max_avg_qrmv_time <<std::endl;}
	  if(proc==0){std::cout << "Qr rel error         : " << avg_qrmv_err << std::endl;}
		
		// Test for orthogonality of k_nm
		double orth_err = test_ortheye(nyst_qr);
		if(proc==0){std::cout << "How close to orth?   : " << orth_err << std::endl;}

		// Test for conditioning of diag D
		double cond;
		double diag_err = test_orthd(nyst_qr,cond);
		if(proc==0){std::cout << "Conditioning of D    : " << cond << std::endl;}
		
		// Test appinv
		double inv_err = test_appinv(nyst_qr);
		if(proc==0){std::cout << "Appinv err (1)       : " << inv_err <<std::endl;}
		inv_err = test_appinv(nyst_qr,nyst_samp/2);
		if(proc==0){std::cout << "Appinv err (0.5)     : " << inv_err <<std::endl;}
		inv_err = test_appinv(nyst_qr,nyst_samp/4);
		if(proc==0){std::cout << "Appinv err (0.25)    : " << inv_err <<std::endl;}

		if(regression){
			// Pick out subset we will test on of both Xtest, Ytest
			make_testIdx(testIdx,ntest);
			
			// get w_os
			nyst_qr.appinv(Ytrain,w_qr);
		
			// test regression errors
			double class_corr,err_l2;
			nyst_qr.regress_test(&Xtest,&Ytest,testIdx,class_corr,err_l2,do_exact);
			if(proc==0){std::cout << "L2 error (qr)        : " << err_l2 <<std::endl;}
			if(proc==0){std::cout << "Class corr (qr)      : " << class_corr <<std::endl;}
			
			// Test relative error between w_orth,w_os
			auto err(w_qr);
			Axpy(-1.0,w_os,err);
			double abs_err = FrobeniusNorm(err);
			double err_w = abs_err/FrobeniusNorm(w_qr);

			if(proc==0){std::cout <<std::endl << "qr vs os (w l2 err): " << err_w <<std::endl;}
		}
	
	}
	//////////////////////////////
	
	// Finalize
	mpi::Barrier(mpi::COMM_WORLD);
	Finalize();
	return 0;
}

