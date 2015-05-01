
#include "nystrom_alg.hpp"
//#include "nystrom_utils.hpp"
#include <string>
#include <vector>

using namespace El;
using std::string;

/*
 * Tests orthog by checking the value of D*D-1
 */
double test_orthd(NystromAlg& nyst){
	const Grid& g = nyst.K_nm.Grid();

	// Make eye
	int height = nyst.D.Height();
	DistMatrix<double,VR,STAR> ones(height,1,g);
	Fill(ones,1.0);

	// Do multiply
	auto testD(nyst.D);
	DiagonalSolve(LEFT,NORMAL,nyst.D,testD);
	
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
	int width = nyst.K_nm.Width();

	// Make eye
	DistMatrix<double> I(g);
	Identity(I,width,width);

	// Do multiply
	Herk(UPPER,ADJOINT,1.0,nyst.K_nm, -1.0,I);

	// Get norm
	double eye_err = FrobeniusNorm(I);
	return eye_err/sqrt(width);
}

/**
 * Tests orthog by applying a weight vector to the multiply with 
 * and without orthog. Since these are both approximations
 * the result should be the same
 */
double test_orthmv(NystromAlg& nyst){
	// Initialize and orthogonalize
	const Grid& g = nyst.K_nm.Grid();
	DistMatrix<double,VR,STAR> vec1(g);
	DistMatrix<double,VR,STAR> out1(g);
	DistMatrix<double,VR,STAR> out2(g);
	Uniform(vec1,nyst.K_nm.Height(),1);
	
	// Do matvec (no orth)
	nyst.matvec(vec1,out1);

	// Do matvec (orth)
	nyst.orthog();
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
double test_appinv(NystromAlg& nyst){
	// Initialize
	const Grid& g = nyst.K_nm.Grid();
	DistMatrix<double,VR,STAR> x(g);
	DistMatrix<double,VR,STAR> xp(g);
	DistMatrix<double,VR,STAR> y(g);

	// Put dummy_vec in correct space (now in x)
	int height = nyst.K_nm.Height();
	Uniform(xp,height,1);
	x.Resize(height,1);
	Fill(x,0.0);
	Gemv(NORMAL, 1.0,nyst.K_nm,xp, 1.0,x);

	// Apply forward multiply into y
	nyst.matvec(x,y);

	// Apply inverse into xp
	nyst.appinv(y,xp);
	double test = FrobeniusNorm(xp);

	// Get norm
	Axpy(-1.0,x,xp);
	double err = FrobeniusNorm(xp);
	double base_norm = FrobeniusNorm(x);

	return err/base_norm;
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
	Grid grid(mpi::COMM_WORLD);
	
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
	const bool do_tests    = Input("--tt","do tests?",false);

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
	GaussKernel gKern(kernel_inputs, &grid);
	
	// Initialize alg
	NystromAlg nyst(&Xtrain,&Ytrain,nystrom_inputs,&grid, gKern);
	double init_time = mpi::Time() - start;

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
	if(do_tests){
		if(proc==0){std::cout << "Running tests ... " <<std::endl;}

		// Test multiply before and after orth
		double orth_mv_err = test_orthmv(nyst);
		if(proc==0){std::cout << "Error from orth mult : " << orth_mv_err <<std::endl;}

		// Test for orthogonality of k_nm
		double orth_err = test_ortheye(nyst);
		if(proc==0){std::cout << "How close to orth?   : " << orth_err << std::endl;}

		// Test for orthogonality of k_nm
		double diag_err = test_ortheye(nyst);
		if(proc==0){std::cout << "Diag^-1 Diag err     : " << diag_err << std::endl;}
		
		// Test appinv
		double inv_err = test_appinv(nyst);
		if(proc==0){std::cout << "Error from appinv    : " << inv_err <<std::endl;}

		// Put other tests here


	}
	//////////////////////////////
	
	
	//////////////////////////////
	// ------ Regression ------ //
	//////////////////////////////
	mpi::Barrier(mpi::COMM_WORLD);
	if(regression){
		double class_corr,err_l2;
		start = mpi::Time();
		// Pick out subset we will test on of both Xtest, Ytest
		make_testIdx(testIdx,ntest);

		// Orthogonalize
		nyst.orthog();
		double regress_time = mpi::Time() - start;
		
		// Run regression tests
		nyst.regress_test(&Xtest,&Ytest,testIdx,class_corr,err_l2,do_exact);
		
		// Get timing
		double max_regress_time;
		mpi::Reduce(&regress_time,&max_regress_time,1,mpi::MAX,0,mpi::COMM_WORLD);
		if(proc==0){std::cout << "Regression time      : " << max_regress_time <<std::endl;}
		if(proc==0){std::cout << "L2 error             : " << err_l2 <<std::endl;}
		if(proc==0){std::cout << "Class corr           : " << class_corr <<std::endl;}
	}
	//////////////////////////////

	// Finalize
	mpi::Barrier(mpi::COMM_WORLD);
	Finalize();
	return 0;
}

