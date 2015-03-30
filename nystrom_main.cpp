
#include "nystrom_alg.hpp"

using namespace El;

int main(int argc, char* argv []){
	Initialize(argc,argv);
	Int ntrain = 20;
	Int dim = 10;
	
	int mpicomms = mpi::Size(mpi::COMM_WORLD);
	std::cout << "Creating grid on " << mpicomms << " mpi tasks"<<std::endl;
	Grid grid(mpi::COMM_WORLD);

	try{
		//std::cout << "Initializing data" <<std::endl;
		DistMatrix<double> refData(ntrain,dim,grid);

		//std::cout << "Loading data" <<std::endl;
		//Uniform(*refData,ntrain,dim);
		for(Int i=0;i<ntrain;i++){
			for(Int j =0;j<dim;j++){
				//std::cout << i << " " << j << std::endl;
				refData.Set(i,j, (double) (i+j));
			}
		}
		
		//std::cout << "Loading kernel params" <<std::endl;
		KernelInputs kernel_inputs;
		kernel_inputs.bandwidth = 10;

		//std::cout <<"Loading nystrom params" <<std::endl;
		NystromInputs nystrom_inputs(6);
  
		mpi::Barrier(mpi::COMM_WORLD);
		//std::cout << "Making kernel class" <<std::endl;
		GaussKernel gKernel(kernel_inputs, &grid);

		//std::cout << "Initializing NystromAlg obj" <<std::endl;
		NystromAlg nyst(&refData,kernel_inputs,nystrom_inputs,&grid, gKernel);

		//std::cout << "Running decomp" <<std::endl;
		nyst.decomp();

		//std::cout << "Running orthog" <<std::endl;
		nyst.orthog();

		DistMatrix<double> test_vec(grid);
		Uniform(test_vec,ntrain,1);
		DistMatrix<double> ans(grid);
		
		//std::cout << "Testing multiply" <<std::endl;
		nyst.matvec(test_vec,ans);

		//std::cout << "Testing appinv" << std::endl;
		ans.Empty();
		nyst.appinv(test_vec,ans);
	
		/*	
		// Trying to write kernel business
		DistMatrix<double> A(grid);
		int height = 4; int width = 4;
		Uniform(A,height,width);
		Print(A, "Data");

		DistMatrix<double> K(width,width,grid);
		Fill(K,0.0);
		Herk(UPPER,NORMAL, -2.0,A, 1.0,K);

		auto elem_sqr = [](double x){return x*x;};
		EntrywiseMap(A,function<double(double)> (elem_sqr));
		Print(A,"Ptwise sqr data");
			
		DistMatrix<double,VR,STAR> ones(width,1,grid);
		Fill(ones,1.0);
		
		DistMatrix<double,VR,STAR> norms(height,1,grid);
		Fill(norms,0.0);

		Gemv(NORMAL, 1.0,A,ones, 1.0,norms);
		Print(norms, "norms");
		
		// Need to combine it all back
		ones.Resize(height,1);
		Fill(ones,1.0);
		Print(ones,"ones");
		Her2(UPPER, 1.0,ones,norms, K);
		auto elem_sqrt = [](double x){return sqrt(x);};
		EntrywiseMap(K,function<double(double)> (elem_sqrt));

		Print(K,"unexponentiated kernel");
		
		auto elem_exp = [](double x){return exp(x);};
		Scale(-1.0,K);
		EntrywiseMap(K,function<double(double)>(elem_exp));
		Print(K,"final kernel (upper)");
		*/


		//Print(A);
	}
	catch(exception& e){ ReportException(e); }
	Finalize();

	return 0;
}

