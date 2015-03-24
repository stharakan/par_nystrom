
#include "nystrom_alg.hpp"
#include <iostream>

using namespace El;

int main(int argc, char* argv []){
	Initialize(argc,argv);

	try{
		Int ntrain = 50;
		Int dim = 50;
	
		std::cout << "Creating grid" <<std::endl;
		Grid grid(mpi::COMM_WORLD);
	

		std::cout << "Initializing data" <<std::endl;
		DistMatrix<double> refData(ntrain,dim,grid);

		std::cout << "Loading data" <<std::endl;
		//Uniform(*refData,ntrain,dim);
		for(Int i=0;i<ntrain;i++){
			for(Int j =0;j<dim;j++){
				//std::cout << i << " " << j << std::endl;
				refData.Set(i,j, (double) (i+j));
			}
		}

		std::cout << "Loading kernel params" <<std::endl;
		KernelInputs kernel_inputs;
		kernel_inputs.bandwidth = 1;

		std::cout <<"Loading nystrom params" <<std::endl;
		NystromInputs nystrom_inputs(10);
  	
		std::cout << "Initializing NystromAlg obj" <<std::endl;
		NystromAlg nyst(&refData,kernel_inputs,nystrom_inputs,&grid); 


	}
	catch(exception& e){ ReportException(e); }

	Finalize();
	return 0;
}

