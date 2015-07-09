#include "El.hpp"
//#include <type_traits>
#include <iostream>
#include <vector>

using namespace El;

int main(int argc, char* argv []){
	
	Initialize(argc,argv);
	
	// Inputs
	Int w= Input<Int>("--w","width of matrix",12);
	Int h= Input<Int>("--h","matrix height",12);
	Int wsub= Input<Int>("--ws","width of submatrix",8);
	Int hsub= Input<Int>("--hs","submatrix height",12);
	bool do_range = Input("--r","do range?",true);
	bool print = Input("--p","print?",true);
	ProcessInput();
	PrintInputReport();


	const Grid grid(mpi::COMM_WORLD);
	int proc = mpi::WorldRank();
	//int big_guy = 1625000000;
	//w = 1625;
	//h = 19*(big_guy/w);

	// Generate matrix
	if(proc == 0) {std::cout << w << " " << h << " "<< w*h << std::endl;}
	DistMatrix<double> K(grid);
	Uniform(K,h,w);
	//Print(K,"K");
	mpi::Barrier(mpi::COMM_WORLD);

	// Extract Submatrix with either Ranges or std::vec
	DistMatrix<double> Ksub(grid);
	if(do_range){
		Range<int> hrng = Range<int>(0,hsub);
		Range<int> wrng = Range<int>(0,wsub);
		GetSubmatrix(K,hrng,wrng,Ksub);
		if(print){Print(Ksub,"Sub (ranges)");}
	}else{
		std::vector<int> widx(wsub);
		std::vector<int> hidx(hsub);
		for(int i = 0;i<wsub;i++){widx[i] = i;}
		for(int i = 0;i<hsub;i++){hidx[i] = i;}
		GetSubmatrix(K,hidx,widx,Ksub);
		if(print){Print(Ksub,"Sub (indices)");}
	}
	mpi::Barrier(mpi::COMM_WORLD);
	Ksub.Empty();
	
	// Check if same thing happens when we just restrict
	K.Resize(hsub,wsub);
	mpi::Barrier(mpi::COMM_WORLD);
	if(print){Print(K,"Restrict");}

	K.Empty();
	Uniform(K,20,20);
	DistMatrix<double> Umm(grid);
	DistMatrix<double,VC,STAR> Lmm(grid);
	HermitianEig(UPPER,K,Lmm,Umm,DESCENDING);

	Finalize();
	return 0;
}


