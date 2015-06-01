#include "El.hpp"
//#include <type_traits>
#include <iostream>
#include <vector>

using namespace El;

int main(int argc, char* argv []){
	
	Initialize(argc,argv);
	Int w= Input<Int>("--w","width of matrix",12);
	Int h= Input<Int>("--h","matrix height",12);
	Int wsub= Input<Int>("--ws","width of submatrix",8);
	Int hsub= Input<Int>("--hs","submatrix height",12);

	ProcessInput();
	PrintInputReport();

	const Grid grid(mpi::COMM_WORLD);
	int proc = mpi::WorldRank();
	int big_guy = 1625000000;
	//w = 1625;
	//h = 19*(big_guy/w);

	if(mpi::WorldRank() == 0) {std::cout << w << " " << h << " "<< w*h << std::endl;}
	DistMatrix<double> K(grid);
	Uniform(K,h,w);
	Print(K,"K");
	
	DistMatrix<double> Ksub(grid);
	std::vector<int> widx(wsub);
	std::vector<int> hidx(hsub);
	for(int i = 0;i<wsub;i++){widx[i] = i;}
	for(int i = 0;i<hsub;i++){hidx[i] = i;}
	GetSubmatrix(K,hidx,widx,Ksub);
	Print(Ksub,"Sub");

	K.Resize(hsub,wsub);
	Print(K,"restrict");

	Finalize();
	return 0;
}


