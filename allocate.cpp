#include "El.hpp"
//#include <type_traits>
#include <iostream>
#include <vector>

using namespace El;
/*
class Tester {
	public : 
		DistMatrix<double,MC,MR> K;

		Grid* g;
		
		Int height = 500000;

		Int width;

		Tester(Grid* _g, Int _width):
			g(_g),
			width(_width)
	{
		K.SetGrid(*_g);
		K.Resize(height,_width);
	};
		
};
template<const Grid * const g>
class TempTester {
	public : 
		DistMatrix<double,MC,MR> K(*g);
		
		Int height = 50000;

		Int width;
		
		TempTester(Int _width):
			width(_width)
	{
		//std::cout << g <<std::endl;
		K.Resize(height,_width);
	};

		
};
*/

int main(int argc, char* argv []){
	
	Initialize(argc,argv);
	Int width = Input<Int>("--w","width of matrix");
	Int alg   = Input<Int>("--a","alg to run",0);

	ProcessInput();
	PrintInputReport();

	const Grid grid(mpi::COMM_WORLD);
	int big_guy = 1625000000;
	int w = 1625;
	int h = 19*(big_guy/w);
	//int h = 500000;
	//int w = 32768;

	if(mpi::WorldRank() == 0) {std::cout << w << " " << h << " "<< w*h << std::endl;}
	//std::vector<double> test(big_guy);
	DistMatrix<double> K(h,w,grid);
	std::cout << "mem" << K.AllocatedMemory()<<std::endl;

	//K.Resize(500000,width);
	
	// Call Tester
	//switch(alg){
	//	case 0:
	//		Tester test(&grid, width);
	//	case 1:
	//		const int g = 1;
	//		TempTester<g> test1(width);
	//	default:
	//		Tester test2(&grid, width);
	//}
	//const Grid * const g = &grid;
	//TempTester<g> test(width);
	Finalize();
	return 0;
}


