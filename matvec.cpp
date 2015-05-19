#include "El.hpp"

using namespace El;

int main(int argc, char* argv[]){
	Initialize(argc,argv);
	
	const Int m = Input("--size","size of matrix",1000);
	const Int runs = Input("--runs","runs to average",10);
	
	Grid grid(mpi::COMM_WORLD);

	DistMatrix<double> A(grid);
	Uniform(A,m,m,0.5,0.5);

	DistMatrix<double> vec(grid);
	DistMatrix<double> ans1(m,1,grid);
	Fill(ans1,0.0);
	DistMatrix<double> ans2(m,1,grid);
	Fill(ans2,0.0);

	double time_normal = 0.0;
	double time_trans  = 0.0;
	double start;

	for(int run = 0; run<runs;run++){
		Uniform(vec,m,1);
		
		start = mpi::Time();
		Gemv(NORMAL,1.0, A, vec, 1.0, ans1);
		time_normal += mpi::Time() - start;

		start = mpi::Time();
		Gemv(TRANSPOSE,1.0, A, vec, 1.0, ans2);
		time_trans  += mpi::Time() - start;
	}

	time_normal /= runs;
	time_trans /= runs;

	double max_time_normal;
	double max_time_trans;
	mpi::Reduce(&time_normal,&max_time_normal,1,mpi::MAX,0,mpi::COMM_WORLD);
	mpi::Reduce(&time_trans,&max_time_trans,1,mpi::MAX,0,mpi::COMM_WORLD);

	if(mpi::WorldRank() == 0){
		std::cout << "Max time normal: " << max_time_normal <<std::endl;
		std::cout << "Max time trans: " << max_time_trans <<std::endl;
	}


	Finalize();
	return 0;
}
