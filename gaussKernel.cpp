#include "gaussKernel.hpp"

using namespace El;


void GaussKernel::SelfKernel(DistMatrix<double>& data, DistMatrix<double>& K){
	// Find distances
	this->SelfDists(data,K);
	
	// Exponentiate
	auto elem_exp = [](double x){return exp(x);};
	Scale(-gamma,K);
	EntrywiseMap(K,function<double(double)>(elem_exp));

}
		
void GaussKernel::SelfDists(DistMatrix<double>& data, DistMatrix<double>& dists){
	// Get parameters of the data
	Int dim = data.Height(); 
	Int ntrain = data.Width(); 

	// Check the dimensions
	if (dim != dists.Height() || ntrain != dists.Width()){ 
		if (mpi::WorldRank() == 0){
			std::cout << "COMP_SELFDISTS error: dist matrix was not initialized correctly!!!" << std::endl;
		}
		return;
	}
	
	// Mem-allocated already, fill with 0s
	Fill(dists,0.0);

	// dists = -2 * data * data^T
	Herk(UPPER, ADJOINT, -2.0,data, 1.0,dists);
	
	// Elementwise Sqr and data copy
	auto data_cpy(data);
	auto elem_sqr = [](double x){return x*x;};
	EntrywiseMap(data_cpy,function<double(double)> (elem_sqr));

	// Add up along each row for sqred norms
	DistMatrix<double,VR,STAR> ones(dim,1,*grid); 
	Fill(ones,1.0);
	DistMatrix<double,VR,STAR> norms(ntrain,1,*grid); 
	Fill(norms,0.0);
	Gemv(TRANSPOSE, 1.0,data_cpy,ones, 1.0,norms); 
	
	// Need to combine it all back
	ones.Resize(ntrain,1);
	Fill(ones,1.0);
	
	// Add outer products dists += norms * ones^T + ones * norms^T
	Her2(UPPER, 1.0,ones,norms, dists); 

}

void GaussKernel::Kernel(DistMatrix<double>& data1, DistMatrix<double>& data2, DistMatrix<double>& K){
	this->Dists(data1,data2,K);
	
	// Exponentiate
	auto elem_exp = [](double x){return exp(x);};
	Scale(-gamma,K);
	EntrywiseMap(K,function<double(double)>(elem_exp));
}


void GaussKernel::Dists(DistMatrix<double>& data1, DistMatrix<double>& data2, DistMatrix<double> &dists){
	// Get initial sizes
	Int dim    = data1.Height(); 
	Int ntrain = data1.Width();
	Int ntest  = data2.Width();
	
	// Dimension checks
	if (dim != data2.Height()){
		if(mpi::WorldRank() == 0){ 
			std::cout << "COMP_DISTS error: data dimensions don't match!!!" <<std::endl;
		}
		return;
	}
	
	// Test if dists is initialized appropriately
	Int d_train = dists.Height();
	Int d_test  = dists.Width();
	if(d_train != ntrain || d_test != ntest){
		if(mpi::WorldRank() == 0){
			std::cout << "COMP_DISTS error: dists dimension is wrong!!!" <<std::endl;
		}
		return;
	}

	auto elem_sqr = [](double x){return x*x;};//Will need later
	
	// Mem-allocated already, fill with 0s
	Fill(dists,0.0);

	// dists = -2 * data * data^T
	Gemm(TRANSPOSE, NORMAL, -2.0,data1,data2, 1.0,dists);//switch up
	
	// Deal with the first guys norms
	// Elementwise sqr
	auto data_cpy(data1);
	EntrywiseMap(data_cpy,function<double(double)> (elem_sqr));

	// Add up along each row for sqred norms//switch up
	DistMatrix<double,VR,STAR> sum_vec(dim,1,*grid);
	Fill(sum_vec,1.0);
	DistMatrix<double,VR,STAR> norms(ntrain,1,*grid);
	Fill(norms,0.0);
	Gemv(TRANSPOSE, 1.0,data_cpy,sum_vec, 1.0,norms);
	data_cpy.Empty();
	
	// Need to combine it all back
	DistMatrix<double,VR,STAR> ones(ntest,1,*grid);
	Fill(ones,1.0);
	
	// Add outer products dists += norms * ones^T
	Ger(1.0,norms,ones, dists);
	ones.Empty();
	

	// Deal with the second guys norms
	// Elementwise sqr
	data_cpy = data2;
	EntrywiseMap(data_cpy,function<double(double)> (elem_sqr));

	// Add up along each row for sqred norms
	norms.Resize(ntest,1);
	Fill(norms,0.0);
	Gemv(TRANSPOSE, 1.0,data_cpy,sum_vec, 1.0,norms);
	data_cpy.Empty();

	// Need to combine it all back
	ones.Resize(ntrain,1);
	Fill(ones,1.0);
	
	// Add outer products dists += ones * norms^T
	Ger(1.0,ones,norms, dists);
}

