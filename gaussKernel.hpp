#include "El.hpp"
#include "math.h"
#include "omp.h"
#include "kernel_inputs.hpp"
#include <iostream>
#include <functional>

using namespace El;

class GaussKernel {

	public:
		// Kernel Params
		double sigma;
		double gamma;

		// Grid
		Grid* grid;
		/*
		 * Constructor:
		 * kernel_inputs: contains sigma/gamma
		 */
		GaussKernel(KernelInputs kernel_inputs,Grid* _grid) :
			grid(_grid)
		{
			sigma = kernel_inputs.bandwidth;
			gamma = 1.0 / (2 * sigma * sigma);
		};

		~GaussKernel(){};

		/*
		 * Computes a Gaussian kernel matrix of a particular set of
		 * data with itself. The memory should be allocated already 
		 * for K or an error will be thrown. NOTE: Because of the way 
		 * Elemental works, only the UPPER part of K is accurate
		 */
		void SelfKernel(DistMatrix<double>& data, DistMatrix<double>& K);

		/* 
		 * Computes pairwise distances of a data matrix with itself. 
		 * User allocates memory for dists
		 */
		void SelfDists(DistMatrix<double>& data, DistMatrix<double>& dists);

		/*
		 * Computes a Gaussian kernel matrix of a set of data with 
		 * given target points. Again, the space for the kernel matrix 
		 * should already be allocated.
		 */
		void Kernel(DistMatrix<double>& data1, DistMatrix<double>& data2, DistMatrix<double>& K); 

		/*
		 * Computes pairwise distances between two sets of data. user 
		 * allocates memory for dists
		 */
		void Dists(DistMatrix<double>& data1, DistMatrix<double>& data2, DistMatrix<double> &dists);

};


