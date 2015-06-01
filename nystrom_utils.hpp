#ifndef NYSTROM_UTILS_HPP_
#define NYSTROM_UTILS_HPP_

#include <vector>
#include "El.hpp"

using namespace El;

/**
 * This class just holds the parameters needed for different kernel functions
 * It is the user's responsibility to set the parameters needed for the
 * kernel function being used.
 */
class NystromInputs {

	public:

		// Approximation rank
		int rank;
		
		// Sampling rank (default is same as approximation rank)
		int samples;

		NystromInputs(){};

		// Overloaded constructor (takes both rank and samples)
		NystromInputs(int _rank, int _samples)
			:
				rank(_rank),
				samples(_samples)
		{};
		
}; // class

/*
 * Local exclusive scan (every process will do an exclusive 
 * scan on its own vector x)
 */
std::vector<int> loc_exscan(std::vector<int> x);

/*
 * Takes contiguous local data (i.e. process 0 owns the first 
 * nlocal elements of the global vector, process 1 the next, etc.) 
 * and creates and elemental vector from it. The elemental vector must 
 * have an associated grid, but does not need to be correctly sized
 */
void make_vec_from_loc(std::vector<double>& loc_vec, El::DistMatrix<double,VC,STAR> el_vec);

#endif
