#ifndef NYSTROM_UTILS_HPP_
#define NYSTROM_UTILS_HPP_

#include<vector>

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

		// Overloaded constructor (takes both rank and samples)
		NystromInputs(int _rank, int _samples)
			:
				rank(_rank),
				samples(_samples)
		{};
		
}; // class

/**
 * Makes the test index set and returns it in testIdx
 * User should initialize testIdx to the appropriate length,
 * generally 1000
 *
void make_testIdx(std::vector<int>& testIdx, int ntrain){
	
	int test_pts =  testIdx.size();
	double step = ((double)ntrain)/((double) test_pts);
	
	for(int i=0;i<test_pts;i++){
		int currIdx = (int)(i * step);
		testIdx[i] = currIdx;
	}
}
*/
#endif
