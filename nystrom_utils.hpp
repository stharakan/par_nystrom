#ifndef NYSTROM_UTILS_HPP_
#define NYSTROM_UTILS_HPP_


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


#endif
