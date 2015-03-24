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

		// Overloaded constructor (this one takes both rank and samples)
		NystromInputs(int _rank, int _samples)
			:
				rank(_rank),
				samples(_samples)
		{};
		
		// Overloaded constructor (takes just rank, samples = rank)
		NystromInputs(int _rank)
			:
				rank(_rank),
				samples(_rank)
		{};


}; // class


#endif
