#ifndef KERNEL_INPUTS_HPP_
#define KERNEL_INPUTS_HPP_


/**
 * This class just holds the parameters needed for different kernel functions
 * It is the user's responsibility to set the parameters needed for the
 * kernel function being used.
 */
class KernelInputs {

	public:

		// For Gaussian kernel -- only need to specify these for Gaussian

		// h
		double bandwidth;
		// if true, do the source vector dependent variable bandwidth kernel
		bool do_variable_bandwidth;

		// For Laplace kernel -- no extra parameters necessary


		// For Polynomial kernel -- (x' * y / h + c)^p
		// The exponent to use, a double for now, maybe an int?
		double power;

		// The constant added (inside the exponent)
		double constant;

		// Also need to specify the bandwidth above

		KernelInputs()
			:
				do_variable_bandwidth(false)
	{}


}; // class


#endif
