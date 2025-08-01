package CNN;

/**
 * Simple interface for activation functions used in the neural network.
 */
public interface ActivationFunc {
	
	/**
	 * Applies the activation function to the supplied value.
	 */
	double[] activate(double[] x);
	double[][] activate2D(double[][] x);
	/**
	 * Derivative of the activation function with respect to its input.
	 */
	double[] derivative(double[] x);
	double[][] derivative2D(double[][] x);
}
