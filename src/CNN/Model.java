package CNN;

/**
 * Simple interface for model used in the neural network.
 */
public interface Model {
	/**
	 * Computes the loss between the predicted values and the expected
	 * values. Implementations may calculate different loss metrics
	 * (e.g. mean squared error or cross entropy).
	 *
	 * @param predicted values predicted by the network
	 * @param target    the ground truth values
	 * @return computed loss
	 */
	public double lossFunc(double[] predicted, double[] target);
	public double[] forward(double[] input);
	public double[] backward(double[] input);
}
