package CNN;

/**
 * Simple interface for model used in the neural network.
 */
public interface Model {
	
	/**
     * Applies the loss function to the supplied value.
     */
	public double lossFunc();
}
