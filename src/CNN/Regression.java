package CNN;

public class Regression implements Model {

	private Layer[] hiddenLayer;
	private Layer outputLayer;
	private final ActivationFunc activation;
	
	public Regression(int inputSize, int hiddenLayerCount, int[] hiddenLayerSizes, int outputLayerSize, ActivationFunc activation) {
		this.activation = activation;
		hiddenLayer = new Layer[hiddenLayerCount];
		int prevSize = inputSize;
		for (int i = 0; i < hiddenLayerCount; i++) {
			hiddenLayer[i] = new Layer(prevSize, hiddenLayerSizes[i], activation);
			prevSize = hiddenLayerSizes[i];
		}
		outputLayer = new Layer(prevSize, outputLayerSize, activation);
	}
	
    /**
     * Computes the mean squared error between predicted and target values.
     */
    @Override
    public double lossFunc(double[] predicted, double[] target) {
        double sum = 0.0;
        for (int i = 0; i < predicted.length; i++) {
            double diff = predicted[i] - target[i];
            sum += diff * diff;
        }
        return sum / predicted.length;
    }
}
