package CNN;

/**
 * Neural Network for Regression
 */
public class Regression implements Model {

        private final Layer[] hiddenLayer;
        private final Layer outputLayer;
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
        * Executes a forward pass through all layers of the network.
        *
        * @param input input vector for the model
        * @return activated output of the network
        */
       public double[] forword(double[] input) {
               double[] out = input;
               for (Layer layer : hiddenLayer) {
                       out = layer.activation(layer.forward(out));
               }
               return outputLayer.activation(outputLayer.forward(out));
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
