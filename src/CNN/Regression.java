package CNN;

/**
 * Neural Network for Regression
 */
public class Regression implements Model {
	
	private final Layer[] hiddenLayer;
	private final Layer outputLayer;
	private final ActivationFunc activation;
	
	private double[][] layerInputs;
	private double[] lastOutput;
	
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
	
	public int getOutputVectorSize() {
		return outputLayer.getLayerSize();
	}
	
	/**
	 * Executes a forward pass through all layers of the network.
	 *
	 * @param input input vector for the model
	 * @return activated output of the network
	 */
	public double[] forward(double[] input) {
		double[] out = input;
		layerInputs = new double[hiddenLayer.length + 1][];
		layerInputs[0] = input;
		for (int i = 0; i < hiddenLayer.length; i++) {
			out = hiddenLayer[i].activation(hiddenLayer[i].forward(out));
			layerInputs[i + 1] = out;
		}
		lastOutput = outputLayer.activation(outputLayer.forward(out));
		return lastOutput;
	}
	
	public double[] backward(double[] input) {
		double learningRate = 0.01;
		double[] delta = new double[lastOutput.length];
		for(int i = 0; i < lastOutput.length; i++) {
			delta[i] = lastOutput[i] - input[i];
		}
		
		double[] grad = outputLayer.backward(layerInputs[hiddenLayer.length], delta, learningRate);
		for(int i = hiddenLayer.length - 1; i >= 0; i--) {
			grad = hiddenLayer[i].backward(layerInputs[i], grad, learningRate);
		}
		return grad;
	}
	
	
	/**
	 * Computes the mean squared error between predicted and target values.
	 */
	@Override
	public double lossFunc(double[][] predicted, double[] target) {
		double sum = 0.0;
		for (int i = 0; i < predicted.length; i++) {
			double diff = predicted[i][0] - target[i];
			sum += diff * diff;
		}
		return sum / predicted.length;
	}
}
