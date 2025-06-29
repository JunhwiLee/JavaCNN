package CNN;

/**
 * Neural Network for Classification
 */
public class Classification implements Model{
	
	private final Layer[] hiddenLayers;
	private final Layer outputLayer;
	private final ActivationFunc activation;
	
	// Store inputs to each layer during the forward pass for backprop
	private double[][] layerInputs;
	// Store raw logits of the output layer
	private double[] lastLogits;
	
	public Classification(int inputSize, int hiddenLayerCount, int[] hiddenLayerSizes, int outputLayerSize, ActivationFunc activation) {
		if(hiddenLayerCount != hiddenLayerSizes.length) {
			throw new LayerCountMissmatchException(hiddenLayerCount, hiddenLayerSizes.length);
		}
		this.activation = activation;
		hiddenLayers = new Layer[hiddenLayerCount];
		int prevSize = inputSize;
		for (int i = 0; i < hiddenLayerCount; i++) {
			hiddenLayers[i] = new Layer(prevSize, hiddenLayerSizes[i], activation);
			prevSize = hiddenLayerSizes[i];
		}
		outputLayer = new Layer(prevSize, outputLayerSize, activation);
	}
	
	public int getOutputVectorSize() {
		return outputLayer.getLayerSize();
	}
	
	/**
	 * Applies the softmax function to the given logits.
	 *
	 * @param logits raw output scores from the network
	 * @return probability distribution
	 */
	private static double[] softmax(double[] logits) {
		double max = logits[0];
		for (int i = 1; i < logits.length; i++) {
			if (logits[i] > max) {
				max = logits[i];
			}
		}
		
		double[] probs = new double[logits.length];
		double sum = 0.0;
		for (int i = 0; i < logits.length; i++) {
			probs[i] = Math.exp(logits[i] - max);
			sum += probs[i];
		}
		for (int i = 0; i < probs.length; i++) {
			probs[i] /= sum;
		}
		return probs;
	}
	
	public double[] forward(double[] input) {
		double[] out = input;
		// Prepare storage for backprop inputs
		layerInputs = new double[hiddenLayers.length + 1][];
		layerInputs[0] = input;
		
		for (int i = 0; i < hiddenLayers.length; i++) {
			out = hiddenLayers[i].activation(hiddenLayers[i].forward(out));
			layerInputs[i + 1] = out;
		}
		lastLogits = outputLayer.forward(out);
		return softmax(lastLogits);
	}
	
	public double[] backward(double[] input) {
		double learningRate = 0.01;
		double[] probs = softmax(lastLogits);
		double[] delta = new double[probs.length];
		for(int i = 0; i < probs.length && i < input.length; i++) {
			delta[i] = probs[i] - input[i];
		}
		
		double[] grad = outputLayer.backward(layerInputs[hiddenLayers.length], delta, learningRate);
		for(int i = hiddenLayers.length - 1; i >= 0; i--) {
			grad = hiddenLayers[i].backward(layerInputs[i], grad, learningRate);
		}
		return grad;
	}
	
	
	/**
	 * Computes cross entropy between a predicted probability distribution and
	 * the target distribution.
	 */
	private double crossEntropy(double[][] probs, int[] target) {
		double ce = 0.0;
		for (int i = 0; i < probs.length; i++) {
			int lbl = target[i];
			double p = probs[i][lbl];
			if (p <= 0.0) {
				p = 1e-15; // avoid log(0)
			}
			ce -= Math.log(p);
		}
		return ce / probs.length;
	}
	
	private double klDivergence(double[][] probs, double[] target) {
		double kl = 0.0;
		for (int i = 0; i < probs.length; i++) {
			for (int j = 0; j < probs[i].length && j < target.length; j++) {
				double p = probs[i][j];
				double q = target[j];
				if (p <= 0.0 || q <= 0.0) {
					continue;
				}
				kl += p * Math.log(p / q);
			}
		}
		return kl / probs.length;
	}
	
	/**
	 * Computes a combined loss consisting of cross entropy and KL divergence
	 * for classification tasks.
	 */
	@Override
	public double lossFunc(double[][] predicted, double[] target) {
		int[] label = new int[target.length];
		for(int i = 0; i<target.length; i++) label[i] = (int)target[i];
		return lossFunc(predicted, label);
	}
	
	public double lossFunc(double[][] predicted, int[] target) {
		int[] label = target;
		double ce = crossEntropy(predicted, label);
		// use uniform distribution for KL divergence regularization
		double[] uniform = new double[predicted[0].length];
		for (int i = 0; i < uniform.length; i++) {
			uniform[i] = 1.0 / uniform.length;
		}
		double kl = klDivergence(predicted, uniform);
		return ce + kl;
	}
}
