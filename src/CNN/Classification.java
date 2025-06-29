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
		double[] out = input;
		return out;
	}
	
	//TODO implement backward
	
	/**
	 * Computes cross entropy between a predicted probability distribution and
	 * the target distribution.
	 */
	private double crossEntropy(double[][] probs, int[] target) {
		double ce = 0.0;
		int index = 0;
		double maxP = 0.0;
		for (int i = 0; i < probs.length; i++) {
			for(int j = 0; j<probs[0].length; j++) {
				if(maxP < probs[i][j]) {
					maxP = probs[i][j];
					index = j;
				}
			}
			
			ce -= Math.log(maxP);			
		}
		return ce;
	}
	
	private double klDivergence(double[][] probs, double[] target) {
		return 0.0;
	}
	
	/**
	 * Computes a combined loss consisting of cross entropy and KL divergence
	 * for classification tasks.
	 */
	@Override
	public double lossFunc(double[][] predicted, double[] target) {
		int[] label = new int[target.length];
		for(int i = 0; i<target.length; i++) label[i] = (int)target[i];
		return 0.0;
	}

	public double lossFunc(double[][] predicted, int[] target) {
		int[] label = target;
		return 0.0;
	}
}
