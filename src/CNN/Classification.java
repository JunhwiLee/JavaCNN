package CNN;

/**
 * Neural Network for Classification
 */
public class Classification implements Model{
	
	private final Layer[] hiddenLayers;
	private final Layer outputLayer;
	private final ActivationFunc activation;
	private final int batch;
		
	public Classification(int inputSize, int hiddenLayerCount, int[] hiddenLayerSizes, int outputLayerSize, ActivationFunc activation, int batch) {
		if(hiddenLayerCount != hiddenLayerSizes.length) {
			throw new LayerCountMissmatchException(hiddenLayerCount, hiddenLayerSizes.length);
		}
		this.batch = batch;
		this.activation = activation;
		hiddenLayers = new Layer[hiddenLayerCount];
		int prevSize = inputSize;
		for (int i = 0; i < hiddenLayerCount; i++) {
			hiddenLayers[i] = new Layer(prevSize, hiddenLayerSizes[i], activation, batch);
			prevSize = hiddenLayerSizes[i];
		}
		outputLayer = new Layer(prevSize, outputLayerSize, activation, batch);
	}
	
	public int getOutputVectorSize() {
		return outputLayer.getLayerSize();
	}
	
	public int getBatch() {
		return batch;
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
	
	public double[] estimate(double[] input) {
		double[] out = input;		
		for (int i = 0; i < hiddenLayers.length; i++) {
			out = hiddenLayers[i].activation(hiddenLayers[i].estimate(out));
		}
		return softmax(outputLayer.estimate(out));
	}
	
	public double[][] forward(double[][] input) {
		double[][] out = input;
		for (int i = 0; i < hiddenLayers.length; i++) {
			out = activation.activate2D(hiddenLayers[i].forward(out));
		}
		return outputLayer.forward(out);
	}
	
	public double[] backward(double[] input) {
		return new double[0];
	}
	
	
	public static double[][] ceDerivative(double[][] predictions, int[] targets) {
		int N = predictions.length;
		if (N == 0 || N != targets.length) {
			throw new IllegalArgumentException("predictions.length must equal targets.length and > 0");
		}
		int C = predictions[0].length;
		double[][] grad = new double[N][C];
		
		for (int i = 0; i < N; i++) {
			int t = targets[i];
			double p = predictions[i][t];
			if (p <= 0.0) {
				throw new IllegalArgumentException(
						String.format("predictions[%d][%d] must be > 0, but got %f", i, t, p));
			}
			// one-hot y_j: only at j==t is y=1, else 0
			grad[i][t] = -1.0 / (N * p);
			// other entries remain 0
		}
		return grad;
	}
	
	@Override
	public double lossFunc(double[][] predicted, double[] target) {
		// double[] target 을 int[] 로 변환
		int[] label = new int[target.length];
		for (int i = 0; i < target.length; i++) {
			label[i] = (int) target[i];
		}
		return lossFunc(predicted, label);
	}
	
	public double lossFunc(double[][] predicted, int[] target) {
		if(predicted.length != target.length) throw new InputSizeMissmatchException();
		double ce = 0.0;
		for (int i = 0; i < predicted.length; i++) {
			int lbl = target[i];
			double p = predicted[i][lbl];
			if (p <= 0.0) {
				throw new IllegalArgumentException(
						String.format("predictions[%d][%d] must be > 0, but got %f", i, lbl, p));
			}
			ce -= Math.log(p);
		}
		return ce / predicted.length;
	}	
	
	@Override
	public double lossDerivative(double[][] predicted, double[] target) {
		int[] label = new int[target.length];
		for (int i = 0; i < target.length; i++) {
			label[i] = (int) target[i];
		}
		return lossDerivative(predicted, label)[0][0];
	}
	
	public double[][] lossDerivative(double[][] predicted, int[] target) {
		int N = predicted.length;
		if (N == 0 || N != target.length) {
			throw new IllegalArgumentException("predictions.length must equal targets.length and > 0");
		}
		int C = predicted[0].length;
		double[][] grad = new double[N][C];
		
		for (int i = 0; i < N; i++) {
			int t = target[i];
			double p = predicted[i][t];
			if (p <= 0.0) {
				throw new IllegalArgumentException(
						String.format("predictions[%d][%d] must be > 0, but got %f", i, t, p));
			}
			// one-hot y_j: only at j==t is y=1, else 0
			grad[i][t] = -1.0 / (N * p);
			// other entries remain 0
		}
		return grad;
	}
}
