package CNN;

public class ConvolutionNeuralNetwork {
	
	private final Convolution2D convolution;
	private final Model model;
	
	public ConvolutionNeuralNetwork(Convolution2D convolution, Model model) {
		this.convolution = convolution;
		this.model = model;
	}
	
	public double[] estimate(double[][][] input) {
		return model.forward(convolution.forward(input));
	}
	
	public void learning(double[][][][] trainingSet, int[] trainLabel, double[][][][] validationSet, int[] validLabel) {
		double learningRate = 0.01;
	}
}
