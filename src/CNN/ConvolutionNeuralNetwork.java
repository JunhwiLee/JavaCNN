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
	
	//using Stochastic Gradient Descent
	public void learning(double[][][][] trainingSet, double[] trainLabel, double[][][][] validationSet, int[] validLabel) {
		double learningRate = 0.01;
		double[][] predicted = new double[trainingSet.length][model.getOutputVectorSize()];
		
		while(true) { // repeat until validation set accuracy doesn't improve for specific time
			for(int i = 0; i<trainingSet.length; i++) {
				predicted[i] = estimate(trainingSet[i]);
			}
			
			double loss = model.lossFunc(predicted, trainLabel);
		}
		
	}
}
