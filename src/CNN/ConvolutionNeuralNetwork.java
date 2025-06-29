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
		int samples = Math.min(trainingSet.length, validationSet.length);
		
		java.util.Random rnd = new java.util.Random();
		int epochs = 1;
		
		for(int epoch = 0; epoch < epochs; epoch++) {
			int[] order = new int[samples];
			for(int i = 0; i < samples; i++) order[i] = i;
			// shuffle order for SGD
			for(int i = samples - 1; i > 0; i--) {
				int j = rnd.nextInt(i + 1);
				int tmp = order[i];
				order[i] = order[j];
				order[j] = tmp;
			}
			
			for(int idx = 0; idx < samples; idx++) {
				int i = order[idx];
				double[][][] input = trainingSet[i];
				
				double[] features = convolution.forward(input);
				model.forward(features);
				
				double[] grad;
				if(model instanceof Classification) {
					grad = ((Classification)model).backward(trainLabel, learningRate);
				} else if(model instanceof Regression) {
					grad = ((Regression)model).backward(trainLabel, learningRate);
				} else {
					grad = model.backward(trainLabel);
				}
				
				convolution.backward(grad, learningRate);
			}
		}
	}
}
