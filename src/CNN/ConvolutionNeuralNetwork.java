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
                double[][] predicted = new double[trainingSet.length][model.getOutputVectorSize()];

                int patience = 5;
                int noImprove = 0;
                double bestAcc = 0.0;

                while (noImprove < patience) {
                        for (int i = 0; i < trainingSet.length; i++) {
                                double[] output = model.forward(convolution.forward(trainingSet[i]));
                                predicted[i] = output;

                                double[] target = new double[model.getOutputVectorSize()];
                                if (target.length == 1) {
                                        target[0] = trainLabel[i];
                                } else {
                                        int idx = (int) trainLabel[i];
                                        if (idx >= 0 && idx < target.length) {
                                                target[idx] = 1.0;
                                        }
                                }

                                double[] grad = model.backward(target);
                                convolution.backward(grad);
                        }

                        double loss = model.lossFunc(predicted, trainLabel);

                        int correct = 0;
                        for (int i = 0; i < validationSet.length; i++) {
                                double[] out = estimate(validationSet[i]);
                                int predLabel = 0;
                                for (int j = 1; j < out.length; j++) {
                                        if (out[j] > out[predLabel]) predLabel = j;
                                }
                                if (predLabel == validLabel[i]) correct++;
                        }
                        double acc = (double) correct / validationSet.length;
                        if (acc > bestAcc) {
                                bestAcc = acc;
                                noImprove = 0;
                        } else {
                                noImprove++;
                        }
                }

        }
}
