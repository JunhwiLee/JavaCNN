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

       public void learning(double[][][][] trainingSet, double[][][][] validationSet) {
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
                               double[] target = flatten(validationSet[i]);

                               double[] features = convolution.forward(input);
                               model.forward(features);

                               double[] grad;
                               if(model instanceof Classification) {
                                       grad = ((Classification)model).backward(target, learningRate);
                               } else if(model instanceof Regression) {
                                       grad = ((Regression)model).backward(target, learningRate);
                               } else {
                                       grad = model.backward(target);
                               }

                               convolution.backward(grad, learningRate);
                       }
               }
       }

        private static double[] flatten(double[][][] tensor) {
                int depth = tensor.length;
                int height = tensor[0].length;
                int width = tensor[0][0].length;
                double[] out = new double[depth * height * width];
                int idx = 0;
                for(int d = 0; d < depth; d++) {
                        for(int y = 0; y < height; y++) {
                                for(int x = 0; x < width; x++) {
                                        out[idx++] = tensor[d][y][x];
                                }
                        }
                }
                return out;
        }
}
