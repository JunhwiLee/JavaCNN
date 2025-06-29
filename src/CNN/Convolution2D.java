package CNN;

public class Convolution2D {
	private final Convolution2DLayer[] convolutionLayers;
	private final ActivationFunc activation;
	private final int kernel, stride;
	
	private double[][][][] lastInputs;
	private double[][][][] lastActivations;
	private double[][][] lastConvOutput;
	
	public Convolution2D(int layerCount, int[] layerSizes, int outputVectorSize,
			int kernel, int[] strides, int stride, ActivationFunc activation) {
		if(layerCount != layerSizes.length ||
				layerCount != strides.length
				) {
			throw new LayerCountMissmatchException(layerCount, layerSizes.length);
		}
		this.activation = activation;
		this.kernel = kernel;
		this.stride = stride;
		convolutionLayers = new Convolution2DLayer[layerCount];
		for(int i = 0; i<layerCount - 1; i++) {
			convolutionLayers[i] = new Convolution2DLayer(layerSizes[i], layerSizes[i + 1], kernel, kernel - 1, strides[i], activation);
		}
		convolutionLayers[layerCount - 1] = new Convolution2DLayer(layerSizes[layerCount - 1], outputVectorSize, kernel, kernel - 1, strides[layerCount - 1], activation);
	}
	
	public double[] forward(double[][][] input){
		double[][][] out = input;
		lastInputs = new double[convolutionLayers.length][][][];
		lastActivations = new double[convolutionLayers.length - 1][][][];
		
		for(int i = 0; i < convolutionLayers.length - 1; i++) {
			lastInputs[i] = out;
			out = convolutionLayers[i].forward(out);
			for(int f = 0; f < out.length; f++) {
				out[f] = activation.activate2D(out[f]);
			}
			lastActivations[i] = out;
			out = pooling(out);
		}
		lastInputs[convolutionLayers.length - 1] = out;
		out = convolutionLayers[convolutionLayers.length - 1].forward(out);
		lastConvOutput = out;
		return globalPooling(out);
	}
	
	public double[][][] pooling(double[][][] input){
		int depth = input.length;
		int height = input[0].length;
		int width = input[0][0].length;
		
		int outHeight = height / stride;
		int outWidth = width / stride;
		double[][][] out = new double[depth][outHeight][outWidth];
		
		for(int d = 0; d < depth; d++) {
			for(int y = 0; y < outHeight; y += stride) {
				for(int x = 0; x < outWidth; x += stride) {
					double sum = 0.0;
					for(int i = 0; i < kernel; i++) {
						for(int j = 0; j < kernel; j++) {
							int inY = y + i;
							int inX = x + j;
							if (inY >= 0 && inY < height && inX >= 0 && inX < width) {
								sum += input[d][inY][inX];
							}
						}
					}
					out[d][y][x] = sum / (kernel * kernel);
				}
			}
		}
		
		return out;
	}
	
	public double[] globalPooling(double[][][] input){
		int depth = input.length;
		int height = input[0].length;
		int width = input[0][0].length;
		
		double[] out = new double[depth];
		
		for(int d = 0; d < depth; d++) {
			double sum = 0.0;
			for(int y = 0; y < height; y++) {
				for(int x = 0; x < width; x++) {
					sum += input[d][y][x];
				}
			}
			out[d] = sum / (height * width);
		}
		
		return out;
	}
	
	//TODO implement backward, poolingBackward, globalPoolingBackward
}
