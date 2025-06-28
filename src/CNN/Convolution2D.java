package CNN;

public class Convolution2D {
	private final Convolution2DLayer[] convolutionLayers;
	private final ActivationFunc activation;
	private final int padding, stride;
	
	public Convolution2D(int layerCount, int[] layerSizes, int outputVectorSize,
			int kernel, int[] strides,
			int padding, int stride,
			ActivationFunc activation) {
		if(layerCount != layerSizes.length ||
				layerCount != strides.length
				) {
			throw new LayerCountMissmatchException(layerCount, layerSizes.length);
		}
		this.activation = activation;
		this.padding = padding;
		this.stride = stride;
		convolutionLayers = new Convolution2DLayer[layerCount];
		for(int i = 0; i<layerCount - 1; i++) {
			convolutionLayers[i] = new Convolution2DLayer(layerSizes[i], layerSizes[i + 1], kernel, kernel - 1, strides[i], activation);
		}
		convolutionLayers[layerCount - 1] = new Convolution2DLayer(layerSizes[layerCount - 1], outputVectorSize, kernel, kernel - 1, strides[layerCount - 1], activation);
	}
	
	public double[] forward(double[][][] input){
		double[][][] out = input;
		for(Convolution2DLayer layer : convolutionLayers) {
			out = layer.forward(out);
			for(int f = 0; f < out.length; f++) {
				out[f] = activation.activate2D(out[f]);
			}
			out = pooling(out);
		}
		
		double[] output = new double[out.length];
		for(int i = 0; i<out.length; i++) {
			output[i] = out[i][0][0];
		}
		
		return output;
	}
	
	public double[][][] pooling(double[][][] input){
		
	}
}
