package CNN;

import java.util.Random;

/**
 * Simple 2D convolution layer supporting multiple input channels and filters.
 * The layer performs a convolution with configurable padding and stride.
 * activation function is applied separately using {@link #activation(double[][][])}.
 */
public class Convolution2DLayer {
	
	private final double[][][][] filters; // [filter][channel][y][x]
	private final double[] biases;
	private final ActivationFunc activation;
	
	// Stores the pre-activation outputs of the last forward pass
	private double[][][] lastZ;
	
	private final int kernel;
	private final int padding;
	private final int stride;
	
	/**
	 * Creates a convolution layer with randomly initialized filters.
	 *
	 * @param inputDepth   number of input channels
	 * @param filterCount  number of convolution filters
	 * @param kernel      size of each (square) filter kernel
	 * @param padding     amount of zero-padding around the input
	 * @param stride      stride of the convolution
	 * @param activation  activation function to apply
	 */
	public Convolution2DLayer(int inputDepth, int filterCount, int kernel,
			int padding, int stride,
			ActivationFunc activation) {
		this.activation = activation;
		this.kernel = kernel;
		this.padding = padding;
		this.stride = stride;
		this.filters = new double[filterCount][inputDepth][kernel][kernel];
		this.biases = new double[filterCount];
		
		Random rnd = new Random();
		for (int f = 0; f < filterCount; f++) {
			for (int d = 0; d < inputDepth; d++) {
				for (int i = 0; i < kernel; i++) {
					for (int j = 0; j < kernel; j++) {
						filters[f][d][i][j] = rnd.nextGaussian() * 0.01;
					}
				}
			}
			biases[f] = 0.0;
		}
	}
	
	/**
	 * Executes the convolution operation without applying the activation
	 * function. The result of this method should typically be passed to
	 * {@link #activation(double[][][])}.
	 *
	 * @param input 3D input tensor indexed as [channel][y][x]
	 * @return raw convolution outputs
	 */
	public double[][][] forward(double[][][] input) {
		int depth = input.length;
		int height = input[0].length;
		int width = input[0][0].length;
		
		int outHeight = (height - kernel + padding + 1) / stride;
		int outWidth = (width - kernel + padding + 1) / stride;
		double[][][] out = new double[filters.length][outHeight][outWidth];
		lastZ = new double[filters.length][outHeight][outWidth];
		
		for (int f = 0; f < filters.length; f++) {
			for (int y = 0; y < outHeight; y += stride) {
				for (int x = 0; x < outWidth; x += stride) {
					double sum = biases[f];
					for(int d = 0; d < depth; d++) {
						for(int i = 0; i < kernel; i++) {
							for(int j = 0; j < kernel; j++) {
								int inY = y + i;
								int inX = x + j;
								if (inY >= 0 && inY < height && inX >= 0 && inX < width) {
									sum += input[d][inY][inX] * filters[f][d][i][j];
								}
							}
						}
					}
					out[f][y][x] = sum;
					lastZ[f][y][x] = sum;
				}
			}
		}
		return out;
	}
	
	/**
	 * Applies the configured activation function element-wise to the supplied
	 * tensor.
	 */
	public double[][][] activation(double[][][] input) {
		double[][][] out = new double[input.length][input[0].length][input[0][0].length];
		for (int f = 0; f < input.length; f++) {
			for (int y = 0; y < input[0].length; y++) {
			}
		}
		return out;
	}
	
	/**
	 * Propagates gradients backward through this layer and updates the filter
	 * weights and biases.
	 *
	 * @param input        inputs that produced the current outputs
	 * @param gradOutput   gradient of the loss w.r.t. the layer outputs
	 * @param learningRate step size for gradient descent
	 * @return gradient of the loss w.r.t. the inputs
	 */
	public double[][][] backward(double[][][] input, double[][][] gradOutput, double learningRate) {
		int depth = input.length;
		int height = input[0].length;
		int width = input[0][0].length;
		
		int outHeight = (height + 2 * padding - kernel) / stride + 1;
		int outWidth = (width + 2 * padding - kernel) / stride + 1;
		if (gradOutput.length != filters.length ||
				gradOutput[0].length != outHeight ||
				gradOutput[0][0].length != outWidth) {
			throw new InputSizeMissmatchException();
		}
		
		double[][][] gradInput = new double[depth][height][width];
		
		for(int f = 0; f < filters.length; f++) {
			for(int y = 0; y < outHeight; y++) {
				double[] activate = activation.derivative(lastZ[f][y]);
				for(int x = 0; x < outWidth; x++) {
					double delta = gradOutput[f][y][x] * activate[x];
					
					for(int d = 0; d < depth; d++) {
						for(int i = 0; i<kernel; i++) {
							for(int j = 0; j<kernel; j++) {
								int inY = y * stride - padding + i;
								int inX = x * stride - padding + j;
								if (inY >= 0 && inY < height && inX >= 0 && inX < width) {
									gradInput[d][inY][inX] += delta * filters[f][d][i][j];
									filters[f][d][i][j] -= learningRate * delta * input[d][inY][inX];
								}
							}
						}
					}
					biases[f] -= learningRate * delta;
				}
			}
		}
		return gradInput;
	}
}
