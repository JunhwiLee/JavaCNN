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
	private double[][][][] lastZ;
	
	private final int kernel;
	private final int padding;
	private final int stride, batch;
	
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
			ActivationFunc activation, int batch) {
		this.activation = activation;
		this.kernel = kernel;
		this.batch = batch;
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
	
	public double[][][][] getLastZ(){
		return lastZ;
	}
	
	/**
	 * Executes the convolution operation without applying the activation
	 * function. The result of this method should typically be passed to
	 * {@link #activation(double[][][])}.
	 *
	 * @param input 3D input tensor indexed as [channel][y][x]
	 * @return raw convolution outputs
	 */
	public double[][][] estimate(double[][][] input) {
		int depth = input.length;
		int height = input[0].length;
		int width = input[0][0].length;
		
		int outHeight = (height - kernel + padding + 1) / stride;
		int outWidth = (width - kernel + padding + 1) / stride;
		double[][][] out = new double[filters.length][outHeight][outWidth];
		
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
				}
			}
		}
		return out;
	}
	
	public double[][][][] forward(double[][][][] input) {
		int depth = input.length;
		int height = input[0].length;
		int width = input[0][0].length;
		
		int outHeight = (height - kernel + padding + 1) / stride;
		int outWidth = (width - kernel + padding + 1) / stride;
		lastZ = new double[batch][filters.length][outHeight][outWidth];
		
		for(int b = 0; b<batch; b++) {
			lastZ[b] = estimate(input[b]);
		}
		
		return lastZ;
	}
	
	/**
	 * Applies the configured activation function element-wise to the supplied
	 * tensor.
	 */
	public double[][][] activation(double[][][] input) {
		double[][][] out = new double[input.length][input[0].length][input[0][0].length];
		for (int f = 0; f < input.length; f++) {
			out[f] = activation.activate2D(input[f]);
		}
		return out;
	}
	
	/**
	 * Backward pass for this convolution layer.
	 *
	 * @param input       input tensor used during the forward pass
	 * @param gradOutput  gradient with respect to the activated output
	 * @param lr          learning rate for parameter updates
	 * @return gradient with respect to the input tensor
	 */
        public double[][][] backward(double[][][] input, double[][][] gradOutput, double lr) {
                int depth = input.length;
                int height = input[0].length;
                int width = input[0][0].length;
                int outHeight = gradOutput[0].length;
                int outWidth = gradOutput[0][0].length;

                double[][][] gradInput = new double[depth][height][width];

                for (int f = 0; f < filters.length; f++) {
                        double[][] actDeriv = activation.derivative2D(lastZ[0][f]);
                        for (int y = 0; y < outHeight; y += stride) {
                                for (int x = 0; x < outWidth; x += stride) {
                                        double delta = gradOutput[f][y][x];
                                        if (actDeriv.length > y && actDeriv[0].length > x) {
                                                delta *= actDeriv[y][x];
                                        }
                                        biases[f] -= lr * delta;
                                        for (int d = 0; d < depth; d++) {
                                                for (int i = 0; i < kernel; i++) {
                                                        for (int j = 0; j < kernel; j++) {
                                                                int inY = y + i;
                                                                int inX = x + j;
                                                                if (inY >= 0 && inY < height && inX >= 0 && inX < width) {
                                                                        gradInput[d][inY][inX] += filters[f][d][i][j] * delta;
                                                                        filters[f][d][i][j] -= lr * delta * input[d][inY][inX];
                                                                }
                                                        }
                                                }
                                        }
                                }
                        }
                }

                return gradInput;
        }
}
