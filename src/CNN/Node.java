package CNN;

import java.util.Random;

/**
 * A simple fully connected neuron that stores its weights and bias.
 */
public class Node {
	
	private double[] weights;
	private double bias;
	private double[] lastZ; // weighted sum before activation
	private final int batch;
	
	/**
	 * Creates a node with weights initialized randomly using a small
	 * Gaussian distribution.
	 *
	 * @param inputSize number of inputs that connect to this node
	 */
	public Node(int inputSize, int batch) {
		this.weights = new double[inputSize];
		this.batch = batch;
		Random rnd = new Random();
		for (int i = 0; i < inputSize; i++) {
			// small random initial weights
			this.weights[i] = rnd.nextGaussian() * 0.01;
		}
		this.bias = 0.0;
	}
	
	public double estimate(double[] input) {
		if (input.length != weights.length) {
			throw new InputSizeMissmatchException(weights.length, input.length);
		}
		double sum = bias;
		for (int i = 0; i < weights.length; i++) {
			sum += weights[i] * input[i];
		}
		return sum;
	}
	
	/**
	 * Forward pass for the node. Returns the weighted sum before activation
	 * and stores it for use during backpropagation.
	 */
	public double[] forward(double[][] input) {
		if (input.length != weights.length) {
			throw new InputSizeMissmatchException(weights.length, input.length);
		}
		
		for(int b = 0; b < batch; b++) {
			lastZ[b] = estimate(input[b]);
		}
		return lastZ;
	}
	
	/**
	 * Backward pass for this node. Updates the weights and bias using
	 * the provided learning rate and returns the gradient with respect
	 * to the node inputs.
	 *
	 * @param input  the inputs that were used during the forward pass
	 * @param delta  gradient of the loss with respect to this node's output
	 * @param lr     learning rate for the weight update
	 * @return gradient with respect to the input vector
	 */
	public double[] backward(double[] input, double delta, double lr) {
		if (input.length != weights.length) {
			throw new InputSizeMissmatchException(weights.length, input.length);
		}
		
		double[] gradInput = new double[weights.length];
		
		for (int i = 0; i < weights.length; i++) {
			gradInput[i] = weights[i] * delta;
			// SGD weight update
			//System.out.println("Parameter Update");
			weights[i] -= lr * delta * input[i];
		}
		bias -= lr * delta;
		return gradInput;
	}
	
	public double[] getLastZ() {
		return lastZ;
	}
	
	public int inputSize() {
		return weights.length;
	}
	
	public double getWeight(int index) {
		return weights[index];
	}
}
