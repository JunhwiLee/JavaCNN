package CNN;

import java.util.Random;

/**
 * A simple fully connected neuron that stores its weights and bias.
 */
public class Node {
	
	private final double[] weights;
	private double bias;
	private double lastZ; // weighted sum before activation
	
	/**
	 * Creates a node with weights initialized randomly using a small
	 * Gaussian distribution.
	 *
	 * @param inputSize number of inputs that connect to this node
	 */
	public Node(int inputSize) {
		this.weights = new double[inputSize];
		Random rnd = new Random();
		for (int i = 0; i < inputSize; i++) {
			// small random initial weights
			this.weights[i] = rnd.nextGaussian() * 0.01;
		}
		this.bias = 0.0;
	}
	
	/**
	 * Forward pass for the node. Returns the weighted sum before activation
	 * and stores it for use during backpropagation.
	 */
	public double forward(double[] input) {
		if (input.length != weights.length) {
			throw new InputSizeMissmatchException(weights.length, input.length);
		}
		double sum = bias;
		for (int i = 0; i < weights.length; i++) {
			sum += weights[i] * input[i];
		}
		lastZ = sum;
		return sum;
	}
	
	/**
	 * Performs a simple weight update using the provided delta.
	 *
	 * @param input        the input vector that produced this node's output
	 * @param delta        gradient of the loss with respect to this node's pre-activation value
	 * @param learningRate step size for gradient descent
	 */
	public void backward(double[] input, double delta, double learningRate) {
		if (input.length != weights.length) {
			throw new InputSizeMissmatchException(weights.length, input.length);
		}
		for (int i = 0; i < weights.length; i++) {
			weights[i] -= learningRate * delta * input[i];
		}
		bias -= learningRate * delta;
	}
	
	public double getLastZ() {
		return lastZ;
	}
	
	public int inputSize() {
		return weights.length;
	}
	
	public double getWeight(int index) {
		return weights[index];
	}
}

