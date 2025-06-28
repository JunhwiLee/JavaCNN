package CNN;

/**
 * Represents a fully connected neural network layer.
 */
public class Layer {
	
	private final Node[] nodes;
	private final ActivationFunc activation;
	
	/**
	 * Creates a layer with the given number of nodes.
	 *
	 * @param inputSize  number of inputs each node receives
	 * @param nodeCount  number of nodes in this layer
	 * @param activation activation function to apply
	 */
	public Layer(int inputSize, int nodeCount, ActivationFunc activation) {
		this.activation = activation;
		this.nodes = new Node[nodeCount];
		for(int i = 0; i < nodeCount; i++) {
			nodes[i] = new Node(inputSize);
		}
	}
	
	/**
	 * Executes a forward pass through the layer.
	 *
	 * @param input values from the previous layer
	 * @return outputs of this layer
	 */
	public double[] forward(double[] input) {
		double[] out = new double[nodes.length];
		for(int i = 0; i < nodes.length; i++) {
			out[i] = nodes[i].forward(input);
		}
		return out;
	}
	
	/**
	 * Executes a activation function through the layer.
	 *
	 * @param input values from the forwarded vector
	 * @return activations of this layer
	 */
	public double[] activation(double[] input) {
		double[] out = new double[nodes.length];
		for(int i = 0; i < nodes.length; i++) {
			out[i] = activation.activate(input[i]);
		}
		return out;
	}
	
	/**
	 * Backpropagates errors through this layer and updates its parameters.
	 *
	 * @param input        inputs that produced the outputs of this layer
	 * @param gradOutput   gradient of the loss with respect to this layer's outputs
	 * @param learningRate step size for gradient descent
	 * @return gradient of the loss with respect to the layer inputs
	 */
	public double[] backward(double[] input, double[] gradOutput, double learningRate) {
		double[] gradInput = new double[nodes[0].inputSize()];
		for(int i = 0; i < nodes.length; i++) {
			double delta = gradOutput[i] * activation.derivative(nodes[i].getLastZ());
			nodes[i].backward(input, delta, learningRate);
			for(int j = 0; j < gradInput.length; j++) {
				gradInput[j] += delta * nodes[i].getWeight(j);
			}
		}
		return gradInput;
	}
}

