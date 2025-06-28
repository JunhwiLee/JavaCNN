package CNN;

/**
 * Represents a fully connected neural network layer that performs the
 * linear transformation of its inputs. The activation function is applied
 * separately using {@link #activation(double[])}.
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
	 * Executes the linear forward pass through this layer.
	 *
	 * @param input values from the previous layer
	 * @return unactivated outputs of this layer
	 */
	public double[] forward(double[] input) {
		double[] out = new double[nodes.length];
		for(int i = 0; i < nodes.length; i++) {
			out[i] = nodes[i].forward(input);
		}
		return out;
	}
	
	/**
	 * Applies the configured activation function to each value.
	 *
	 * @param input raw outputs from {@link #forward(double[])}
	 * @return activated outputs of this layer
	 */
	public double[] activation(double[] input) {
		double[] out = new double[nodes.length];
		out = activation.activate(input);
		return out;
	}
	
	/**
	 * Propagates errors backward through this layer and updates its parameters.
	 *
	 * @param input        inputs that produced the outputs of this layer
	 * @param gradOutput   gradient of the loss with respect to this layer's outputs
	 * @param learningRate step size for gradient descent
	 * @return gradient of the loss with respect to the layer inputs
	 */
	public double[] backward(double[] input, double[] gradOutput, double learningRate) {
		double[] gradInput = new double[nodes[0].inputSize()];
		double[] LastZs = new double[nodes.length];
		for(int i = 0; i<nodes.length; i++) {
			LastZs[i] = nodes[i].getLastZ();
		}
		double[] activate = activation.derivative(LastZs);
		for(int i = 0; i < nodes.length; i++) {
			double delta = gradOutput[i] * activate[i];
			nodes[i].backward(input, delta, learningRate);
			for(int j = 0; j < gradInput.length; j++) {
				gradInput[j] += delta * nodes[i].getWeight(j);
			}
		}
		return gradInput;
	}
}

