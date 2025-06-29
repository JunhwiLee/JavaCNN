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
	
	public int getLayerSize() {
		return nodes.length;
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
         * Backward pass for the layer.
         *
         * @param input       the input vector that was fed to this layer in the forward pass
         * @param gradOutput  gradient with respect to the activated output of this layer
         * @param lr          learning rate used for parameter updates
         * @return gradient with respect to the input vector
         */
        public double[] backward(double[] input, double[] gradOutput, double lr) {
                if (gradOutput.length != nodes.length) {
                        throw new InputSizeMissmatchException(nodes.length, gradOutput.length);
                }

                // Compute derivative of activation for each node
                double[] z = new double[nodes.length];
                for (int i = 0; i < nodes.length; i++) {
                        z[i] = nodes[i].getLastZ();
                }
                double[] actDeriv = activation.derivative(z);

                double[] delta = new double[nodes.length];
                for (int i = 0; i < nodes.length; i++) {
                        delta[i] = gradOutput[i] * actDeriv[i];
                }

                double[] gradInput = new double[input.length];
                for (int i = 0; i < nodes.length; i++) {
                        double[] nodeGrad = nodes[i].backward(input, delta[i], lr);
                        for (int j = 0; j < gradInput.length; j++) {
                                gradInput[j] += nodeGrad[j];
                        }
                }
                return gradInput;
        }
}

