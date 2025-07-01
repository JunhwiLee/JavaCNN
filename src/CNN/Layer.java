package CNN;

/**
 * Represents a fully connected neural network layer that performs the
 * linear transformation of its inputs. The activation function is applied
 * separately using {@link #activation(double[])}.
 */
public class Layer {
	
	private final Node[] nodes;
	private final ActivationFunc activation;
	private final int batch;
	
	private double[][] lastZs;
	
	/**
	 * Creates a layer with the given number of nodes.
	 *
	 * @param inputSize  number of inputs each node receives
	 * @param nodeCount  number of nodes in this layer
	 * @param activation activation function to apply
	 */
	public Layer(int inputSize, int nodeCount, ActivationFunc activation, int batch) {
		this.activation = activation;
		this.nodes = new Node[nodeCount];
		lastZs = new double[batch][nodeCount];
		this.batch = batch;
		for(int i = 0; i < nodeCount; i++) {
			nodes[i] = new Node(inputSize, batch);
		}
	}
	
	public int getLayerSize() {
		return nodes.length;
	}
	
	public double[][] getLastZ() {
		return lastZs;
	}
	
	/**
	 * Executes the linear forward pass through this layer.
	 *
	 * @param input values from the previous layer
	 * @return unactivated outputs of this layer
	 */
	
	public double[] estimate(double[] input) {
		double[] out = new double[nodes.length];
		for(int i = 0; i < nodes.length; i++) {
			out[i] = nodes[i].estimate(input);
		}
		return out;
	}
	
	public double[][] forward(double[][] input) {
		double[][] out = new double[batch][nodes.length];
		
		for(int b = 0; b<batch; b++) {
			out[b] = estimate(input[b]);
		}
		
		lastZs = out;
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

                double[] gradInput = new double[input.length];
                double[] actDeriv = activation.derivative(lastZs[0]);

                for (int n = 0; n < nodes.length; n++) {
                        double delta = gradOutput[n];
                        if (actDeriv.length > n) {
                                delta *= actDeriv[n];
                        }
                        double[] nodeGrad = nodes[n].backward(input, delta, lr);
                        for (int i = 0; i < gradInput.length; i++) {
                                gradInput[i] += nodeGrad[i];
                        }
                }

                return gradInput;
        }
}
