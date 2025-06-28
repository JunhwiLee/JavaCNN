package CNN;

/**
 * Rectified Linear Unit activation function.
 */
public class ReLU implements ActivationFunc {
	
	@Override
	public double[] activate(double[] x) {
		double[] out = new double[x.length];
		for(int i = 0; i<x.length; i++) {
			out[i] = Math.max(0.0, x[i]);
		}
		return out;
	}
	
	@Override
	public double[] derivative(double[] x) {
		double[] out = new double[x.length];
		for(int i = 0; i<x.length; i++) {
			out[i] = x[i] > 0.0 ? 1.0 : 0.0;
		}
		return out;
	}
}
