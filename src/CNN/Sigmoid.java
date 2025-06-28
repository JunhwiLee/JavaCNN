package CNN;

/**
 * Standard sigmoid activation function.
 */
public class Sigmoid implements ActivationFunc {
	
	@Override
	public double[] activate(double[] x) {
		double[] out = new double[x.length];
		for(int i = 0; i<x.length; i++) {
			out[i] = 1.0 / (1.0 + Math.exp(-x[i]));
		}
		return out;
	}
	
	@Override
	public double[] derivative(double[] x) {
		double[] out = new double[x.length];
		double[] y = activate(x);
		for(int i = 0; i<x.length; i++) {
			out[i] = y[i] * (1.0 - y[i]);
		}
		return out;
	}
}
