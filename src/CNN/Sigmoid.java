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
	public double[][] activate2D(double[][] x){
		double[][] out = new double[x.length][x[0].length];
		for(int i = 0; i<x.length; i++) {
			for(int j = 0; j<x[0].length; i++) {
				out[i][j] = 1.0 / (1.0 + Math.exp(-x[i][j]));			
			}
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
	
	@Override
	public double[][] derivative2D(double[][] x){
		double[][] out = new double[x.length][x[0].length];
		double[][] y = activate2D(x);
		for(int i = 0; i<x.length; i++) {
			for(int j = 0; j<x[0].length; i++) {
				out[i][j] = y[i][j] * (1.0 - y[i][j]);			
			}
		}
		return out;
	}
}
