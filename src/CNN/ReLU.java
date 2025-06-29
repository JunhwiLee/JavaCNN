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
	public double[][] activate2D(double[][] x) {
		double[][] out = new double[x.length][x[0].length];
		for(int i = 0; i<x.length; i++) {
			for(int j = 0; j<x[0].length; j++) {
				out[i][j] = Math.max(0.0, x[i][j]);				
			}
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
	
	@Override
	public double[][] derivative2D(double[][] x){
                double[][] out = new double[x.length][x[0].length];
                for(int i = 0; i<x.length; i++) {
                        for(int j = 0; j<x[0].length; j++) {
                                out[i][j] = x[i][j] > 0.0 ? 1.0 : 0.0;
                        }
                }
                return out;
        }
}
