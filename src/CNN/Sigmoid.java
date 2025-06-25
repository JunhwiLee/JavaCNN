package CNN;

/**
 * Standard sigmoid activation function.
 */
public class Sigmoid implements ActivationFunc {

    @Override
    public double activate(double x) {
        return 1.0 / (1.0 + Math.exp(-x));
    }

    @Override
    public double derivative(double x) {
        double y = activate(x);
        return y * (1.0 - y);
    }
}

