package CNN;

/**
 * Rectified Linear Unit activation function.
 */
public class ReLU implements ActivationFunc {

    @Override
    public double activate(double x) {
        return Math.max(0.0, x);
    }

    @Override
    public double derivative(double x) {
        return x > 0.0 ? 1.0 : 0.0;
    }
}

