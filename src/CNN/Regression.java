package CNN;

public class Regression implements Model {

    /**
     * Computes the mean squared error between predicted and target values.
     */
    @Override
    public double lossFunc(double[] predicted, double[] target) {
        double sum = 0.0;
        for (int i = 0; i < predicted.length; i++) {
            double diff = predicted[i] - target[i];
            sum += diff * diff;
        }
        return sum / predicted.length;
    }
}
