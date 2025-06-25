package CNN;

public class Classification implements Model {

    /**
     * Applies the softmax function to the given logits.
     *
     * @param logits raw output scores from the network
     * @return probability distribution
     */
    private static double[] softmax(double[] logits) {
        double max = logits[0];
        for (int i = 1; i < logits.length; i++) {
            if (logits[i] > max) {
                max = logits[i];
            }
        }

        double[] probs = new double[logits.length];
        double sum = 0.0;
        for (int i = 0; i < logits.length; i++) {
            probs[i] = Math.exp(logits[i] - max);
            sum += probs[i];
        }
        for (int i = 0; i < probs.length; i++) {
            probs[i] /= sum;
        }
        return probs;
    }

    /**
     * Computes cross entropy between a predicted probability distribution and
     * the target distribution.
     */
    private static double crossEntropy(double[] probs, double[] target) {
        double ce = 0.0;
        for (int i = 0; i < probs.length; i++) {
            double p = Math.max(Math.min(probs[i], 1.0 - 1e-15), 1e-15);
            ce -= target[i] * Math.log(p);
        }
        return ce;
    }

    /**
     * Computes the Kullback–Leibler divergence between two probability
     * distributions.
     */
    private static double klDivergence(double[] probs, double[] target) {
        double kl = 0.0;
        for (int i = 0; i < probs.length; i++) {
            double t = Math.max(Math.min(target[i], 1.0 - 1e-15), 1e-15);
            double p = Math.max(Math.min(probs[i], 1.0 - 1e-15), 1e-15);
            kl += t * Math.log(t / p);
        }
        return kl;
    }

    /**
     * Computes a combined loss consisting of cross entropy and KL divergence
     * for classification tasks.
     */
    @Override
    public double lossFunc(double[] predicted, double[] target) {
        double[] probs = softmax(predicted);
        double ce = crossEntropy(probs, target);
        double kl = klDivergence(probs, target);
        return ce + kl;
    }
}
