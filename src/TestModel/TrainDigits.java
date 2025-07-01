package TestModel;

import CNN.*;
import java.io.IOException;

/**
 * Example program that loads digit images from labeled directories and
 * trains a simple convolutional neural network.
 */
public class TrainDigits {
    public static void main(String[] args) throws IOException {
        // Load dataset
        DigitDatasetLoader.Data data = DigitDatasetLoader.load("data");

        Convolution2D conv = new Convolution2D(
                3,
                new int[]{1, 4, 16},
                32,
                3,
                new int[]{1, 1, 1},
                2,
                new ReLU(), 16);

        Classification model = new Classification(
                32,
                2,
                new int[]{32, 16},
                10,
                new ReLU(), 16);

        ConvolutionNeuralNetwork net = new ConvolutionNeuralNetwork(conv, model);

        double[] labels = new double[data.labels.length];
        int[] ilabels = new int[data.labels.length];
        for (int i = 0; i < data.labels.length; i++) {
            for (int j = 0; j < data.labels[i][0][0].length; j++) {
                if (data.labels[i][0][0][j] == 1.0) {
                    labels[i] = j;
                    ilabels[i] = j;
                    break;
                }
            }
        }

        net.learning(data.images, labels, data.images, ilabels, 32);
        System.out.println("Learning over");
    }
}
