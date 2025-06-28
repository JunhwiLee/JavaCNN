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
                new ReLU());

        Classification model = new Classification(
                32,
                2,
                new int[]{32, 16},
                10,
                new ReLU());

        ConvolutionNeuralNetwork net = new ConvolutionNeuralNetwork(conv, model);
        net.learning(data.images, data.labels);
    }
}
