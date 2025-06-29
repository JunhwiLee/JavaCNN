package TestModel;
import CNN.*;

public class ImageTest {
	public static void main(String[] args) {
		Convolution2D convolution = new Convolution2D(3, new int[] {1, 4, 16}, 32, 3, new int[] {1, 1, 1}, 2, new ReLU());
		Model classification = new Classification(32, 1, new int[] {32}, 10, new ReLU());
		
		ConvolutionNeuralNetwork cnn = new ConvolutionNeuralNetwork(convolution, classification);
		
		double[][][] tensor =
			{{{0, 0, 0, 0, 0, 0, 0, 0},
			  {0, 0, 0, 1, 1, 1, 0, 0},
			  {0, 0, 1, 0, 0, 0, 0, 0},
			  {0, 1, 1, 1, 1, 0, 0, 0},
			  {0, 1, 0, 0, 0, 1, 0, 0},
			  {0, 1, 0, 0, 0, 1, 0, 0},
			  {0, 0, 1, 1, 1, 0, 0, 0},
			  {0, 0, 0, 0, 0, 0, 0, 0},
			}};
		
		for(double d : cnn.estimate(tensor)) {
			System.out.print(d + " ");
		}
	}
}
