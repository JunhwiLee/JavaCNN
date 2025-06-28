package TestModel;
import CNN.*;

public class ImageTest {
	public static void main(String[] args) {
		Convolution2D cnn = new Convolution2D(3, new int[] {1, 4, 16}, 32, 3, new int[] {1, 1, 1}, 2, new ReLU());
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
		
		for(double d : cnn.forward(tensor)) {
			System.out.print(d + " ");
		}
	}
}
