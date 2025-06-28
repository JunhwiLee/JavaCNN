package TestModel;
import CNN.*;
public class Test1 {
	public static void main(String[] args) {
		int[] arr = {8, 8, 8};
		Classification model = new Classification(8, 3, arr, 3, new ReLU());
		double[] result = model.forword(new double[] {1, 2, 3, 4, 5, 6, 7, 8});
		for(double d : result) System.out.print(d + " ");
	}
}
