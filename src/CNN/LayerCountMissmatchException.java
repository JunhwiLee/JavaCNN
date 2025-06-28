package CNN;

public class LayerCountMissmatchException extends RuntimeException{
	public LayerCountMissmatchException(String message) {
		super(message);
	}

	public LayerCountMissmatchException(int expected, int actual) {
		super("Expected layer count " + expected + " but got " + actual);
	}
	
	public LayerCountMissmatchException() {
		super();
	}
}
