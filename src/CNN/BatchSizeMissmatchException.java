package CNN;

public class BatchSizeMissmatchException extends RuntimeException{
	public BatchSizeMissmatchException(String message) {
		super(message);
	}

	public BatchSizeMissmatchException(int expected, int actual) {
		super("Expected layer count " + expected + " but got " + actual);
	}
	
	public BatchSizeMissmatchException() {
		super();
	}
}