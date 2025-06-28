package CNN;

/**
 * Thrown to indicate that an operation received an input
 * array whose length does not match the expected size.
 */
public class InputSizeMissmatchException extends RuntimeException {
	
	/**
	 * Constructs a new exception with the specified detail message.
	 *
	 * @param message the detail message
	 */
	public InputSizeMissmatchException(String message) {
		super(message);
	}
	
	/**
	 * Constructs a new exception describing the expected and actual sizes.
	 *
	 * @param expected the expected input length
	 * @param actual   the provided input length
	 */
	public InputSizeMissmatchException(int expected, int actual) {
		super("Expected input size " + expected + " but got " + actual);
	}
	
	/**
	 * Constructs a new exception with no detail message.
	 */
	public InputSizeMissmatchException() {
		super();
	}
}
