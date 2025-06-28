package CNN;

public class Convolution2D {
	private final Convolution2DLayer[] convolutionLayers;
	
	public Convolution2D(int layerCount) {
		convolutionLayers = new Convolution2DLayer[layerCount];
	}
}
