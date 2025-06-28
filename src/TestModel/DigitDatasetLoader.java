package TestModel;

import java.io.*;
import java.util.*;

/**
 * Utility class to load digit images stored as text files.
 * Files should be placed under a directory named by the digit label
 * (e.g. data/6/image_123.txt). Each text file must contain 256 lines of
 * space separated 0/1 values representing pixel intensities.
 */
public class DigitDatasetLoader {
    public static class Data {
        public final double[][][][] images; // [sample][channel][h][w]
        public final double[][][][] labels; // [sample][1][1][10]
        public Data(double[][][][] i, double[][][][] l) {
            images = i; labels = l;
        }
    }

    public static Data load(String root) throws IOException {
        List<double[][][]> imgs = new ArrayList<>();
        List<double[][][]> labs = new ArrayList<>();
        for (int label = 0; label <= 9; label++) {
            File dir = new File(root, Integer.toString(label));
            if (!dir.isDirectory()) continue;
            File[] files = dir.listFiles();
            if (files == null) continue;
            for (File f : files) {
                if (!f.isFile()) continue;
                double[][][] img = readImage(f);
                imgs.add(img);
                double[][][] lab = new double[1][1][10];
                for (int i = 0; i < 10; i++) lab[0][0][i] = (i == label) ? 1.0 : 0.0;
                labs.add(lab);
            }
        }
        int n = imgs.size();
        double[][][][] images = new double[n][1][256][256];
        double[][][][] labels = new double[n][1][1][10];
        for (int i = 0; i < n; i++) {
            images[i] = imgs.get(i);
            labels[i] = labs.get(i);
        }
        return new Data(images, labels);
    }

    private static double[][][] readImage(File file) throws IOException {
        List<String> lines = java.nio.file.Files.readAllLines(file.toPath());
        double[][][] out = new double[1][256][256];
        for (int y = 0; y < 256 && y < lines.size(); y++) {
            String[] tokens = lines.get(y).trim().split("\\s+");
            for (int x = 0; x < 256 && x < tokens.length; x++) {
                out[0][y][x] = Double.parseDouble(tokens[x]);
            }
        }
        return out;
    }
}
