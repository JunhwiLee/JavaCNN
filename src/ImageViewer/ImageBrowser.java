package ImageViewer;

import java.awt.Color;
import java.awt.Graphics;
import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import javax.swing.BorderFactory;
import javax.swing.JFrame;
import javax.swing.JPanel;

public class ImageBrowser {
    public static void main(String[] args) {
        JFrame f = new JFrame("Image Browser");
        f.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        // 패널 생성 시 파일 경로를 전달하여 한 번만 이미지 로드
        DrawSquarePanel p = new DrawSquarePanel(
            "C:/Users/jhlee/eclipse-workspace/CNN/data/4/image_1751203094365.txt"
        );
        f.add(p);
        f.setSize(600, 600);
        f.setLocationRelativeTo(null);
        f.setVisible(true);
    }
}

class DrawSquarePanel extends JPanel {
    private static final int START_X = 50;
    private static final int START_Y = 50;
    private static final int CELL_SIZE = 1;
    private int[][] pixels;

    public DrawSquarePanel(String filepath) {
        setBorder(BorderFactory.createLineBorder(Color.black));
        setBackground(Color.WHITE); // 배경을 흰색으로 설정
        loadImage(filepath);
    }

    private void loadImage(String filepath) {
        List<int[]> rows = new ArrayList<>();
        try (BufferedReader br = new BufferedReader(new FileReader(filepath))) {
            String line;
            while ((line = br.readLine()) != null) {
                String[] tokens = line.trim().split("\\s+");
                int[] row = new int[tokens.length];
                for (int i = 0; i < tokens.length; i++) {
                    String t = tokens[i];
                    char c = t.charAt(0);
                    // '０'(U+FF10)과 '１'(U+FF11) 처리
                    if (c == '0') {
                        row[i] = 0;
                    } else if (c == '1') {
                        row[i] = 1;
                    } else {
                        // 일반 ASCII 숫자로 파싱
                        row[i] = Character.getNumericValue(c);
                    }
                }
                rows.add(row);
            }
            pixels = rows.toArray(new int[0][]);
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    @Override
    protected void paintComponent(Graphics g) {
        super.paintComponent(g);
        if (pixels == null) return;
        g.setColor(Color.BLACK);
        g.fillRect(50, 50, CELL_SIZE, CELL_SIZE);
        for (int row = 0; row < pixels.length; row++) {
            for (int col = 0; col < pixels[row].length; col++) {
                // 1인 경우에만 픽셀 그리기
            	//System.out.print(pixels[row][col]);
                if(pixels[row][col] == 1) {
                    int x = START_X + col * CELL_SIZE;
                    int y = START_Y + row * CELL_SIZE;
                    //System.out.println(x + " " + y);
                    g.fillRect(x, y, CELL_SIZE, CELL_SIZE);
                }
            }
            //System.out.println();
        }
    }
}
