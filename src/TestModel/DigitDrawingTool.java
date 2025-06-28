package TestModel;

import javax.swing.*;
import java.awt.*;
import java.awt.event.*;
import java.awt.image.BufferedImage;
import java.io.*;

/**
 * Simple GUI to create 256x256 digit images for training.
 * The drawn image is saved as a text file containing 0s and 1s
 * where 1 represents a black pixel. Images are stored in a
 * directory named after the selected digit label.
 */
public class DigitDrawingTool extends JFrame {
    private static final int SIZE = 256;
    private final BufferedImage canvas;
    private final JPanel canvasPanel;
    private final ButtonGroup group;
    private final String dataDir = "data";

    public DigitDrawingTool() {
        super("Digit Drawing Tool");
        setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        setLayout(new BorderLayout());

        canvas = new BufferedImage(SIZE, SIZE, BufferedImage.TYPE_BYTE_BINARY);
        clearCanvas();

        canvasPanel = new JPanel() {
            @Override
            protected void paintComponent(Graphics g) {
                super.paintComponent(g);
                g.drawImage(canvas, 0, 0, null);
            }
        };
        canvasPanel.setPreferredSize(new Dimension(SIZE, SIZE));
        canvasPanel.addMouseMotionListener(new MouseMotionAdapter() {
            @Override
            public void mouseDragged(MouseEvent e) {
                Graphics2D g2 = canvas.createGraphics();
                g2.setColor(Color.BLACK);
                int x = e.getX();
                int y = e.getY();
                g2.fillOval(x, y, 8, 8);
                g2.dispose();
                canvasPanel.repaint();
            }
        });
        add(canvasPanel, BorderLayout.CENTER);

        JPanel controls = new JPanel();
        group = new ButtonGroup();
        for (int i = 0; i < 10; i++) {
            JRadioButton btn = new JRadioButton(Integer.toString(i));
            btn.setActionCommand(Integer.toString(i));
            group.add(btn);
            controls.add(btn);
        }

        JButton save = new JButton("Save");
        save.addActionListener(e -> saveImage());
        controls.add(save);

        JButton clear = new JButton("Clear");
        clear.addActionListener(e -> clearCanvas());
        controls.add(clear);

        add(controls, BorderLayout.SOUTH);
        pack();
        setLocationRelativeTo(null);
        setVisible(true);
    }

    private void clearCanvas() {
        Graphics2D g = canvas.createGraphics();
        g.setColor(Color.WHITE);
        g.fillRect(0, 0, SIZE, SIZE);
        g.dispose();
        if (canvasPanel != null) canvasPanel.repaint();
    }

    private void saveImage() {
        if (group.getSelection() == null) {
            JOptionPane.showMessageDialog(this, "Select a digit label first.");
            return;
        }
        String label = group.getSelection().getActionCommand();
        File dir = new File(dataDir, label);
        dir.mkdirs();
        File file = new File(dir, "image_" + System.currentTimeMillis() + ".txt");

        try (PrintWriter pw = new PrintWriter(new FileWriter(file))) {
            for (int y = 0; y < SIZE; y++) {
                for (int x = 0; x < SIZE; x++) {
                    int rgb = canvas.getRGB(x, y) & 0xFFFFFF;
                    int val = (rgb == 0x000000) ? 1 : 0;
                    pw.print(val);
                    if (x < SIZE - 1) pw.print(' ');
                }
                pw.println();
            }
        } catch (IOException ex) {
            JOptionPane.showMessageDialog(this, "Failed to save image: " + ex.getMessage());
            return;
        }
        clearCanvas();
    }

    public static void main(String[] args) {
        SwingUtilities.invokeLater(DigitDrawingTool::new);
    }
}
