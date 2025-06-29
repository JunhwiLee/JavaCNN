package ImageGenerater;
import java.awt.*;
import java.awt.event.*;
import java.awt.image.BufferedImage;
import javax.swing.*;

public class DrawDigit extends JFrame {
    private static final int SIZE = 512;
    private Color currentColor = Color.BLACK;
    private final DrawPanel drawPanel;

    public DrawDigit() {
        super("Draw Digit");
        drawPanel = new DrawPanel(SIZE, SIZE);
        initUI();
    }

    private void initUI() {
        setLayout(new BorderLayout());

        // Toolbar
        JPanel toolbar = new JPanel();
        JButton colorButton = new JButton("색상 선택");
        colorButton.addActionListener(e -> {
            Color chosen = JColorChooser.showDialog(this, "색상을 선택하세요", currentColor);
            if (chosen != null) currentColor = chosen;
        });
        JButton clearButton = new JButton("지우기");
        clearButton.addActionListener(e -> drawPanel.clear());
        toolbar.add(colorButton);
        toolbar.add(clearButton);
        add(toolbar, BorderLayout.NORTH);

        // Drawing area
        JScrollPane scroll = new JScrollPane(drawPanel);
        add(scroll, BorderLayout.CENTER);

        setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        pack();
        setLocationRelativeTo(null);
        setVisible(true);
    }

    private class DrawPanel extends JPanel implements MouseListener, MouseMotionListener {
        private final BufferedImage canvas;
        private boolean drawing = false;

        public DrawPanel(int width, int height) {
            canvas = new BufferedImage(width, height, BufferedImage.TYPE_INT_ARGB);
            Graphics2D g = canvas.createGraphics();
            g.setColor(Color.WHITE);
            g.fillRect(0, 0, width, height);
            g.dispose();
            setPreferredSize(new Dimension(width, height));
            addMouseListener(this);
            addMouseMotionListener(this);
        }

        @Override
        protected void paintComponent(Graphics g) {
            super.paintComponent(g);
            g.drawImage(canvas, 0, 0, null);
        }

        public void clear() {
            Graphics2D g = canvas.createGraphics();
            g.setColor(Color.WHITE);
            g.fillRect(0, 0, canvas.getWidth(), canvas.getHeight());
            g.dispose();
            repaint();
        }

        private void drawAt(int x, int y) {
            if (x < 0 || y < 0 || x >= canvas.getWidth() || y >= canvas.getHeight()) return;
            canvas.setRGB(x, y, currentColor.getRGB());
            repaint(x, y, 1, 1);
        }

        @Override
        public void mousePressed(MouseEvent e) {
            drawing = true;
            drawAt(e.getX(), e.getY());
        }

        @Override
        public void mouseReleased(MouseEvent e) {
            drawing = false;
        }

        @Override
        public void mouseDragged(MouseEvent e) {
            if (drawing) drawAt(e.getX(), e.getY());
        }

        // Unused
        @Override public void mouseClicked(MouseEvent e) {}
        @Override public void mouseEntered(MouseEvent e) {}
        @Override public void mouseExited(MouseEvent e) {}
        @Override public void mouseMoved(MouseEvent e) {}
    }

    public static void main(String[] args) {
        SwingUtilities.invokeLater(DrawDigit::new);
    }
}
