package ImageViewer;

import javax.swing.*;
import java.awt.*;
import java.awt.event.*;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.FilenameFilter;
import java.io.IOException;
import javax.imageio.ImageIO;

/**
 * Simple GUI to browse image folders and delete images.
 */
public class ImageBrowser extends JFrame {
        private File currentDir;
        private final JTextField dirField;
        private final DefaultListModel<File> imageListModel;
        private final JList<File> imageList;
        private final JLabel imageLabel;

        public ImageBrowser() {
                super("Image Browser");
                setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
                setLayout(new BorderLayout());

                // Top panel to choose directory
                JPanel top = new JPanel(new BorderLayout());
                dirField = new JTextField();
                dirField.setEditable(false);
                JButton chooseDir = new JButton("폴더 선택");
                chooseDir.addActionListener(e -> chooseDirectory());
                top.add(dirField, BorderLayout.CENTER);
                top.add(chooseDir, BorderLayout.EAST);
                add(top, BorderLayout.NORTH);

                // Image list on the left
                imageListModel = new DefaultListModel<>();
                imageList = new JList<>(imageListModel);
                imageList.setSelectionMode(ListSelectionModel.SINGLE_SELECTION);
                imageList.addListSelectionListener(e -> {
                        if (!e.getValueIsAdjusting()) displayImage(imageList.getSelectedValue());
                });
                add(new JScrollPane(imageList), BorderLayout.WEST);

                // Image display in center
                imageLabel = new JLabel();
                imageLabel.setHorizontalAlignment(JLabel.CENTER);
                add(new JScrollPane(imageLabel), BorderLayout.CENTER);

                // Delete button at bottom
                JButton deleteBtn = new JButton("삭제");
                deleteBtn.addActionListener(e -> deleteSelected());
                JPanel bottom = new JPanel();
                bottom.add(deleteBtn);
                add(bottom, BorderLayout.SOUTH);

                setSize(800, 600);
                setLocationRelativeTo(null);

                // Try default data directory
                File defaultDir = new File("data");
                if (defaultDir.isDirectory()) {
                        setDirectory(defaultDir);
                }
        }

        private void chooseDirectory() {
                JFileChooser fc = new JFileChooser(currentDir == null ? new File(".") : currentDir);
                fc.setFileSelectionMode(JFileChooser.DIRECTORIES_ONLY);
                if (fc.showOpenDialog(this) == JFileChooser.APPROVE_OPTION) {
                        setDirectory(fc.getSelectedFile());
                }
        }

        private void setDirectory(File dir) {
                currentDir = dir;
                dirField.setText(currentDir.getAbsolutePath());
                loadImages();
        }

        private void loadImages() {
                imageListModel.clear();
                if (currentDir == null) return;
                File[] files = currentDir.listFiles(new FilenameFilter() {
                        @Override
                        public boolean accept(File d, String name) {
                                String lower = name.toLowerCase();
                                return lower.endsWith(".png") || lower.endsWith(".jpg") ||
                                       lower.endsWith(".jpeg") || lower.endsWith(".gif") ||
                                       lower.endsWith(".bmp");
                        }
                });
                if (files != null) {
                        for (File f : files) imageListModel.addElement(f);
                }
        }

        private void displayImage(File file) {
                if (file == null) {
                        imageLabel.setIcon(null);
                        return;
                }
                try {
                        BufferedImage img = ImageIO.read(file);
                        if (img == null) {
                                imageLabel.setIcon(null);
                                return;
                        }
                        Image display = img;
                        if (img.getWidth() > 600 || img.getHeight() > 600) {
                                double scale = Math.min(600.0 / img.getWidth(), 600.0 / img.getHeight());
                                display = img.getScaledInstance((int)(img.getWidth()*scale), (int)(img.getHeight()*scale), Image.SCALE_SMOOTH);
                        }
                        imageLabel.setIcon(new ImageIcon(display));
                } catch (IOException ex) {
                        JOptionPane.showMessageDialog(this, "이미지를 읽을 수 없습니다: " + ex.getMessage());
                }
        }

        private void deleteSelected() {
                File file = imageList.getSelectedValue();
                if (file == null) return;
                int res = JOptionPane.showConfirmDialog(this,
                                file.getName() + " 파일을 삭제하시겠습니까?",
                                "확인", JOptionPane.YES_NO_OPTION);
                if (res == JOptionPane.YES_OPTION) {
                        if (file.delete()) {
                                imageListModel.removeElement(file);
                                imageLabel.setIcon(null);
                        } else {
                                JOptionPane.showMessageDialog(this, "파일을 삭제할 수 없습니다.");
                        }
                }
        }

        public static void main(String[] args) {
                SwingUtilities.invokeLater(() -> new ImageBrowser().setVisible(true));
        }
}
