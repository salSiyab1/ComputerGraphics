import javax.swing.*;
import java.awt.*;
import java.awt.event.*;
import java.util.ArrayList;

public class BezierEditor extends JPanel {
    private final ArrayList<Point> controlPoints = new ArrayList<>();
    private Point draggedPoint = null;
    private static final int RADIUS = 6;

    public BezierEditor() {
        setPreferredSize(new Dimension(800, 600));
        setBackground(Color.WHITE);

        MouseAdapter mouseAdapter = new MouseAdapter() {
            @Override
            public void mousePressed(MouseEvent e) {
                for (Point p : controlPoints) {
                    if (p.distance(e.getPoint()) < RADIUS * 2) {
                        draggedPoint = p;
                        return;
                    }
                }
                controlPoints.add(e.getPoint());
                repaint();
            }

            @Override
            public void mouseReleased(MouseEvent e) {
                draggedPoint = null;
            }

            @Override
            public void mouseDragged(MouseEvent e) {
                if (draggedPoint != null) {
                    draggedPoint.setLocation(e.getPoint());
                    repaint();
                }
            }
        };

        addMouseListener(mouseAdapter);
        addMouseMotionListener(mouseAdapter);
    }

    private Point computeBezierPoint(double t, ArrayList<Point> points) {
        ArrayList<Point> temp = new ArrayList<>(points);
        while (temp.size() > 1) {
            ArrayList<Point> next = new ArrayList<>();
            for (int i = 0; i < temp.size() - 1; i++) {
                int x = (int) ((1 - t) * temp.get(i).x + t * temp.get(i + 1).x);
                int y = (int) ((1 - t) * temp.get(i).y + t * temp.get(i + 1).y);
                next.add(new Point(x, y));
            }
            temp = next;
        }
        return temp.get(0);
    }

    @Override
    protected void paintComponent(Graphics g) {
        super.paintComponent(g);

        // Draw control points
        g.setColor(Color.BLUE);
        for (Point p : controlPoints) {
            g.fillOval(p.x - RADIUS, p.y - RADIUS, RADIUS * 2, RADIUS * 2);
        }

        // Draw control lines
        g.setColor(Color.LIGHT_GRAY);
        for (int i = 0; i < controlPoints.size() - 1; i++) {
            g.drawLine(controlPoints.get(i).x, controlPoints.get(i).y,
                    controlPoints.get(i + 1).x, controlPoints.get(i + 1).y);
        }

        // Draw Bezier curve
        if (controlPoints.size() >= 2) {
            g.setColor(Color.RED);
            for (double t = 0; t <= 1.0; t += 0.01) {
                Point p1 = computeBezierPoint(t, controlPoints);
                Point p2 = computeBezierPoint(t + 0.01, controlPoints);
                g.drawLine(p1.x, p1.y, p2.x, p2.y);
            }
        }
    }

    public static void main(String[] args) {
        JFrame frame = new JFrame("Bezier Curve Editor");
        frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        frame.getContentPane().add(new BezierEditor());
        frame.pack();
        frame.setLocationRelativeTo(null);
        frame.setVisible(true);
    }
}
