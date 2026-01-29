import mpi.Intracomm;
import mpi.MPI;
import mpi.Request;

import java.util.ArrayList;
import java.util.List;

public class PointBuffer {
    private double[] data;
    private int dimensions;
    private int numPoints;

    public PointBuffer(ArrayList<Point> points) {
        // some processes might have zero points in specific regions.
        if (points == null || points.isEmpty()) {
            this.dimensions = 0;
            this.numPoints = 0;
            this.data = new double[0];
            return;
        }

        this.dimensions = points.get(0).dimensions;
        this.numPoints = points.size();
        this.data = new double[numPoints * dimensions];

        for (int i = 0; i < numPoints; i++) {
            System.arraycopy(points.get(i).coords, 0, data, i * dimensions, dimensions);
        }
    }

    private PointBuffer(double[] data, int numPoints, int dimensions) {
        this.data = data;
        this.numPoints = numPoints;
        this.dimensions = dimensions;
    }

    public void send(Intracomm comm, int dest, int tag) {
        // Send Header
        int[] header = {numPoints, dimensions};
        comm.Send(header, 0, 2, MPI.INT, dest, tag);

        // Send Data
        comm.Send(data, 0, data.length, MPI.DOUBLE, dest, tag);
    }

    public static PointBuffer receive(Intracomm comm, int source, int tag) {
        // Receive Header
        int[] header = new int[2];
        comm.Recv(header, 0, 2, MPI.INT, source, tag);

        int numPoints = header[0];
        int dimensions = header[1];

        // Receive Data
        double[] data = new double[numPoints * dimensions];
        comm.Recv(data, 0, data.length, MPI.DOUBLE, source, tag);

        return new PointBuffer(data, numPoints, dimensions);
    }

    public ArrayList<Point> toPointList() {
        ArrayList<Point> points = new ArrayList<>(numPoints);
        for (int i = 0; i < numPoints; i++) {
            double[] coords = new double[dimensions];
            System.arraycopy(data, i * dimensions, coords, 0, dimensions);
            points.add(new Point(coords));
        }
        return points;
    }

}
