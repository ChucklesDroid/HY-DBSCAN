import mpi.Intracomm;
import mpi.MPI;
import mpi.Request;

import java.util.ArrayList;
import java.util.List;

public class PointBuffer {
    private double[] data;
    private int dimensions;
    private int numPoints;
    private int[] type;

    public PointBuffer(ArrayList<Point> points) {
        // some processes might have zero points in specific regions.
        if (points == null || points.isEmpty()) {
            this.dimensions = 0;
            this.numPoints = 0;
            this.data = new double[0];
            this.type = new int[0]; // NOTE: 0 means no viable value possible
            return;
        }

        this.dimensions = points.get(0).dimensions;
        this.numPoints = points.size();
        this.data = new double[numPoints * dimensions];
        this.type = new int[numPoints];

        for (int i = 0; i < numPoints; i++) {
            System.arraycopy(points.get(i).coords, 0, data, i * dimensions, dimensions);
        }

        for (int i = 0; i < numPoints; i++) {
            this.type[i] = points.get(i).type;
            
        }
    }

    private PointBuffer(double[] data, int numPoints, int dimensions, int[] type) {
        this.type = type;
        this.data = data;
        this.numPoints = numPoints;
        this.dimensions = dimensions;
    }

    public void send(Intracomm comm, int dest, int tag) {
        // Send Header
        System.out.println("Sending header to process with group address " + dest);
        int[] header = {numPoints, dimensions};
        comm.Send(header, 0, 2, MPI.INT, dest, tag);

        // Send Data and Type
        System.out.println("Sending data to process with group address " + dest);
        comm.Send(data, 0, data.length, MPI.DOUBLE, dest, tag);
        comm.Send(type, 0, type.length, MPI.INT, dest, tag);
    }

    public static PointBuffer receive(Intracomm comm, int source, int tag) {
        // Receive Header
        System.out.println("Receiving header from process with group address " + source);
        int[] header = new int[2];
        comm.Recv(header, 0, 2, MPI.INT, source, tag);

        int numPoints = header[0];
        int dimensions = header[1];

        // Receive Data and Type
        System.out.println("Receiving data from process with group address " + source);
        double[] data = new double[numPoints * dimensions];
        int[] type = new int[numPoints];
        comm.Recv(data, 0, data.length, MPI.DOUBLE, source, tag);
        comm.Recv(type, 0, type.length, MPI.INT, source, tag);

        return new PointBuffer(data, numPoints, dimensions, type);
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
