import mpi.Intracomm;
import mpi.MPI;

import java.util.*;
import java.util.stream.IntStream;

public class BoundingBox {
    double[][] minMaxPerDimension;
    int numOfDimensions;
    int globalCommGroupAddress;


    public BoundingBox(int numOfDimensions) {
        minMaxPerDimension = new double[numOfDimensions][2];
        this.globalCommGroupAddress = MPI.COMM_WORLD.Rank();
        this.numOfDimensions = numOfDimensions;

        for (int i = 0; i < numOfDimensions; i++) {
            minMaxPerDimension[i][0] = Double.MIN_VALUE;
            minMaxPerDimension[i][1] = Double.MAX_VALUE;
        }
    }

    private BoundingBox(int globalCommGroupAddress, int numOfDimensions, double[][] minMaxPerDimension) {
        this.globalCommGroupAddress = globalCommGroupAddress;
        this.numOfDimensions = numOfDimensions;
        this.minMaxPerDimension = minMaxPerDimension;
    }

    boolean isTouching(BoundingBox other) {
        for (int i = 0; i < numOfDimensions; i++) {
            double min = minMaxPerDimension[i][0];
            double max = minMaxPerDimension[i][1];
            double otherMin = other.minMaxPerDimension[i][0];
            double otherMax = other.minMaxPerDimension[i][1];
            if (max < otherMin || otherMax < min) {
                return false;
            }
        }
        return true;
    }

    // public Set<BoundingBox> neighbourSet(Set<BoundingBox> others) {
    //     Set<BoundingBox> ret = new HashSet<>();
    //     for (BoundingBox other : others) {
    //         if (isTouching(other)) {
    //             ret.add(other);
    //             others.remove(other);
    //         }
    //     }
    //     return ret;
    // }

    public Set<BoundingBox> neighbourSet(Set<BoundingBox> others) {
        Set<BoundingBox> ret = new HashSet<>();
        Iterator<BoundingBox> it = others.iterator();
        while (it.hasNext()) {
            BoundingBox other = it.next();
            if (isTouching(other)) {
                ret.add(other);
                it.remove();
            }
        }
        return ret;
    }

    // public void send(int dest, int tag) {
    //     int[] header = {globalCommGroupAddress, numOfDimensions};
    //     MPI.COMM_WORLD.Send(header, 0, 2, MPI.INT, dest, tag);
    //     double[] sendBuffer = Arrays.stream(minMaxPerDimension)
    //             .flatMapToDouble(Arrays::stream)
    //             .toArray();
    //
    //     MPI.COMM_WORLD.Send(
    //             sendBuffer,
    //             0, numOfDimensions * 2, MPI.DOUBLE, dest, tag);
    // }

    public void send(Intracomm comm, int dest, int tag) {
        int[] header = {globalCommGroupAddress, numOfDimensions};
        comm.Send(header, 0, 2, MPI.INT, dest, tag);

        double[] sendBuffer = new double[numOfDimensions + 2];
        for (int i = 0; i < numOfDimensions; i++) {
            sendBuffer[i * 2] = minMaxPerDimension[i][0];
            sendBuffer[i * 2 + 1] = minMaxPerDimension[i][1];
        }
        comm.Send(sendBuffer, 0, sendBuffer.length, MPI.DOUBLE, dest, tag);
    }

    public static BoundingBox receive(int source, int tag) {
        int[] headerReceiver = new int[2];
        MPI.COMM_WORLD.Recv(headerReceiver, 0, 2, MPI.INT, source, tag);
        int receiveSize = headerReceiver[1]*2;
        double[] minMaxReceiver = new double[receiveSize];
        MPI.COMM_WORLD.Recv(minMaxReceiver, 0, receiveSize, MPI.DOUBLE, source, tag);
        double[][] newMinMax = IntStream.range(0, receiveSize / 2)
                .mapToObj(i -> new double[]{minMaxReceiver[i * 2], minMaxReceiver[(i * 2)+1]})
                .toArray(double[][]::new); //turns data back into a pair of min and max per dimension

        return new BoundingBox(headerReceiver[0], headerReceiver[1], newMinMax);
    }

    public void setMin(int dimension, double newMin) {
        minMaxPerDimension[dimension][0] = newMin;
    }
    public void setMax(int dimension, double newMax) {
        minMaxPerDimension[dimension][1] = newMax;
    }

    public double distanceToPoint(Point point){ // euclidian distance
        double distSqr = 0;

        for (int d = 0; d < numOfDimensions; d++) {
            double min = minMaxPerDimension[d][0];
            double max = minMaxPerDimension[d][1];
            double coord = point.coords[d];
            double delta = 0;
            if (coord < min) {
                delta = (min - coord);
            } else if (coord > max) {
                delta = (coord - max);
            } // if none are true, don't add something
            distSqr += Math.pow(delta, 2);
        }
        return Math.sqrt(distSqr);
    }

}
