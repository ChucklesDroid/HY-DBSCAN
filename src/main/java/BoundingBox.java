import mpi.MPI;

import java.util.Arrays;
import java.util.stream.IntStream;

public class BoundingBox {
    double[][] minMaxPerDimension;
    int numOfDimensions;
    int globalCommGroupAddress;


    public BoundingBox(double[][] minMaxPerDimension) {
        this.minMaxPerDimension = minMaxPerDimension;
        this.globalCommGroupAddress = MPI.COMM_WORLD.Rank();
        this.numOfDimensions = minMaxPerDimension.length;
    }

    private BoundingBox(int globalCommGroupAddress, int numOfDimensions, double[][] minMaxPerDimension) {
        this.globalCommGroupAddress = globalCommGroupAddress;
        this.numOfDimensions = numOfDimensions;
        this.minMaxPerDimension = minMaxPerDimension;
    }

    boolean isNeighbour(BoundingBox other) {
        return false;
        //TODO
    }

    public void send(int dest, int tag) {
        int[] header = {globalCommGroupAddress, numOfDimensions};
        MPI.COMM_WORLD.Send(header, 0, 2, MPI.INT, dest, tag);
        MPI.COMM_WORLD.Send(
                Arrays.stream(minMaxPerDimension)
                        .flatMapToDouble(Arrays::stream)
                        .toArray(),
                0, numOfDimensions * 2, MPI.DOUBLE, dest, tag);
    }

    public static BoundingBox receive(int source, int tag) {
        int[] headerReceiver = new int[2];
        MPI.COMM_WORLD.Recv(headerReceiver, 0, 2, MPI.INT, source, tag);
        int receiveSize = headerReceiver[1]*2;
        double[] minMaxReceiver = new double[receiveSize];
        MPI.COMM_WORLD.Recv(minMaxReceiver, 0, receiveSize, MPI.DOUBLE, source, tag);
        double[][] newMinMax = IntStream.range(0, receiveSize / 2)
                .mapToObj(i -> new double[]{minMaxReceiver[i * 2], minMaxReceiver[i * 2 + 1]})
                .toArray(double[][]::new); //turns data back into a pair of min and max per dimension

        return new BoundingBox(headerReceiver[0], headerReceiver[1], newMinMax);
    }

    public void setMin(int dimension, double newMin) {
        minMaxPerDimension[dimension][0] = newMin;
    }
    public void setMax(int dimension, double newMax) {
        minMaxPerDimension[dimension][1] = newMax;
    }

}
