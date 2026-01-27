import mpi.MPI;

import java.util.Arrays;
import java.util.stream.IntStream;

public class BoundingBox {
    double[][] minMaxPerDimension;
    int numOfDimensions;
    int globalCommGroupAddress;
    double epsilon;


    public BoundingBox(int numOfDimensions, double epsilon) {
        minMaxPerDimension = new double[numOfDimensions][2];
        this.globalCommGroupAddress = MPI.COMM_WORLD.Rank();
        this.numOfDimensions = numOfDimensions;
        this.epsilon = epsilon;

        for (int i = 0; i < numOfDimensions; i++) {
            minMaxPerDimension[i][0] = Double.MIN_VALUE;
            minMaxPerDimension[i][1] = Double.MAX_VALUE;
        }
    }

    private BoundingBox(int globalCommGroupAddress, int numOfDimensions, double[][] minMaxPerDimension, double epsilon) {
        this.globalCommGroupAddress = globalCommGroupAddress;
        this.numOfDimensions = numOfDimensions;
        this.minMaxPerDimension = minMaxPerDimension;
        this.epsilon = epsilon;
    }

    boolean isNeighbour(BoundingBox other) {
        for (int i = 0; i < numOfDimensions; i++) {
            double lowerBound = Math.min(minMaxPerDimension[i][0], minMaxPerDimension[i][0] - epsilon);
            double upperBound = Math.max(minMaxPerDimension[i][1], minMaxPerDimension[i][1] + epsilon);
            double otherMin = other.minMaxPerDimension[i][0];
            double otherMax = other.minMaxPerDimension[i][1];
            boolean overlap = (lowerBound <= otherMax && upperBound >= otherMin);
            if (!overlap) {
                return false;
            }
        }
        return true;
    }

    public void send(int dest, int tag) {
        int[] header = {globalCommGroupAddress, numOfDimensions};
        MPI.COMM_WORLD.Send(header, 0, 2, MPI.INT, dest, tag);
        double[] sendBuffer = new double[(numOfDimensions*2)+1];
        sendBuffer[0] = epsilon;
        System.arraycopy(
                Arrays.stream(minMaxPerDimension)
                        .flatMapToDouble(Arrays::stream)
                        .toArray(),
                0, sendBuffer, 1, numOfDimensions*2);

        MPI.COMM_WORLD.Send(
                sendBuffer,
                0, (numOfDimensions * 2)+1, MPI.DOUBLE, dest, tag);
    }

    public static BoundingBox receive(int source, int tag) {
        int[] headerReceiver = new int[2];
        MPI.COMM_WORLD.Recv(headerReceiver, 0, 2, MPI.INT, source, tag);
        int receiveSize = headerReceiver[1]*2;
        double[] minMaxReceiver = new double[receiveSize];
        MPI.COMM_WORLD.Recv(minMaxReceiver, 0, receiveSize+1, MPI.DOUBLE, source, tag);
        double epsilon = minMaxReceiver[0];
        double[][] newMinMax = IntStream.range(0, receiveSize / 2)
                .mapToObj(i -> new double[]{minMaxReceiver[(i * 2)+1], minMaxReceiver[(i+1) * 2]})
                .toArray(double[][]::new); //turns data back into a pair of min and max per dimension

        return new BoundingBox(headerReceiver[0], headerReceiver[1], newMinMax, epsilon);
    }

    public void setMin(int dimension, double newMin) {
        minMaxPerDimension[dimension][0] = newMin;
    }
    public void setMax(int dimension, double newMax) {
        minMaxPerDimension[dimension][1] = newMax;
    }

}
