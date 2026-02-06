import mpi.Intracomm;
import mpi.MPI;

import java.util.ArrayList;
import java.util.Map;
import java.util.NavigableMap;
import java.util.TreeMap;

public class Histogram {
    private final NavigableMap<Double, Integer> hist = new TreeMap<>();

    public Histogram(int dimension, ArrayList<Point> points) {
        for (Point point : points) {
            double key = point.coords[dimension];
            int count = hist.getOrDefault(key, 0);
            count++;
            hist.put(key, count);
        }
    }

    public void send(int dest, int tag, Intracomm comm) {

        int binNumber = hist.size();

        int[] header = {binNumber};
        comm.Send(header, 0, 1, MPI.INT, dest, tag);

        if (binNumber == 0) {
            return;
        }

        double[] keys = new double[binNumber];
        int[] counts = new int[binNumber];
        int i = 0;
        for (Map.Entry<Double, Integer> e : hist.entrySet()) {
            keys[i] = e.getKey();
            counts[i] = e.getValue();
            i++;
        }
        comm.Send(keys, 0, binNumber, MPI.DOUBLE, dest, tag);
        comm.Send(counts, 0, binNumber, MPI.INT, dest, tag);
    }

    public void receiveAndMerge(int src, int tag, Intracomm comm) {

        int[] header = new int[1];
        comm.Recv(header, 0, 1, MPI.INT, src, tag);
        int binNumber = header[0];

        if (binNumber == 0) {
            return;
        }

        double[] keys = new double[binNumber];
        int[] counts = new int[binNumber];
        comm.Recv(keys, 0, binNumber, MPI.DOUBLE, src, tag);
        comm.Recv(counts, 0, binNumber, MPI.INT, src, tag);

        for (int i = 0; i < binNumber; i++) {
            hist.put(keys[i], hist.getOrDefault(keys[i], 0) + counts[i]);
        }
    }

    public double determineSlidingWindowMedian(double epsilon) {
        if (hist.isEmpty())
            return 0.0;

        //turn histogram into arrays for efficient sorting
        double[] keys = hist.keySet().stream()
                .mapToDouble(Double::doubleValue)
                .toArray();
        int[] counts = hist.values().stream()
                .mapToInt(Integer::intValue)
                .toArray();


        int leftIdx = 0;
        int rightIdx = 0;
        int n = hist.size();

        //prefix-sum
        int[] prefix = new int[n + 1];
        for (int i = 0; i < n; i++)
            prefix[i + 1] = prefix[i] + counts[i];

        int total = prefix[n];

        double bestSplit = keys[0];
        int bestImbalance = Integer.MAX_VALUE;

        for (int splitIdx = 0; splitIdx < n; splitIdx++) {

            double split = keys[splitIdx];

            while (rightIdx < n && keys[rightIdx] <= split + epsilon)
                rightIdx++;

            while (leftIdx < rightIdx && keys[leftIdx] < split - epsilon)
                leftIdx++;

            int leftLoad = prefix[rightIdx];
            int rightLoad = total - prefix[leftIdx];

            int imbalance = Math.abs(leftLoad - rightLoad);

            if (imbalance < bestImbalance) {
                bestImbalance = imbalance;
                bestSplit = split;
            } else if (imbalance > bestImbalance) {
                return bestSplit;
            }
        }
        return bestSplit;
    }
}
