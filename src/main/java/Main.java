import mpi.MPI;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.stream.Collectors;
import java.util.stream.Stream;

public class Main {
    static double epsilon = 0.1;
    static int minpts = 3;
    // static final String DATASET = "densired_2_truncated.csv";
    static final String DATASET = "test.csv";

    static private ArrayList<Point> data;
    static private int dimensions;

    /**
     * Reads the specified csv file and sets the data and dimensions fields.
     */
    static private void readData(String filename) throws IOException {
        try (Stream<String> stream = Files.lines(Paths.get(filename))) {
            data = (ArrayList<Point>) stream.map(s -> s.split(","))
                    .map(arr -> {
                        double[] doubles = new double[arr.length];
                        for (int i = 0; i < arr.length; i++) {
                            doubles[i] = Double.parseDouble(arr[i].trim());
                        }
                        return new Point(doubles);
                    })
                    .collect(Collectors.toList());
        }
        dimensions = data.get(0).dimensions;
    }

    public static void main(String[] args) throws Exception {
        MPI.Init(args);

        int numProcesses = MPI.COMM_WORLD.Size();
        int rank         = MPI.COMM_WORLD.Rank();
        boolean isMaster = (rank == 0);

        // ----------------------------------------------------------------
        // Read data (all processes read the full file, then take a slice)
        // ----------------------------------------------------------------
        long tTotal = now();
        long t0 = now();

        readData("src/main/resources/datasets/" + DATASET);

        long readMs = elapsed(t0);

        int dataSize  = data.size();
        int blockSize = dataSize / numProcesses + (dataSize % numProcesses != 0 ? 1 : 0);

        ArrayList<Point> localData = new ArrayList<>(
                data.subList(rank * blockSize, Math.min((rank + 1) * blockSize, dataSize)));

        Process process = isMaster
                ? new Master(localData, epsilon)
                : new Worker(localData, epsilon);

        process.log("Finished reading data (" + readMs + " ms)");

        // ----------------------------------------------------------------
        // Domain decomposition (kd-tree)
        // ----------------------------------------------------------------
        t0 = now();
        process.decomposeDomain();
        long decomposeMs = elapsed(t0);

        // ----------------------------------------------------------------
        // Exchange bounding boxes
        // ----------------------------------------------------------------
        t0 = now();
        process.exchangeBoundingBoxes();
        long exchangeBBMs = elapsed(t0);

        // ----------------------------------------------------------------
        // Exchange ghost points
        // ----------------------------------------------------------------
        t0 = now();
        process.exchangeGhostPoints();
        long exchangeGhostMs = elapsed(t0);

        // ----------------------------------------------------------------
        // Local DBSCAN
        // ----------------------------------------------------------------
        t0 = now();
        process.localDBScan(minpts);
        long localDbscanMs = elapsed(t0);

        // ----------------------------------------------------------------
        // Distributed cluster merge
        // ----------------------------------------------------------------
        t0 = now();
        process.mergeClusters();
        long mergeMs = elapsed(t0);

        // ----------------------------------------------------------------
        // Assign global cluster IDs
        // ----------------------------------------------------------------
        t0 = now();
        process.assignClusterIds();
        long assignIdsMs = elapsed(t0);

        long totalMs = elapsed(tTotal);

        process.log("Pipeline done. Total: " + totalMs + " ms");

        // ----------------------------------------------------------------
        // Gather points to rank 0 and write output files
        // ----------------------------------------------------------------
        gatherAndWrite(process, rank, numProcesses,
                readMs, decomposeMs, exchangeBBMs, exchangeGhostMs,
                localDbscanMs, mergeMs, assignIdsMs, totalMs);

        MPI.Finalize();
    }

    // --------------------------------------------------------------------
    // Gather all points to rank 0, then write both output files.
    // Each rank sends: header [n, dims], coords double[], clusterIds int[].
    // Rank 0 receives them all, then writes both CSV files.
    // --------------------------------------------------------------------
    private static void gatherAndWrite(
            Process process, int rank, int numProcesses,
            long readMs, long decomposeMs, long exchangeBBMs, long exchangeGhostMs,
            long localDbscanMs, long mergeMs, long assignIdsMs, long totalMs) {

        final int GATHER_TAG = 10;

        if (rank == 0) {
            // Start with rank 0's own local (non-ghost) points
            ArrayList<Point> allPoints = new ArrayList<>(process.points);

            // Receive from all other ranks
            for (int src = 1; src < numProcesses; src++) {
                int[] header = new int[2];
                MPI.COMM_WORLD.Recv(header, 0, 2, MPI.INT, src, GATHER_TAG);
                int n    = header[0];
                int dims = header[1];

                double[] coords     = new double[n * dims];
                int[]    clusterIds = new int[n];
                MPI.COMM_WORLD.Recv(coords,     0, coords.length,     MPI.DOUBLE, src, GATHER_TAG);
                MPI.COMM_WORLD.Recv(clusterIds, 0, clusterIds.length, MPI.INT,    src, GATHER_TAG);

                for (int i = 0; i < n; i++) {
                    double[] c = new double[dims];
                    System.arraycopy(coords, i * dims, c, 0, dims);
                    Point p = new Point(c);
                    p.globalClusterId = clusterIds[i];
                    allPoints.add(p);
                }
            }

            // Write points CSV (always overwrite)
            ResultWriter.writePoints(allPoints);
            System.out.println("Process 0: wrote " + allPoints.size() + " points to points.csv");

            // Write bounding.csv
            int numOfBoxes = ResultWriter.writeBoundingBoxes(process.boundingBox, process.otherBoundingBoxesCopy);
            System.out.println("Process 0: wrote " + numOfBoxes + " bounding.csv");

            // Write timing CSV (append)
            ResultWriter.writeTiming(
                    DATASET, numProcesses, epsilon, minpts,
                    readMs, decomposeMs, exchangeBBMs, exchangeGhostMs,
                    localDbscanMs, mergeMs, assignIdsMs, totalMs);
            System.out.println("Process 0: appended timing row to timing.csv");

        } else {
            ArrayList<Point> local = process.points;
            int n    = local.size();
            int dims = process.dimCount;

            int[]    header     = {n, dims};
            double[] coords     = new double[n * dims];
            int[]    clusterIds = new int[n];

            for (int i = 0; i < n; i++) {
                System.arraycopy(local.get(i).coords, 0, coords, i * dims, dims);
                clusterIds[i] = local.get(i).globalClusterId;
            }

            MPI.COMM_WORLD.Send(header,     0, 2,             MPI.INT,    0, GATHER_TAG);
            MPI.COMM_WORLD.Send(coords,     0, coords.length, MPI.DOUBLE, 0, GATHER_TAG);
            MPI.COMM_WORLD.Send(clusterIds, 0, n,             MPI.INT,    0, GATHER_TAG);
        }
    }

    private static long now() {
        return System.currentTimeMillis();
    }

    private static long elapsed(long startMs) {
        return System.currentTimeMillis() - startMs;
    }
}
