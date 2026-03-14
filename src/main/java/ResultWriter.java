import java.io.*;
import java.nio.file.*;
import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;
import java.util.ArrayList;
import java.util.Set;

/**
 * Writes two output files:
 *
 *  1. timing.csv  — one row per run, appended. Records run metadata and the
 *                   wall-clock time (ms) spent in each pipeline stage.
 *
 *  2. points.csv  — one row per data point, overwritten each run. Contains the
 *                   coordinates and the assigned global cluster ID (-1 = noise),
 *                   suitable for plotting.
 */
public class ResultWriter {

    // -----------------------------------------------------------------------
    // Timing CSV
    // -----------------------------------------------------------------------

    private static final String TIMING_FILE = "timing.csv";

    private static final String TIMING_HEADER =
            "timestamp,dataset,numProcesses,epsilon,minPts," +
                    "readMs,decomposeMs,exchangeBoundingBoxesMs,exchangeGhostPointsMs," +
                    "localDbscanMs,mergeMs,assignIdsMs,totalMs";

    /**
     * Appends one timing row to timing.csv, creating the file with a header
     * row if it does not yet exist.
     *
     * @param dataset               filename/label of the input dataset
     * @param numProcesses          total MPI process count
     * @param epsilon               DBSCAN epsilon parameter
     * @param minPts                DBSCAN minPts parameter
     * @param readMs                time to read input data (ms)
     * @param decomposeMs           time for kd-tree domain decomposition (ms)
     * @param exchangeBBMs          time to exchange bounding boxes (ms)
     * @param exchangeGhostMs       time to exchange ghost points (ms)
     * @param localDbscanMs         time for local DBSCAN (ms)
     * @param mergeMs               time for distributed cluster merge (ms)
     * @param assignIdsMs           time to assign global cluster IDs (ms)
     * @param totalMs               total wall-clock time (ms)
     */
    public static void writeTiming(
            String dataset, int numProcesses, double epsilon, int minPts,
            long readMs, long decomposeMs, long exchangeBBMs, long exchangeGhostMs,
            long localDbscanMs, long mergeMs, long assignIdsMs, long totalMs) {

        boolean fileExists = Files.exists(Paths.get(TIMING_FILE));

        try (BufferedWriter bw = new BufferedWriter(
                new FileWriter(TIMING_FILE, true))) { // true = append

            if (!fileExists) {
                bw.write(TIMING_HEADER);
                bw.newLine();
            }

            String timestamp = LocalDateTime.now()
                    .format(DateTimeFormatter.ofPattern("yyyy-MM-dd HH:mm:ss"));

            bw.write(String.join(",",
                    timestamp,
                    dataset,
                    String.valueOf(numProcesses),
                    String.valueOf(epsilon),
                    String.valueOf(minPts),
                    String.valueOf(readMs),
                    String.valueOf(decomposeMs),
                    String.valueOf(exchangeBBMs),
                    String.valueOf(exchangeGhostMs),
                    String.valueOf(localDbscanMs),
                    String.valueOf(mergeMs),
                    String.valueOf(assignIdsMs),
                    String.valueOf(totalMs)
            ));
            bw.newLine();

        } catch (IOException e) {
            System.err.println("ResultWriter: failed to write timing file: " + e.getMessage());
        }
    }

    // -----------------------------------------------------------------------
    // Points CSV
    // -----------------------------------------------------------------------

    private static final String POINTS_FILE = "points.csv";

    /**
     * Writes all points with their cluster IDs to points.csv, overwriting any
     * previous file. Each row contains the coordinates followed by the global
     * cluster ID (-1 for noise).
     *
     * Column header: x0,x1,...,x(d-1),clusterId
     *
     * @param allPoints flat list of all data points (gathered from all processes)
     */
    public static void writePoints(ArrayList<Point> allPoints) {
        if (allPoints == null || allPoints.isEmpty()) return;

        int dims = allPoints.get(0).dimensions;

        try (BufferedWriter bw = new BufferedWriter(
                new FileWriter(POINTS_FILE, false))) { // false = overwrite

            // Header
            StringBuilder header = new StringBuilder();
            for (int d = 0; d < dims; d++) {
                header.append("x").append(d).append(",");
            }
            header.append("clusterId");
            bw.write(header.toString());
            bw.newLine();

            // One row per point
            StringBuilder row = new StringBuilder();
            for (Point p : allPoints) {
                row.setLength(0);
                for (int d = 0; d < dims; d++) {
                    row.append(p.coords[d]).append(",");
                }
                row.append(p.globalClusterId);
                bw.write(row.toString());
                bw.newLine();
            }

        } catch (IOException e) {
            System.err.println("ResultWriter: failed to write points file: " + e.getMessage());
        }
    }

    private static final String BOUNDING_BOX_FILE = "bounding.csv";

    public static int writeBoundingBoxes(BoundingBox masterBoundingBox, Set<BoundingBox> boundingBoxes) {
        boundingBoxes.add(masterBoundingBox);

        int dims = masterBoundingBox.numOfDimensions;

        try (BufferedWriter bw = new BufferedWriter(
                new FileWriter(BOUNDING_BOX_FILE, false))) { // false = overwrite

            // Header
            StringBuilder header = new StringBuilder();
            for (int d = 0; d < dims; d++) {
                header.append("x").append(d).append("min").append(",");
                header.append("x").append(d).append("max").append(",");
            }
            bw.write(header.toString());
            bw.newLine();

            // One row per bounding box
            StringBuilder row = new StringBuilder();
            for (BoundingBox b : boundingBoxes) {
                row.setLength(0);
                for (int d = 0; d < dims; d++) {
                    double minX = b.minMaxPerDimension[d][0];
                    double maxX = b.minMaxPerDimension[d][1];
                    if (minX == Double.MIN_VALUE) {
                        row.append("min_x").append(d).append(",");
                    } else {
                        row.append(minX).append(",");
                    }
                    if (maxX == Double.MAX_VALUE) {
                        row.append("max_x").append(d).append(",");
                    } else {
                        row.append(maxX).append(",");
                    }
                }
                bw.write(row.toString());
                bw.newLine();
            }
            return boundingBoxes.size();
        } catch (IOException e) {
            System.err.println("ResultWriter: failed to write bounding file: " + e.getMessage());
            return 0;
        }
    }
}