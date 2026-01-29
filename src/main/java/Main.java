import mpi.MPI;
import mpi.MPIException; 
import mpi.Request;
import mpi.Comm;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.nio.ByteBuffer;
import java.util.ArrayList;
import java.util.stream.Collectors;
import java.util.stream.Stream;

public class Main {
    static private ArrayList<Point> data;
    static private int dimensions;

    /**
     * Reads the specified csv file and sets the data and dimensions fields
     * @param filename path to the used file
     * @throws IOException just an ioexception
     */
    static private void readData(String filename) throws IOException {
        try (Stream<String> stream = Files.lines(Paths.get(filename))) {
            data = (ArrayList<Point>) stream.map(s -> s.split(","))
                    .map(arr -> {
                                double[] Doubles = new double[arr.length];
                                int i = 0;
                                for (String s : arr) {
                                    Doubles[i] = Double.parseDouble(s.trim());
                                    i++;
                                }
                                return new Point(Doubles);
                            }
                    ).collect(Collectors.toList());
        }
        dimensions = data.get(0).dimensions;
    }

    public static void main(String[] args) throws Exception {
        //read time here
        MPI.Init(args);

        // readData("src/main/resources/datasets/densired_2_truncated.csv");
        readData("src/main/resources/datasets/densired_2_shrink.csv");

        int numOfProcesses = MPI.COMM_WORLD.Size();
        int rank = MPI.COMM_WORLD.Rank();
        int blockSize;
        int dataSize = data.size();
        blockSize = dataSize / numOfProcesses;
        if (dataSize % numOfProcesses != 0) {
            blockSize++; // one more if data size isn't divisible by the number of processes
        }

        Process process = new Process(
                new ArrayList<>(data.subList(rank * blockSize, Math.min((rank + 1) * blockSize, dataSize)))
        );

        process.decomposeDomain();
        process.exchangeBoundingBoxes();
        process.exchangeGhostPoints();

        System.out.println("Still alive");

        MPI.Finalize();
        //read time here
    }
}
