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
    //to be replaced later
    static double epsilon = 0.1;
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
        int dataSize = data.size();
        int blockSize = dataSize / numOfProcesses;
        if (dataSize % numOfProcesses != 0) {
            blockSize++; // one more if data size isn't divisible by the number of processes
        }

        ArrayList<Point> localData = new ArrayList<>(data.subList(rank * blockSize, Math.min((rank + 1) * blockSize, dataSize)));

        Process process;
        if (rank == 0) {
            process = new Master(localData, epsilon);
        } else {
            process = new Worker(localData, epsilon);
        }

        process.log("Finished reading data");

        process.decomposeDomain();
        process.exchangeBoundingBoxes();
        process.exchangeGhostPoints();

        process.log("done");

        MPI.Finalize();
        //read time here
    }
}
