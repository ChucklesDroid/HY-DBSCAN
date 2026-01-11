import mpi.*;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.Stream;

public class Main {

    static private ArrayList<double[]> data;
    static private int dimensions;

    /**
     * Reads the specified csv file and sets the data and dimensions fields
     * @param filename
     * @throws IOException
     */
    static private void readData(String filename) throws IOException {
        try (Stream<String> stream = Files.lines(Paths.get(filename))) {
            data = (ArrayList<double[]>) stream.map(s -> s.split(","))
                    .map(arr -> {
                        double[] Doubles = new double[arr.length];
                        int i = 0;
                        for (String s : arr) {
                            Doubles[i] = Double.parseDouble(s.trim());
                            i++;
                        }
                        return Doubles;
                    }
            ).collect(Collectors.toList());
        }
        dimensions = data.get(0).length;
    }

    static private void makeDataBlocks() {

    }

    public static void main(String[] args) throws Exception {
        readData("src/main/resources/datasets/densired_2_truncated.csv");
        MPI.Init(args);

        int numOfProcesses = MPI.COMM_WORLD.Size();
        int rank = MPI.COMM_WORLD.Rank();
        int blockSize;
        int dataSize = data.size();
        blockSize = dataSize / numOfProcesses;
        if (dataSize%numOfProcesses != 0) {
            blockSize++; // one more if data size isn't divisible by the number of processes
        }

        Process process = new Process(
                new ArrayList<> (data.subList(rank * blockSize, Math.min((rank+1) * blockSize, dataSize)))
        );

        System.out.println("Hello from rank " + process.rank + " of " + MPI.COMM_WORLD.Size() + " with data: " +  process.points.get(0)[0] + ", " + process.points.get(0)[1] + " and " + process.points.size() + " points.");

        MPI.Finalize();
    }
}