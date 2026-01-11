import mpi.MPI;

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

void main(String[] args) throws Exception {
    readData("src/main/resources/datasets/densired_2_truncated.csv");
    MPI.Init(args);

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

    process.calculateIndividualMedian(1);
    IO.println("Hello from rank " + process.rank + " of " + MPI.COMM_WORLD.Size() + " with data: " + process.points.get(0).coords[0] + ", " + process.points.get(0).coords[1] + " and " + process.points.size() + " points and x-median " + process.median);

    MPI.Finalize();
}