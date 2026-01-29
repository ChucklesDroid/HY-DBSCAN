import mpi.Group;
import mpi.Intracomm;
import mpi.MPI;
import mpi.Request;

import java.util.*;

import static mpi.MPI.COMM_WORLD;

public class Process {

    // tags for message passing
    private final int LOCAL_HISTOGRAM = 0;
    private final int GROUP_MEDIAN = 1;
    private final int POINT_EXCHANGE = 3;
    private final int BOUNDING_BOXES = 4;
    private final int GHOST_POINTS = 5;
    private BoundingBox boundingBox;
    private Set<BoundingBox> otherBoundingBoxes = new HashSet<>();

    private double epsilon;
    private int minPts;
    private int dimCount;

    private ArrayList<Point> points;
    private ArrayList<Point> ghostPoints = new ArrayList<>();
    int rank; // rank in the current group
    int numberOfProcessesInGroup; // number of processes in the current node of the kd-tree
    Group group;
    Intracomm communicator;
    double median;
    int currentDimension;

    public Process(ArrayList<Point> data, double epsilon) {
        points = data;
        rank = COMM_WORLD.Rank();
        numberOfProcessesInGroup = COMM_WORLD.Size();
        group = COMM_WORLD.Group();
        communicator = COMM_WORLD;
        currentDimension = 0;
        this.epsilon = epsilon;

        boundingBox = new BoundingBox(points.get(0).dimensions);

        if (data != null && !data.isEmpty()) {
            this.dimCount = data.get(0).dimensions;
            this.boundingBox = new BoundingBox(this.dimCount);
        } else {
            this.dimCount = 0;
        }
    }

    public void calculateAndSendHistogram() { //for the workers
        log("(Worker) Calculating histogram");
        Histogram hist = new Histogram(currentDimension, points);
        log("(Worker) Sending histogram");
        hist.send(0, LOCAL_HISTOGRAM, communicator);

    }

    public void findAndSendGroupMedian() { // for the master
        log("(Master) Calculating histogram");
        Histogram hist = new Histogram(currentDimension, points);
        log("(Master) Receiving histograms");
        for (int sender = 1; sender < numberOfProcessesInGroup; sender++) {
            hist.receiveAndMerge(sender, LOCAL_HISTOGRAM, communicator);
        }
        log("(Master) Calculating median");
        median = hist.determineSlidingWindowMedian(epsilon);

        // Bcast: much faster and solves for class-cast errors
        log("(Master) Broadcasting median");
        double[] m = new double[]{median};
        communicator.Bcast(m, 0, 1, MPI.DOUBLE, 0);
        log("(Master) Broadcasted median + " + median);
    }

    // public void findAndSendGroupMedian() { 
    //     Histogram hist = new Histogram(currentDimension, points);
    //     for (int sender = 1; sender < numberOfProcessesInGroup; sender++) {
    //         hist.receiveAndMerge(sender, LOCAL_HISTOGRAM, communicator);
    //     }
    //     median = hist.determineSlidingWindowMedian(epsilon);
    //
    //     Request[] messageRequests = new Request[(numberOfProcessesInGroup -1)*2];
    //     for (int i = 1; i <= numberOfProcessesInGroup; i++) {
    //         messageRequests[2*i] = communicator.Ibsend(
    //                 median, // send buffer
    //                 0, // offset
    //                 1, // number of items sent
    //                 MPI.DOUBLE, // data type
    //                 i, // receiving process rank
    //                 GROUP_MEDIAN // tag
    //         );
    //     }
    //     Request.Waitall(messageRequests);
    // }

    // public void receiveGroupMedian() { // for the workers
    //     double[] medianReceiver = new double[1];
    //     communicator.Recv(
    //             medianReceiver,
    //             0,
    //             1,
    //             MPI.DOUBLE,
    //             0,
    //             GROUP_MEDIAN
    //     );
    //     median = medianReceiver[0];
    // }

    public void receiveGroupMedian() { // for the workers
        log("(Worker) Receiving median");
        double[] medianReciever = new double[1];

        communicator.Bcast(medianReciever, 0, 1, MPI.DOUBLE, 0);
        median = medianReciever[0];
        log("(Worker) Received median + " + median);
    }

    public void exchangePoints() { // all processes
        log("Exchanging points");

        ArrayList<Point> upperPoints = new ArrayList<>();
        ArrayList<Point> lowerPoints = new ArrayList<>();

        log("Looking for points to exchange");
        for (Point point : points) {
            if (point.coords[currentDimension] > median) {
                upperPoints.add(point);
            } else {
                lowerPoints.add(point);
            }
        }

        log("Sending and receiving points");
        ArrayList<Point> sendList;
        if (rank < numberOfProcessesInGroup /2) {
            log("Sending higher points");
            sendList = upperPoints;
            points = lowerPoints;
            boundingBox.setMax(currentDimension, median);
        }else {
            log("Sending lower points");
            sendList = lowerPoints;
            points = upperPoints;
            boundingBox.setMin(currentDimension, median);
        }
        ArrayList<Point> receivedPoints;
        PointBuffer sendBuffer = new PointBuffer(sendList);

        int[] processAddressList = new int[numberOfProcessesInGroup];
        for (int i = 0; i < numberOfProcessesInGroup; i++) {
            processAddressList[i] = i;
        }

        int partnerProcess = numberOfProcessesInGroup - (rank + 1); // processes on the edges exchange with each other
        if (rank < numberOfProcessesInGroup /2) { // lower process sends first, then receives. Prevents interlocking
            log("Sending to then receiving from group partner " + partnerProcess);
            sendBuffer.send(communicator, partnerProcess, POINT_EXCHANGE);
            log("Sent " + sendList.size() + " points");
            receivedPoints = PointBuffer.receive(communicator, partnerProcess, POINT_EXCHANGE).toPointList();
            log("Received " + receivedPoints.size() + " points");

            //Preparation for next round
            group = group.Incl(Arrays.copyOfRange(processAddressList, 0, numberOfProcessesInGroup /2)); // define new communication group
        } else {
            log("Receiving from then sending to group partner " + partnerProcess);
            receivedPoints = PointBuffer.receive(communicator, partnerProcess, POINT_EXCHANGE).toPointList();
            log("Received " + receivedPoints.size() + " points");
            sendBuffer.send(communicator, partnerProcess, POINT_EXCHANGE);
            log("Sent " + sendList.size() + " points");

            //Preparation for next round
            group = group.Incl(Arrays.copyOfRange(processAddressList, numberOfProcessesInGroup /2, numberOfProcessesInGroup)); // define new communication group
        }
        points.addAll(receivedPoints); // gather all received points

        // Preparing next round

        numberOfProcessesInGroup /= 2; // can now only see half of the previous processes
        currentDimension = (currentDimension + 1) % dimCount;
        communicator = COMM_WORLD.Create(group);
        rank = group.Rank();

        log("Point exchange done");
    }

    // public void decomposeDomain() {
    //     for (int i = 0; i < Math.log(COMM_WORLD.Size()); i++) {
    //         if (rank == 0) {
    //             findAndSendGroupMedian();
    //         } else {
    //             calculateAndSendHistogram();
    //             receiveGroupMedian();
    //         }
    //         exchangePoints();
    //     }
    // }

    public void decomposeDomain() {
        log("Starting kd-tree");
        int depth = (int) (Math.log(COMM_WORLD.Size()) / Math.log(2));

        for (int i = 0; i < depth; i++) {
            log("starting kd-Tree round " + i + " with dimension " + currentDimension);
            if (rank == 0) {
                findAndSendGroupMedian();
            } else {
                calculateAndSendHistogram();
                receiveGroupMedian();
            }
            exchangePoints();
            log("finished kd-Tree round " + i + ". New group rank is " + rank);
        }
        log("kd-Tree done. Size: " + points.size() + ". Resetting communication parameters");
        group = COMM_WORLD.Group();
        communicator = COMM_WORLD;
        rank = group.Rank();
    }

    public void exchangeBoundingBoxes(){
        log("Exchanging bounding boxes. My own is " + boundingBox.toString());
        for (int address = 0; address < COMM_WORLD.Size(); address++) {
            // if (address == COMM_WORLD.Rank()) {
            if (address == rank) {
                log("Gathering bounding boxes");
                for (int sender = 0; sender < COMM_WORLD.Size(); sender++) {
                    //                 if (sender != COMM_WORLD.Rank()) {
                    if (sender != rank) {
                        BoundingBox other = BoundingBox.receive(sender, BOUNDING_BOXES);
                        otherBoundingBoxes.add(other);
                    }
                }
                log("Received all bounding boxes");
            } else {
                log("Sending bounding box to process " + address);
                this.boundingBox.send(COMM_WORLD, address, BOUNDING_BOXES);
            }
        }
    }

    public void exchangeGhostPoints() {
        log("Starting to exchange ghost points");
        // int k = 1;
        Set<BoundingBox> neighbours = boundingBox.neighbourSet(otherBoundingBoxes);
        Map<Integer, ArrayList<Point>> sendMap = new HashMap<>();

        while (true) {
            boolean newQueued = false;
            for (BoundingBox neighbour : neighbours) {
                int neighbourAddress = neighbour.globalCommGroupAddress;
                sendMap.putIfAbsent(neighbourAddress, new ArrayList<>());

                for (Point point : points) {
                    if (neighbour.distanceToPoint(point) <= epsilon) {
                        sendMap.get(neighbourAddress).add(point);
                        newQueued = true;
                    }
                }
            }
            if (!newQueued) {
                break;
            }
            Set<BoundingBox> newNeighbours = new HashSet<>();
            for (BoundingBox neighbour: neighbours) {
                newNeighbours.addAll(neighbour.neighbourSet(otherBoundingBoxes));
            }
            neighbours = newNeighbours;
        }

        for (int address = 0; address < COMM_WORLD.Size(); address++) {
            if (address == COMM_WORLD.Rank()) {
                for (int sender = 0; sender < COMM_WORLD.Size(); sender++) {
                    if (sender != COMM_WORLD.Rank()) {
                        ghostPoints.addAll(PointBuffer.receive(COMM_WORLD, sender, GHOST_POINTS).toPointList());
                    }
                }
            } else {
                new PointBuffer(sendMap.getOrDefault(address, new ArrayList<>())).send(COMM_WORLD, address, GHOST_POINTS);
            }
        }
        log("Finished exchanging ghost points. Received: " + ghostPoints.size());
        log("Epsilon is " + epsilon);
    }

    public void log(String message) {
        System.out.println("Process " + COMM_WORLD.Rank() + ": " + message);
    }

}
