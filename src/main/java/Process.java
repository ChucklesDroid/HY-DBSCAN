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

    private ArrayList<Point> points;
    private ArrayList<Point> ghostPoints;
    int rank; // rank in the current group
    int numberOfProcessesInGroup; // number of processes in the current node of the kd-tree
    Group group;
    Intracomm communicator;
    double median;
    int currentDimension;

    public Process(ArrayList<Point> data) {
        points = data;
        rank = COMM_WORLD.Rank();
        numberOfProcessesInGroup = COMM_WORLD.Size();
        group = COMM_WORLD.Group();
        communicator = COMM_WORLD;
        currentDimension = 0;

        boundingBox = new BoundingBox(points.get(0).dimensions);
    }


    public void calculateAndSendHistogram() { //for the workers
        Histogram hist = new Histogram(currentDimension, points);

        hist.send(0, LOCAL_HISTOGRAM, communicator);

    }



    public void findAndSendGroupMedian() { // for the master

        Histogram hist = new Histogram(currentDimension, points);
        for (int sender = 1; sender < numberOfProcessesInGroup; sender++) {
            hist.receiveAndMerge(sender, LOCAL_HISTOGRAM, communicator);
        }
        median = hist.determineSlidingWindowMedian(epsilon);

        Request[] messageRequests = new Request[(numberOfProcessesInGroup -1)*2];
        for (int i = 1; i <= numberOfProcessesInGroup; i++) {
            messageRequests[2*i] = communicator.Ibsend(
                    median, // send buffer
                    0, // offset
                    1, // number of items sent
                    MPI.DOUBLE, // data type
                    i, // receiving process rank
                    GROUP_MEDIAN // tag
            );
        }
        Request.Waitall(messageRequests);
    }

    public void receiveGroupMedian() { // for the workers
        double[] medianReceiver = new double[1];
        communicator.Recv(
                medianReceiver,
                0,
                1,
                MPI.DOUBLE,
                0,
                GROUP_MEDIAN
        );
        median = medianReceiver[0];

    }

    public void exchangePoints() { // all processes

        int cutoff = 0;

        for (Point point : points) {
            if (point.coords[currentDimension] > median) {
                break;
            }
            cutoff++;
        }

        ArrayList<Point> sendList;
        if (rank < numberOfProcessesInGroup /2) {
            try {
                sendList = new ArrayList<>(points.subList(cutoff, points.size()));
            } catch (IndexOutOfBoundsException e) {
                sendList = new ArrayList<>();
            }
            points = new ArrayList<>(points.subList(0, cutoff));
            boundingBox.setMax(currentDimension, median);
        }else {
            sendList = new ArrayList<>(points.subList(0, cutoff));
            try {
                points = new ArrayList<>(points.subList(cutoff, points.size()));
            } catch (IndexOutOfBoundsException e) {
                points = new ArrayList<>();
            }
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
            sendBuffer.send(communicator, partnerProcess, POINT_EXCHANGE);
            receivedPoints = PointBuffer.receive(communicator, partnerProcess, POINT_EXCHANGE).toPointList();
            group = group.Incl(Arrays.copyOfRange(processAddressList, 0, numberOfProcessesInGroup /2)); // define new communication group
        } else {
            receivedPoints = PointBuffer.receive(communicator, partnerProcess, POINT_EXCHANGE).toPointList();
            sendBuffer.send(communicator, partnerProcess, POINT_EXCHANGE);
            group = group.Incl(Arrays.copyOfRange(processAddressList, numberOfProcessesInGroup /2, numberOfProcessesInGroup)); // define new communication group
        }
        points.addAll(receivedPoints); // gather all received points
        numberOfProcessesInGroup /= 2; // can now only see half of the previous processes

        currentDimension++;
        currentDimension %= points.get(0).dimensions;

    }

    public void decomposeDomain() {
        for (int i = 0; i < Math.log(COMM_WORLD.Size()); i++) {
            if (rank == 0) {
                findAndSendGroupMedian();
            } else {
                calculateAndSendHistogram();
                receiveGroupMedian();
            }
            exchangePoints();
        }
    }

    public void exchangeBoundingBoxes(){
        for (int address = 0; address < COMM_WORLD.Size(); address++) {
            if (address == COMM_WORLD.Rank()) {
                for (int sender = 0; sender < COMM_WORLD.Size(); sender++) {
                    if (sender != COMM_WORLD.Rank()) {
                        BoundingBox other = BoundingBox.receive(sender, BOUNDING_BOXES);
                            otherBoundingBoxes.add(other);
                    }
                }
            } else {
                boundingBox.send(address, BOUNDING_BOXES);
            }
        }
    }

    public void exchangeGhostPoints() {
        int k = 1;
        Set<BoundingBox> neighbours = boundingBox.neighbourSet(otherBoundingBoxes);
        Map<Integer, ArrayList<Point>> sendMap = new HashMap<>();
        while (true) {
            boolean newQueued = false;
            for (BoundingBox neighbour : neighbours) {
                int neighbourAddress = neighbour.globalCommGroupAddress;
                sendMap.put(neighbourAddress, new ArrayList<>());
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
    }

}
