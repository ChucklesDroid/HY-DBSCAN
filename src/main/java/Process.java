import mpi.Group;
import mpi.Intracomm;
import mpi.MPI;
import java.util.*;
import static mpi.MPI.COMM_WORLD;

public abstract class Process {
    // tags for message passing
    protected final int LOCAL_HISTOGRAM = 0;
    protected final int GROUP_MEDIAN = 1;
    protected final int POINT_EXCHANGE = 3;
    protected final int BOUNDING_BOXES = 4;
    protected final int GHOST_POINTS = 5;

    protected BoundingBox boundingBox;
    protected Set<BoundingBox> otherBoundingBoxes = new HashSet<>();
    protected double epsilon;
    protected int dimCount;
    protected ArrayList<Point> points;
    protected ArrayList<Point> ghostPoints = new ArrayList<>();
    
    int numberOfProcessesInGroup; // number of processes in the current node of the kd-tree
    protected int rank; // rank in the current group
    protected Group group;
    protected Intracomm communicator;
    protected double median;
    protected int currentDimension;

    public Process(ArrayList<Point> data, double epsilon) {
        this.points = data;
        this.rank = COMM_WORLD.Rank();
        this.numberOfProcessesInGroup = COMM_WORLD.Size();
        this.group = COMM_WORLD.Group();
        this.communicator = COMM_WORLD;
        this.currentDimension = 0;
        this.epsilon = epsilon;

        if (data != null && !data.isEmpty()) {
            this.dimCount = data.get(0).dimensions;
            this.boundingBox = new BoundingBox(this.dimCount);
        } else {
            this.dimCount = 0;
        }
    }

    public abstract void decomposeDomain(); 

    public void exchangePoints() {
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
            log("Sending to and then receiving from group partner " + partnerProcess);
            sendBuffer.send(communicator, partnerProcess, POINT_EXCHANGE);
            log("Sent " + sendList.size() + " points");
            receivedPoints = PointBuffer.receive(communicator, partnerProcess, POINT_EXCHANGE).toPointList();
            log("Received " + receivedPoints.size() + " points");

            //Preparation for next round
            group = group.Incl(Arrays.copyOfRange(processAddressList, 0, numberOfProcessesInGroup /2)); // define new communication group
        } else {
            log("Receiving from and then sending to group partner " + partnerProcess);
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

    public void exchangeBoundingBoxes(){
        log("Exchanging bounding boxes. My own is " + boundingBox.toString());
        for (int address = 0; address < COMM_WORLD.Size(); address++) {
            if (address == rank) {
                log("Gathering bounding boxes");
                for (int sender = 0; sender < COMM_WORLD.Size(); sender++) {
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

    public void resetCommunication() {
        log("kd-Tree done. Size: " + points.size() + ".Resetting communication parameters");
        group = COMM_WORLD.Group();
        communicator = COMM_WORLD;
        rank = group.Rank();
    }

    public void log(String message) {
        System.out.println("Process " + COMM_WORLD.Rank() + ": " + message);
    }
}
