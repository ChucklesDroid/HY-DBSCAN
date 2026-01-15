import mpi.Group;
import mpi.Intracomm;
import mpi.MPI;
import mpi.Request;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;

public class Process {

    // tags for message passing
    private final int INDIVIDUAL_MEDIAN = 0;
    private final int GROUP_MEDIAN = 1;
    private final int NEW_GROUP = 2;
    private final int POINT_EXCHANGE = 3;

    ArrayList<Point> points;
    int rank; // rank in the current group
    int processesInGroup; // number of processes in the current node of the kd-tree
    Group group;
    Intracomm communicator;
    double median;
    int[] sortedProcesses;
    int currentDimension;

    public Process(ArrayList<Point> data) {
        points = data;
        rank = MPI.COMM_WORLD.Rank();
        processesInGroup = MPI.COMM_WORLD.Size();
        group = MPI.COMM_WORLD.Group();
        communicator = MPI.COMM_WORLD;
        currentDimension = 0;
    }

    public void calculateIndividualMedian() {
        points.sort(Comparator.comparingDouble(point -> point.coords[currentDimension]));
        int size = points.size();
        if (points.size() % 2 == 1) {
            median = points.get((size - 1) / 2).coords[currentDimension];
            return;
        }
        median = (points.get((size - 2) / 2).coords[currentDimension] + points.get((size / 2)).coords[currentDimension]) / 2;
    }

    public void sendIndividualMedian() { // for the workers
        double[] message = new double[]{median};
        communicator.Send(
                message, // message buffer to send
                0, // offset of message start
                1, // number of items to send
                MPI.DOUBLE, // data type to send
                0, // receiving rank (group master)
                INDIVIDUAL_MEDIAN // tag
        );
    }

    public void finaAndSendGroupMedian() { // for the master
        RankValuePair[] allMedians = new RankValuePair[group.Size()];
        allMedians[0] = new RankValuePair(this.rank, this.median);

        Request[] requests = new Request[group.Size()-1];
        double[][] receiveBuffers = new double[group.Size() - 1][1];

        for (int i = 0; i < group.Size()-1; i++) {
            requests[i] = communicator.Irecv( // non-blocking so messages don't have to arrive in order of rank
                    receiveBuffers[i], // receive buffer
                    0, // offset
                    1, // number of received message items
                    MPI.DOUBLE, // data type
                    i+1, // from process with specified rank
                    INDIVIDUAL_MEDIAN // tag
            );
        }
        Request.Waitall(requests); // wait for messages to arrive
        for (int i = 0; i < receiveBuffers.length; i++) {
            allMedians[i+1] = new RankValuePair(i+1, receiveBuffers[i][0]);
        }

        Arrays.sort(allMedians); // sorting to easily find median and also easily find send/recieve partners that have little data to send

        median = (allMedians[(processesInGroup/2) - 1].value + allMedians[processesInGroup/2].value) / 2;

        sortedProcesses = Arrays.stream(allMedians).mapToInt(rvp -> rvp.rank).toArray();

        double[] groupMedian = new double[]{median};

        Request[] messageRequests = new Request[(processesInGroup-1)*2];
        for (int i = 1; i <= processesInGroup; i++) {
            messageRequests[2*i] = communicator.Ibsend(
                    groupMedian, // send buffer
                    0, // offset
                    1, // number of items sent
                    MPI.DOUBLE, // data type
                    i, // receiving process rank
                    GROUP_MEDIAN // tag
            );

            messageRequests[(2*i)+1] = communicator.Ibsend(
                    sortedProcesses, //send buffer
                    0, // offset
                    processesInGroup, //number of sent items
                    MPI.INT, // data type
                    i, //destination
                    NEW_GROUP // tag
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
        sortedProcesses= new int[processesInGroup];
        communicator.Recv(
                sortedProcesses,
                0,
                processesInGroup,
                MPI.INT,
                0,
                NEW_GROUP
        );
        this.median = medianReceiver[0];
    }

    public void exchangePoints() {
        int myPosition = -1;
        for (int i = 0; i <= processesInGroup; i++) {
            if (sortedProcesses[i] == rank) {
                myPosition = i;
                break;
            }
        }
        if (myPosition < 0) {
            System.out.println("Something went wrong and the process didn't find itself in the sorted process list!");
        }

        int cutoff = 0;

        for (Point point : points) {
            if (point.coords[currentDimension] > median) {
                break;
            }
            cutoff++;
        }

        ArrayList<Point> sendList;
        if (myPosition < processesInGroup/2) {
            try {
                sendList = new ArrayList<>(points.subList(cutoff, points.size()));
            } catch (IndexOutOfBoundsException e) {
                sendList = new ArrayList<>();
            }
            points = new ArrayList<>(points.subList(0, cutoff));
        }else {
            sendList = new ArrayList<>(points.subList(0, cutoff));
            try {
                points = new ArrayList<>(points.subList(cutoff, points.size()));
            } catch (IndexOutOfBoundsException e) {
                points = new ArrayList<>();
            }
        }
        ArrayList<Point> receivedPoints;
        PointBuffer sendBuffer = new PointBuffer(sendList);
        int partnerProcess = sortedProcesses[processesInGroup - (myPosition + 1)]; // processes on the edges exchange with each other
        if (myPosition < processesInGroup/2) { // lower process sends first, then receives. Prevents interlocking
            sendBuffer.send(communicator, partnerProcess, POINT_EXCHANGE);
            receivedPoints = PointBuffer.receive(communicator, partnerProcess, POINT_EXCHANGE).toPointList();
            group = group.Incl(Arrays.copyOfRange(sortedProcesses, 0, processesInGroup/2)); // define new communication group
        } else {
            receivedPoints = PointBuffer.receive(communicator, partnerProcess, POINT_EXCHANGE).toPointList();
            sendBuffer.send(communicator, partnerProcess, POINT_EXCHANGE);
            group = group.Incl(Arrays.copyOfRange(sortedProcesses, processesInGroup/2, processesInGroup)); // define new communication group
        }
        points.addAll(receivedPoints); // gather all received points
        processesInGroup /= 2; // can now only see half of the previous processes

        currentDimension++;
        currentDimension %= points.get(0).dimensions;

    }

// TODO: Also get an iteration counter Or a check on how many processes there are... and implement a workflow of functions.
//  New Masters should be notified to continue. The first master should be started from the main file.
//  Processes also need to exchange points

}
