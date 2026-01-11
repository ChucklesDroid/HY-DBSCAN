import mpi.Group;
import mpi.Intracomm;
import mpi.MPI;
import mpi.Request;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;

public class Process {

    // tags for message passing
    private final int INDIVIDUALMEDIAN = 0;
    private final int GROUPMEDIAN = 1;
    private final int POINTEXCHANGE = 2;

    ArrayList<Point> points;
    int rank; // rank in the current group
    int processesInGroup; // number of processes in the current node of the kd-tree
    Group group;
    Intracomm communicator;
    double median;

    public Process(ArrayList<Point> data) {
        points = data;
        rank = MPI.COMM_WORLD.Rank();
        processesInGroup = MPI.COMM_WORLD.Size();
        group = MPI.COMM_WORLD.Group();
    }

    public void calculateIndividualMedian(int dimension) {
        int column = dimension - 1;
        points.sort(Comparator.comparingDouble(point -> point.coords[column]));
        int size = points.size();
        if (points.size() % 2 == 1) {
            median = points.get((size - 1) / 2).coords[column];
            return;
        }
        median = (points.get((size - 2) / 2).coords[column] + points.get((size / 2)).coords[column]) / 2;
    }

    public void sendIndividualMedian() {
        double[] message = new double[]{median};
        communicator.Send(
                message, // message buffer to send
                0, // offset of message start
                1, // number of items to send
                MPI.DOUBLE, // data type to send
                0, // receiving rank (group master)
                INDIVIDUALMEDIAN // tag
        );
    }

    public void findGlobalMedian() {
        RankValuePair[] allMedians = new RankValuePair[group.Size()];
        allMedians[0] = new RankValuePair(this.rank, this.median);

        Request[] requests = new Request[group.Size()-1];
        double[][] recieveBuffers = new double[group.Size() - 1][1];

        for (int i = 0; i < group.Size()-1; i++) {
            requests[i] = communicator.Irecv( // non-blocking so messages don't have to arrive in order of rank
                    recieveBuffers[i], // recieve buffer
                    0, // offset
                    1, // number of recieved message items
                    MPI.DOUBLE, // data type
                    i, // from process with specified rank
                    INDIVIDUALMEDIAN // tag
            );
        }
        Request.Waitall(requests); // wait for messages to arrive
        for (int i = 0; i < recieveBuffers.length; i++) {
            allMedians[i+1] = new RankValuePair(i+1, recieveBuffers[i][0]);
        }

        Arrays.sort(allMedians); // sorting to easily find median and also easily find send/recieve partners that have little data to send

    }

    // TODO: Get Median, send Median and partner to all processes, define new groups and send message to update.
    //  Also get an iteration counter and implement a workflow of functions.
    //  New Masters should be notified to continue. The first master should be started from the main file.
    //  Processes also need to exchange points

}
