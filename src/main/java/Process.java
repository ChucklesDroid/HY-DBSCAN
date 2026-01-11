import mpi.MPI;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.Random;
import java.util.stream.Collectors;

public class Process {
    ArrayList<double[]> points;
    int rank;
    int rootRank; // rank of the root process
    int partner; // rank of the partner process to send to
    boolean highPartner; // whether to gather high point data or low point data
    int processesInNode; // number of processes in the current node of the kd-tree

    public Process(ArrayList<double[]> data) {
        points = data;
        rank = MPI.COMM_WORLD.Rank();
        rootRank = 0; // initially the first process, changes during partitioning
        processesInNode = MPI.COMM_WORLD.Size();

    }

    public double sortAndCalculateMedian(int dimension) {
        int column = dimension-1;
        points.sort(Comparator.comparingDouble(arr -> arr[column]));
        int size = points.size();
        if (points.size()%2 == 1) {
            return points.get((size-1)/2)[column];
        }
        return (points.get((size-2)/2)[column] + points.get((size/2))[column]) / 2;
    }
}
