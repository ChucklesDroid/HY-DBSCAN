import mpi.MPI;
import java.util.ArrayList;
import static mpi.MPI.COMM_WORLD;

public class Worker extends Process {
    public Worker(ArrayList<Point> data, double epsilon) {
        super(data, epsilon);
    }

    @Override
    public void decomposeDomain() {
        log("Starting kd-tree");
        int depth = (int) (Math.log(COMM_WORLD.Size()) / Math.log(2));

        for (int i = 0; i < depth; i++) {
            if (this.rank == 0) {
                findAndSendGroupMedian();
            } else {
                calculateAndSendHistogram();
                receiveGroupMedian();
            }
            exchangePoints();
        }
        resetCommunication();
    }

    private void calculateAndSendHistogram() {
        log("(Worker) Calculating histograms");
        Histogram hist = new Histogram(currentDimension, points);
        hist.send(0, LOCAL_HISTOGRAM, communicator);
    }

    private void receiveGroupMedian() {
        log("(Worker) Received median");
        double[] medianReceiver = new double[1];
        communicator.Bcast(medianReceiver, 0, 1, MPI.DOUBLE, 0);
        median = medianReceiver[0];
    }

    // Helper function when new master is created in the subgroup
    private void findAndSendGroupMedian() {
        log("(Master) Calculating histograms");
        Histogram hist = new Histogram(currentDimension, points);
        for (int sender = 1; sender < numberOfProcessesInGroup; sender++) {
            hist.receiveAndMerge(sender, LOCAL_HISTOGRAM, communicator);
        }
        median = hist.determineSlidingWindowMedian(epsilon);
        communicator.Bcast(new double[]{median}, 0, 1, MPI.DOUBLE, 0);
    }
}
