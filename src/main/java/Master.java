import mpi.MPI;
import java.util.ArrayList;
import static mpi.MPI.COMM_WORLD;

public class Master extends Process {
    public Master(ArrayList<Point> data, double epsilon) {
        super(data, epsilon);
    }

    @Override
    public void decomposeDomain() {
        log("Starting kd-tree");
        int depth = (int) (Math.log(COMM_WORLD.Size()) / Math.log(2));

        for (int i = 0; i < depth; i++) {
            findAndSendGroupMedian();
            exchangePoints();
        }
        resetCommunication();
    }

    //TODO: check for static usage
    public void findAndSendGroupMedian() {
        log("(Master) Calculating histograms");
        Histogram hist = new Histogram(currentDimension, points);
        log("(Master) Recieving histograms");
        for (int sender = 1; sender < numberOfProcessesInGroup; sender++) {
            hist.receiveAndMerge(sender, LOCAL_HISTOGRAM, communicator);
        }
        log("(Master) Calculating median");
        median = hist.determineSlidingWindowMedian(epsilon);

        log("(Master) Broadcasting median");
        double[] m = new double[]{median};
        communicator.Bcast(m, 0, 1, MPI.DOUBLE, 0);
        log("(Master) Broadcasted median" + median);
    }
}
