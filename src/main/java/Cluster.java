import java.util.ArrayList;
import java.util.HashSet;
import java.util.Set;

public class Cluster {
    long uid;

    ArrayList<Point> corePts = new ArrayList<>();
    ArrayList<Point> boundaryPts = new ArrayList<>();

    //TODO: how to handle merging of clusters across processes (subject to change)
    ArrayList<Point> remoteProcessingNeighbours = new ArrayList<>();

    public Cluster(long uid) {
        this.uid = uid;
        corePts = new ArrayList<>();
        boundaryPts = new ArrayList<>();
        remoteProcessingNeighbours = new ArrayList<>();
    }

    void merge(Cluster other) {
        this.corePts.addAll(other.corePts);
        this.boundaryPts.addAll(other.boundaryPts);
        this.remoteProcessingNeighbours.addAll(other.remoteProcessingNeighbours);
    }

}
