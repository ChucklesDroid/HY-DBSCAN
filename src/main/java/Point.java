public class Point {
    double[] coords;
    int dimensions;
    int type;
    int localId;

    // useful for ghost points, default value is -1 i.e belong to the same process
    int sourceRank;

    int NOISE = 1;
    int BOUNDARY = 2;
    int CORE = 4;
    int GHOST = 8;

    public Point(double[] coords) {
        this.coords = coords;
        this.dimensions = coords.length;
        this.type = this.NOISE;
        this.localId = 0;
    }

    // used for points located on the same process.
    public double distanceToPoint(Point other) {
        double sum = 0.0;
        for (int i = 0; i < this.dimensions; i++) {
            double diff = coords[i] - other.coords[i];
            sum += Math.pow(diff, 2);
        }
        return sum;
    }
}
