import java.util.ArrayList;

public class GridCell {
    ArrayList<Point> points = new ArrayList<>();
    boolean isCoreCell = false;
    long[] pos;
    int reptId = -1;

    public GridCell(ArrayList<Point> data, long[] pos, boolean isCoreCell) {
        this.points = data;
        this.pos = pos;
        this.isCoreCell = isCoreCell;
        setReptId();
    }

    public GridCell(Point pt, long[] pos) {
        this.points = new ArrayList<>();
        this.points.add(pt);
        this.isCoreCell = false;
        this.pos = pos;
        setReptId();
    }

    public int Size() {
        return this.points.size();
    }

    public void setReptId() {
        if (this.Size() == 0) 
            return;
        this.reptId = this.points.get(0).localId;
    }

    public void updateReptToCore() {
        for (Point pt: points) {
            if ((pt.type & pt.CORE) != 0 && (pt.type & pt.GHOST) == 0) {
                this.reptId = pt.localId;
                return;
            }
        }
    }

    //TODO: can improve this by pruning it more using redundancy checks
    //returns true on finding Bi-Chromatic closest pair b/w 2 core cells
    public boolean Bcp(GridCell other, double epsilon) {
        for (Point x: this.points) {
            if ((x.type & x.CORE) != 0) {
                for (Point y: other.points) {
                    if ((y.type & y.CORE) != 0) {
                        if (x.distanceToPoint(y) <= epsilon) {
                            return true;
                        }
                    }
                }
            }
        }
        return false;
    }
}
