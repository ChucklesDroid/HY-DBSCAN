import java.util.Arrays;

class GridKey {
    long[] pos;
    public GridKey(long[] pos) {
        this.pos = pos.clone();
    }

    @Override
    public boolean equals(Object o) {
        if (this == o)
            return true;
        if (!(o instanceof GridKey))
            return false;
        GridKey other = (GridKey) o;
        return Arrays.equals(this.pos, other.pos);
    }

    @Override
    public int hashCode() {
        return Arrays.hashCode(this.pos);
    }
}
