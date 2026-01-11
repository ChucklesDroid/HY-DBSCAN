public class RankValuePair {
    int rank;
    double value;
    public RankValuePair(int rank, double value) {
        this.rank = rank;
        this.value = value;
    }

    public int compareTo(RankValuePair other) {
        return Double.compare(this.value, other.value);
    }
}
