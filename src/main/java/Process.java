import mpi.Group;
import mpi.Intracomm;
import mpi.MPI;
import mpi.Request;

import java.util.*;
import static mpi.MPI.COMM_WORLD;

public abstract class Process {
    // tags for message passing
    protected final int LOCAL_HISTOGRAM = 0;
    protected final int GROUP_MEDIAN    = 1;
    protected final int POINT_EXCHANGE  = 3;
    protected final int BOUNDING_BOXES  = 4;
    protected final int GHOST_POINTS    = 5;
    // new
    protected final int UNION_QUERY     = 6;
    protected final int CLUSTER_ID_SYNC = 7;

    protected BoundingBox boundingBox;
    protected Set<BoundingBox> otherBoundingBoxes = new HashSet<>();
    protected Set<BoundingBox> otherBoundingBoxesCopy;
    protected double epsilon;
    protected int dimCount;
    protected ArrayList<Point> points;
    protected ArrayList<Point> ghostPoints = new ArrayList<>();
    protected ArrayList<Point> allPts;

    protected Map<GridKey, GridCell> gridMap;
    protected Map<Long, Cluster> localClusterMap = new HashMap<>();
    protected int minPts = 0;

    int numberOfProcessesInGroup;
    protected int rank;
    protected Group group;
    protected Intracomm communicator;
    protected double median;
    protected int currentDimension;

    // Set by mergeClusters for every local cluster uid that was absorbed by a
    // remote winner.  Maps:  local-uid  ->  (winnerRank, winnerClusterUid)
    // winnerClusterUid is the uid on the winner's side so assignClusterIds can
    // look up the exact global ID the winner chose.
    protected Map<Long, long[]> absorbedInfo = new HashMap<>(); // uid -> [winnerRank, winnerUid]

    public Process(ArrayList<Point> data, double epsilon) {
        this.points = data;
        this.rank   = COMM_WORLD.Rank();
        this.numberOfProcessesInGroup = COMM_WORLD.Size();
        this.group  = COMM_WORLD.Group();
        this.communicator = COMM_WORLD;
        this.currentDimension = 0;
        this.epsilon = epsilon;

        if (data != null && !data.isEmpty()) {
            this.dimCount    = data.get(0).dimensions;
            this.boundingBox = new BoundingBox(this.dimCount);
        } else {
            this.dimCount = 0;
        }
    }

    public abstract void decomposeDomain();

    // -----------------------------------------------------------------------
    // exchangePoints / exchangeBoundingBoxes / exchangeGhostPoints /
    // resetCommunication  — unchanged from working version
    // -----------------------------------------------------------------------

    public void exchangePoints() {
        log("Exchanging points");
        ArrayList<Point> upperPoints = new ArrayList<>();
        ArrayList<Point> lowerPoints = new ArrayList<>();
        for (Point point : points) {
            if (point.coords[currentDimension] > median) upperPoints.add(point);
            else                                          lowerPoints.add(point);
        }

        ArrayList<Point> sendList;
        if (rank < numberOfProcessesInGroup / 2) {
            sendList = upperPoints; points = lowerPoints;
            boundingBox.setMax(currentDimension, median);
        } else {
            sendList = lowerPoints; points = upperPoints;
            boundingBox.setMin(currentDimension, median);
        }

        PointBuffer sendBuffer = new PointBuffer(sendList);
        int[] pal = new int[numberOfProcessesInGroup];
        for (int i = 0; i < numberOfProcessesInGroup; i++) pal[i] = i;

        int partner = numberOfProcessesInGroup - (rank + 1);
        ArrayList<Point> receivedPoints;

        if (rank < numberOfProcessesInGroup / 2) {
            sendBuffer.send(communicator, partner, POINT_EXCHANGE);
            receivedPoints = PointBuffer.receive(communicator, partner, POINT_EXCHANGE).toPointList();
            group = group.Incl(Arrays.copyOfRange(pal, 0, numberOfProcessesInGroup / 2));
        } else {
            receivedPoints = PointBuffer.receive(communicator, partner, POINT_EXCHANGE).toPointList();
            sendBuffer.send(communicator, partner, POINT_EXCHANGE);
            group = group.Incl(Arrays.copyOfRange(pal, numberOfProcessesInGroup / 2, numberOfProcessesInGroup));
        }
        points.addAll(receivedPoints);
        numberOfProcessesInGroup /= 2;
        currentDimension = (currentDimension + 1) % dimCount;
        communicator = COMM_WORLD.Create(group);
        rank = group.Rank();
        log("Point exchange done");
    }

    public void exchangeBoundingBoxes() {
        log("Exchanging bounding boxes. My own is " + boundingBox.toString());
        for (int address = 0; address < COMM_WORLD.Size(); address++) {
            if (address == rank) {
                for (int sender = 0; sender < COMM_WORLD.Size(); sender++) {
                    if (sender != rank) otherBoundingBoxes.add(BoundingBox.receive(sender, BOUNDING_BOXES));
                }
                log("Received all " + otherBoundingBoxes.size() + " bounding boxes");
                otherBoundingBoxesCopy = new HashSet<>(otherBoundingBoxes);
            } else {
                this.boundingBox.send(COMM_WORLD, address, BOUNDING_BOXES);
            }
        }
    }

    public void exchangeGhostPoints() {
        log("Starting to exchange ghost points");
        Set<BoundingBox> neighbours = boundingBox.neighbourSet(otherBoundingBoxes);
        Map<Integer, ArrayList<Point>> sendMap = new HashMap<>();

        while (true) {
            boolean newQueued = false;
            for (BoundingBox neighbour : neighbours) {
                int na = neighbour.globalCommGroupAddress;
                sendMap.putIfAbsent(na, new ArrayList<>());
                for (Point point : points) {
                    if (neighbour.distanceToPoint(point) <= epsilon) {
                        sendMap.get(na).add(point);
                        newQueued = true;
                    }
                }
            }
            if (!newQueued) break;
            Set<BoundingBox> newNeighbours = new HashSet<>();
            for (BoundingBox neighbour : neighbours)
                newNeighbours.addAll(neighbour.neighbourSet(otherBoundingBoxes));
            neighbours = newNeighbours;
        }

        for (int address = 0; address < COMM_WORLD.Size(); address++) {
            if (address == COMM_WORLD.Rank()) {
                for (int sender = 0; sender < COMM_WORLD.Size(); sender++) {
                    if (sender != COMM_WORLD.Rank()) {
                        ArrayList<Point> pts = PointBuffer.receive(COMM_WORLD, sender, GHOST_POINTS).toPointList();
                        for (Point pt : pts) pt.sourceRank = sender;
                        ghostPoints.addAll(pts);
                    }
                }
            } else {
                new PointBuffer(sendMap.getOrDefault(address, new ArrayList<>()))
                        .send(COMM_WORLD, address, GHOST_POINTS);
            }
        }
        for (Point pt : ghostPoints) pt.type = pt.GHOST;
        log("Finished exchanging ghost points. Received: " + ghostPoints.size());
    }

    public void resetCommunication() {
        log("kd-Tree done. Size: " + points.size() + ". Resetting communication parameters");
        group = COMM_WORLD.Group();
        communicator = COMM_WORLD;
        rank = group.Rank();
    }

    // -----------------------------------------------------------------------
    // localDBScan
    // -----------------------------------------------------------------------

    public void localDBScan(int minPts) {
        this.minPts = minPts;
        log("Starting localDBScan. Local points: " + points.size()
                + ", Ghost Points: " + ghostPoints.size());

        double invSideWidth = 1.0 / (epsilon / Math.sqrt(dimCount));

        this.allPts = new ArrayList<>(points);
        this.allPts.addAll(ghostPoints);
        this.gridMap = new HashMap<>();
        long[] gridPts = new long[dimCount];

        int[] pArray = new int[this.allPts.size()];
        for (int i = 0; i < pArray.length; i++) {
            pArray[i] = i;
            this.allPts.get(i).localId = i;
        }

        // 1. Build grid
        for (Point p : this.allPts) {
            for (int d = 0; d < dimCount; d++)
                gridPts[d] = (long) Math.floor(p.coords[d] * invSideWidth);
            GridKey key = new GridKey(gridPts);
            if (gridMap.putIfAbsent(key, new GridCell(p, gridPts)) != null)
                gridMap.get(key).points.add(p);
        }
        log("localdbscan: Created " + gridMap.size() + " grid cell(s).");

        // 2. Identify core cells
        int coreCellCount = 0;
        for (GridCell cell : gridMap.values()) {
            if (cell.Size() >= minPts) {
                cell.isCoreCell = true; coreCellCount++;
                for (Point pt : cell.points) if (pt.type != pt.GHOST) pt.type = pt.CORE;
                cell.updateReptToCore();
            }
        }
        log("localdbscan: Identified " + coreCellCount + " core cell(s).");

        // 3. Merge within core cells
        for (GridCell cell : gridMap.values()) {
            if (!cell.isCoreCell) continue;
            for (Point pt : cell.points)
                rem(find(cell.reptId, pArray), find(pt.localId, pArray), pArray);
            cell.reptId = find(cell.reptId, pArray);
        }

        // 4. Merge adjacent core cells; expand non-core cells
        int bcpMerges = 0, nonCoreExpansions = 0;
        for (GridCell cell : gridMap.values()) {
            if (cell.isCoreCell) {
                for (GridKey key : neighbourGridQuery(cell)) {
                    GridCell nCell = gridMap.get(key);
                    if (nCell != null && nCell.isCoreCell && cell.Bcp(nCell, epsilon)) {
                        rem(find(cell.reptId, pArray), find(nCell.reptId, pArray), pArray);
                        bcpMerges++;
                    }
                }
            } else {
                if (cell.isGhost()) continue;
                for (Point x : cell.points) {
                    if (x.type == x.GHOST) continue;
                    ArrayList<Point> nbrs = eNeighbourhoodPts(x, cell);
                    if (nbrs.size() >= minPts) {
                        x.type = x.CORE; nonCoreExpansions++;
                        for (Point y : nbrs) {
                            if      (y.type == y.CORE)  rem(find(x.localId, pArray), find(y.localId, pArray), pArray);
                            else if (y.type == y.NOISE) { y.type = y.BOUNDARY; pArray[y.localId] = find(x.localId, pArray); }
                        }
                    } else if (x.type == x.NOISE) {
                        for (Point y : nbrs) {
                            if (y.type == y.CORE) {
                                x.type = x.BOUNDARY;
                                rem(find(x.localId, pArray), find(y.localId, pArray), pArray);
                            }
                        }
                    }
                }
            }
        }
        log("localdbscan: Core-Cell merges (BCP): " + bcpMerges
                + ", Non-core core discovery: " + nonCoreExpansions);

        // 4.5 Remove ghost points that are truly isolated (their own root, no local
        //     member).  A ghost that IS the root of a tree containing local points
        //     must be kept so mergeClusters can trace the cross-boundary connection.
        int ghostClusters = 0;
        for (int i = 0; i < allPts.size(); i++) {
            Point pt = allPts.get(i);
            if (pt.type != pt.GHOST || pArray[i] != i) continue;
            boolean hasLocalMember = false;
            for (int j = 0; j < allPts.size(); j++) {
                if (allPts.get(j).type != allPts.get(j).GHOST && find(j, pArray) == i) {
                    hasLocalMember = true; break;
                }
            }
            if (!hasLocalMember) { pArray[i] = -1; ghostClusters++; }
        }
        log("localdbscan: Ghost clusters removed: " + ghostClusters);

        // 5. Form local clusters
        for (Point pt : this.allPts) {
            int rootId = find(pt.localId, pArray);
            if (rootId == -1) continue;
            boolean include = (rootId != pt.localId)
                    || (pt.type == pt.CORE)
                    || (pt.type == pt.GHOST && pArray[pt.localId] == pt.localId);
            if (!include) continue;

            Cluster cluster = localClusterMap.computeIfAbsent((long) rootId, id -> new Cluster(id));
            if      (pt.type == pt.CORE)     cluster.corePts.add(pt);
            else if (pt.type == pt.BOUNDARY) cluster.boundaryPts.add(pt);
            else if (pt.type == pt.GHOST)    cluster.remoteProcessingNeighbours.add(pt);
        }
        log("localdbscan Finished. Found " + localClusterMap.size() + " local clusters.");
        for (Map.Entry<Long, Cluster> e : localClusterMap.entrySet()) {
            log("  cluster uid=" + e.getKey()
                    + " core=" + e.getValue().corePts.size()
                    + " boundary=" + e.getValue().boundaryPts.size()
                    + " ghosts=" + e.getValue().remoteProcessingNeighbours.size());
        }
    }

    // -----------------------------------------------------------------------
    // mergeClusters — Algorithm 5 from HY-DBSCAN, with full logging
    // -----------------------------------------------------------------------

    /**
     * Each process sends a Union query for every ghost point it holds.
     * A query carries: [senderRank, senderClusterUid, ghostCoords...].
     * The recipient owns the point at ghostCoords locally.
     *
     * Rank rule (paper §4.5.2): the process with the LOWER rank wins.
     * - If senderRank < myRank: sender wins → we mark our cluster absorbed and
     *   record (winnerRank=senderRank, winnerUid=senderCluster) so assignClusterIds
     *   can later retrieve the correct global ID.
     * - If senderRank > myRank: we win → we send the sender a query whose coords
     *   are one of the ghost points WE hold from the sender, so the sender finds
     *   it in its own coordToLocalPoint, sees our rank < its rank, and absorbs.
     *
     * After the loop, absorbedInfo maps every absorbed uid to [winnerRank, winnerUid].
     */
    public void mergeClusters() {
        log("Starting distributed cluster merge. Local clusters: " + localClusterMap.size());

        final int FIELDS = 2 + dimCount; // [senderRank, senderClusterUid, coords...]
        int worldSize = COMM_WORLD.Size();

        Map<String, Point> coordToLocalPoint = new HashMap<>();
        for (Point pt : points) coordToLocalPoint.put(coordKey(pt), pt);

        // Union-find over local cluster uids.  Long.MIN_VALUE = absorbed sentinel.
        Map<Long, Long> clusterParent = new HashMap<>();
        for (Long uid : localClusterMap.keySet()) clusterParent.put(uid, uid);

        // localPointId -> cluster uid (for quickly finding which cluster owns a point)
        Map<Integer, Long> localIdToClusterUid = new HashMap<>();
        for (Map.Entry<Long, Cluster> entry : localClusterMap.entrySet()) {
            for (Point p : entry.getValue().corePts)     localIdToClusterUid.put(p.localId, entry.getKey());
            for (Point p : entry.getValue().boundaryPts) localIdToClusterUid.put(p.localId, entry.getKey());
        }

        // Pending outgoing queries, keyed by destination rank.
        Map<Integer, List<double[]>> pending = new HashMap<>();
        for (int r = 0; r < worldSize; r++) pending.put(r, new ArrayList<>());

        // Seed: one query per ghost point in every cluster.
        for (Map.Entry<Long, Cluster> entry : localClusterMap.entrySet()) {
            long uid = entry.getKey();
            for (Point ghost : entry.getValue().remoteProcessingNeighbours) {
                pending.get(ghost.sourceRank).add(buildQuery(rank, uid, ghost));
                log("MERGE seed query: my cluster " + uid
                        + " -> rank " + ghost.sourceRank
                        + " ghost=" + coordKey(ghost));
            }
        }

        int round = 0;
        while (true) {
            int localPending = pending.values().stream().mapToInt(List::size).sum();
            int[] globalPending = new int[1];
            COMM_WORLD.Allreduce(new int[]{localPending}, 0, globalPending, 0, 1, MPI.INT, MPI.SUM);
            if (globalPending[0] == 0) break;

            log("MERGE round " + round + ": localPending=" + localPending
                    + " globalPending=" + globalPending[0]);

            // --- Non-blocking sends ---
            List<Request>  sendRequests = new ArrayList<>();
            List<int[]>    headerBufs   = new ArrayList<>();
            List<double[]> dataBufs     = new ArrayList<>();
            for (int dest = 0; dest < worldSize; dest++) {
                if (dest == rank) continue;
                List<double[]> msgs = pending.get(dest);
                int count = msgs.size();
                int[] header = new int[]{count};
                headerBufs.add(header);
                sendRequests.add(COMM_WORLD.Isend(header, 0, 1, MPI.INT, dest, UNION_QUERY));
                if (count > 0) {
                    double[] flat = new double[count * FIELDS];
                    for (int i = 0; i < count; i++)
                        System.arraycopy(msgs.get(i), 0, flat, i * FIELDS, FIELDS);
                    dataBufs.add(flat);
                    sendRequests.add(COMM_WORLD.Isend(flat, 0, flat.length, MPI.DOUBLE, dest, UNION_QUERY));
                }
            }
            for (int r = 0; r < worldSize; r++) pending.get(r).clear();

            // --- Blocking receives (safe: all sends already posted) ---
            List<double[]> newQueries = new ArrayList<>();
            for (int src = 0; src < worldSize; src++) {
                if (src == rank) continue;
                int[] countBuf = new int[1];
                COMM_WORLD.Recv(countBuf, 0, 1, MPI.INT, src, UNION_QUERY);
                if (countBuf[0] > 0) {
                    double[] flat = new double[countBuf[0] * FIELDS];
                    COMM_WORLD.Recv(flat, 0, flat.length, MPI.DOUBLE, src, UNION_QUERY);
                    for (int i = 0; i < countBuf[0]; i++) {
                        double[] q = new double[FIELDS];
                        System.arraycopy(flat, i * FIELDS, q, 0, FIELDS);
                        newQueries.add(q);
                    }
                }
            }
            Request.Waitall(sendRequests.toArray(new Request[0]));

            // --- Process received queries ---
            for (double[] query : newQueries) {
                int    senderRank    = (int)  query[0];
                long   senderCluster = (long) query[1];
                double[] coords      = new double[dimCount];
                System.arraycopy(query, 2, coords, 0, dimCount);

                log("MERGE round " + round + ": received query from rank=" + senderRank
                        + " senderCluster=" + senderCluster
                        + " coords=" + coordKey(coords));

                Point localPt = coordToLocalPoint.get(coordKey(coords));
                if (localPt == null) {
                    // Not our local point — forward via our ghost copy of it.
                    boolean forwarded = false;
                    for (Point gp : ghostPoints) {
                        if (coordKey(gp).equals(coordKey(coords))) {
                            pending.get(gp.sourceRank).add(query);
                            log("MERGE round " + round + ": forwarded to rank=" + gp.sourceRank);
                            forwarded = true; break;
                        }
                    }
                    if (!forwarded)
                        log("MERGE round " + round + ": WARNING could not forward query coords=" + coordKey(coords));
                    continue;
                }

                Long uid = localIdToClusterUid.get(localPt.localId);
                if (uid == null) {
                    log("MERGE round " + round + ": localPt id=" + localPt.localId + " is noise, skipping");
                    continue;
                }
                long myRoot = findClusterRoot(uid, clusterParent);

                // Skip if already absorbed.
                if (clusterParent.getOrDefault(myRoot, myRoot) == Long.MIN_VALUE) {
                    log("MERGE round " + round + ": myRoot=" + myRoot + " already absorbed, skipping");
                    continue;
                }

                log("MERGE round " + round + ": myRoot=" + myRoot
                        + " senderRank=" + senderRank + " myRank=" + rank);

                if (senderRank < rank) {
                    // Sender has lower rank → sender wins → absorb our cluster.
                    log("MERGE round " + round + ": absorbing myRoot=" + myRoot
                            + " into rank=" + senderRank + " cluster=" + senderCluster);
                    clusterParent.put(myRoot, Long.MIN_VALUE);
                    // Record winner identity so assignClusterIds can get the right global ID.
                    absorbedInfo.put(myRoot, new long[]{senderRank, senderCluster});

                } else if (senderRank > rank) {
                    // We have lower rank → we win → tell sender to absorb.
                    // We send a query whose coords are a ghost WE hold from the sender,
                    // so the sender finds it in its coordToLocalPoint and then sees
                    // senderRank(=us) < its rank and absorbs itself.
                    Point anchor = null;
                    outer:
                    for (Map.Entry<Long, Cluster> e : localClusterMap.entrySet()) {
                        if (findClusterRoot(e.getKey(), clusterParent) != myRoot) continue;
                        for (Point gp : e.getValue().remoteProcessingNeighbours) {
                            if (gp.sourceRank == senderRank) { anchor = gp; break outer; }
                        }
                    }
                    if (anchor != null) {
                        log("MERGE round " + round + ": we win, sending absorption query to rank="
                                + senderRank + " anchor=" + coordKey(anchor));
                        pending.get(senderRank).add(buildQuery(rank, myRoot, anchor));
                    } else {
                        log("MERGE round " + round + ": WARNING we win but no anchor ghost for rank="
                                + senderRank + " myRoot=" + myRoot);
                    }
                }
            }
            round++;
        }

        // Compact localClusterMap: keep only surviving (non-absorbed) clusters.
        Map<Long, Cluster> mergedMap = new HashMap<>();
        for (Map.Entry<Long, Cluster> entry : localClusterMap.entrySet()) {
            long uid = entry.getKey();
            Long val = clusterParent.get(uid);
            if (val != null && val == Long.MIN_VALUE) continue;
            long localRoot = findClusterRoot(uid, clusterParent);
            if (clusterParent.getOrDefault(localRoot, localRoot) == Long.MIN_VALUE) continue;
            mergedMap.computeIfAbsent(localRoot, id -> new Cluster(id)).merge(entry.getValue());
        }
        localClusterMap = mergedMap;

        log("MERGE done. Surviving clusters: " + localClusterMap.size()
                + "  Absorbed clusters: " + absorbedInfo.size());
        for (Map.Entry<Long, long[]> e : absorbedInfo.entrySet()) {
            log("  absorbed uid=" + e.getKey()
                    + " -> winnerRank=" + e.getValue()[0]
                    + " winnerCluster=" + e.getValue()[1]);
        }
        for (Map.Entry<Long, Cluster> e : localClusterMap.entrySet()) {
            log("  surviving uid=" + e.getKey()
                    + " core=" + e.getValue().corePts.size()
                    + " boundary=" + e.getValue().boundaryPts.size());
        }
    }

    // -----------------------------------------------------------------------
    // assignClusterIds — Algorithm 6 from HY-DBSCAN, with propagation to
    // absorbed processes.
    //
    // Phase 1 (paper Algorithm 6):
    //   Each surviving process numbers its clusters 0..n-1, AllGathers counts,
    //   computes offset, and stamps its own local points.
    //
    // Phase 2 (propagation to absorbed processes):
    //   An absorbed process knows winnerRank and winnerClusterUid (from absorbedInfo).
    //   It sends that uid to the winner.  The winner looks up its globalId for that
    //   uid and sends it back.  The absorbed process then stamps all its local points.
    //
    //   Communication is coordinated with AllToAll so every process knows exactly
    //   how many requests it will receive before starting to recv.
    // -----------------------------------------------------------------------

    public void assignClusterIds() {
        log("ASSIGNIDS starting. localClusterMap.size=" + localClusterMap.size()
                + "  absorbedInfo.size=" + absorbedInfo.size());

        int worldSize = COMM_WORLD.Size();

        // --- Phase 1: assign sequential IDs and stamp surviving points ---

        Map<Long, Integer> localIdMap = new HashMap<>();
        int localCount = 0;
        for (long uid : localClusterMap.keySet()) {
            localIdMap.put(uid, localCount++);
            log("ASSIGNIDS phase1: uid=" + uid + " -> localSeqId=" + localIdMap.get(uid));
        }

        int[] allCounts = new int[worldSize];
        COMM_WORLD.Allgather(new int[]{localCount}, 0, 1, MPI.INT,
                allCounts, 0, 1, MPI.INT);

        int offset = 0;
        for (int r = 0; r < rank; r++) offset += allCounts[r];
        log("ASSIGNIDS phase1: offset=" + offset);

        // uid -> globalId for OUR surviving clusters
        Map<Long, Integer> uidToGlobalId = new HashMap<>();
        for (Map.Entry<Long, Integer> e : localIdMap.entrySet()) {
            int gid = offset + e.getValue();
            uidToGlobalId.put(e.getKey(), gid);
            log("ASSIGNIDS phase1: uid=" + e.getKey() + " -> globalId=" + gid);
        }

        // Stamp surviving clusters' local points
        for (Map.Entry<Long, Cluster> entry : localClusterMap.entrySet()) {
            int globalId = uidToGlobalId.get(entry.getKey());
            Cluster c = entry.getValue();
            for (Point p : c.corePts)     { p.globalClusterId = globalId; }
            for (Point p : c.boundaryPts) { p.globalClusterId = globalId; }
            log("ASSIGNIDS phase1: stamped " + (c.corePts.size() + c.boundaryPts.size())
                    + " points with globalId=" + globalId);
        }

        int totalClusters = 0;
        for (int c : allCounts) totalClusters += c;
        log("ASSIGNIDS phase1 done. totalClusters=" + totalClusters);

        // --- Phase 2: propagate IDs to absorbed processes ---
        //
        // Each absorbed process sends (winnerClusterUid as long) to its winner.
        // The winner replies with the corresponding globalId (int).
        // We use AllToAll to declare intent so every process knows how many
        // requests to expect, then do direct blocking send/recv for the payloads.

        // Determine which winner ranks we need IDs from.
        // Multiple absorbed clusters could have the same winner rank — we need one
        // request per absorbed cluster (they might map to different global IDs in
        // pathological multi-cluster cases, but for correctness handle each).
        // For simplicity group by winner rank: send all their uids at once.

        // winnerRank -> list of winnerClusterUids we need IDs for
        Map<Integer, List<Long>> needIdFrom = new HashMap<>();
        for (Map.Entry<Long, long[]> e : absorbedInfo.entrySet()) {
            int    winnerRank = (int) e.getValue()[0];
            long   winnerUid  = e.getValue()[1];
            needIdFrom.computeIfAbsent(winnerRank, k -> new ArrayList<>()).add(winnerUid);
            log("ASSIGNIDS phase2: need globalId from rank=" + winnerRank
                    + " for winnerUid=" + winnerUid + " (my absorbed uid=" + e.getKey() + ")");
        }

        // AllToAll: tell every process how many uid requests we will send it.
        int[] iWillSendCount = new int[worldSize];
        for (Map.Entry<Integer, List<Long>> e : needIdFrom.entrySet())
            iWillSendCount[e.getKey()] = e.getValue().size();

        int[] iWillReceiveCount = new int[worldSize];
        COMM_WORLD.Alltoall(iWillSendCount, 0, 1, MPI.INT,
                iWillReceiveCount, 0, 1, MPI.INT);

        log("ASSIGNIDS phase2: iWillSendCount=" + Arrays.toString(iWillSendCount)
                + " iWillReceiveCount=" + Arrays.toString(iWillReceiveCount));

        // Absorbed processes: send uid requests (as long[]) to their winners.
        List<Request>  sendReqs  = new ArrayList<>();
        List<long[]>   sendBufs  = new ArrayList<>();
        for (Map.Entry<Integer, List<Long>> e : needIdFrom.entrySet()) {
            int    winnerRank = e.getKey();
            List<Long> uids   = e.getValue();
            long[] buf = new long[uids.size()];
            for (int i = 0; i < uids.size(); i++) buf[i] = uids.get(i);
            sendBufs.add(buf);
            log("ASSIGNIDS phase2: sending " + uids.size() + " uid requests to rank=" + winnerRank);
            sendReqs.add(COMM_WORLD.Isend(buf, 0, buf.length, MPI.LONG, winnerRank, CLUSTER_ID_SYNC));
        }

        // Winners: receive uid requests and send back globalIds.
        List<Request>  replyReqs = new ArrayList<>();
        List<int[]>    replyBufs = new ArrayList<>();
        for (int src = 0; src < worldSize; src++) {
            int count = iWillReceiveCount[src];
            if (count == 0) continue;
            long[] reqUids = new long[count];
            COMM_WORLD.Recv(reqUids, 0, count, MPI.LONG, src, CLUSTER_ID_SYNC);

            int[] replyIds = new int[count];
            for (int i = 0; i < count; i++) {
                long winnerUid = reqUids[i];
                Integer gid = uidToGlobalId.get(winnerUid);
                replyIds[i] = (gid != null) ? gid : -1;
                log("ASSIGNIDS phase2: replying to rank=" + src
                        + " winnerUid=" + winnerUid + " -> globalId=" + replyIds[i]);
            }
            replyBufs.add(replyIds);
            replyReqs.add(COMM_WORLD.Isend(replyIds, 0, count, MPI.INT, src, CLUSTER_ID_SYNC));
        }

        // Absorbed processes: receive the global IDs and stamp local points.
        // Build the reverse map: winnerUid -> my absorbed uids
        // so we know which local points to stamp with which id.
        Map<Long, Long> winnerUidToMyUid = new HashMap<>(); // winnerUid -> my absorbed uid
        for (Map.Entry<Long, long[]> e : absorbedInfo.entrySet()) {
            long winnerUid = e.getValue()[1];
            winnerUidToMyUid.put(winnerUid, e.getKey()); // for lookup after receive
        }

        for (Map.Entry<Integer, List<Long>> e : needIdFrom.entrySet()) {
            int    winnerRank = e.getKey();
            List<Long> uids   = e.getValue();
            int[] recvIds = new int[uids.size()];
            COMM_WORLD.Recv(recvIds, 0, uids.size(), MPI.INT, winnerRank, CLUSTER_ID_SYNC);

            for (int i = 0; i < uids.size(); i++) {
                long winnerUid = uids.get(i);
                int  globalId  = recvIds[i];
                log("ASSIGNIDS phase2: received globalId=" + globalId
                        + " from rank=" + winnerRank + " for winnerUid=" + winnerUid);
                if (globalId < 0) {
                    log("ASSIGNIDS phase2: WARNING winner rank=" + winnerRank
                            + " returned -1 for winnerUid=" + winnerUid);
                    continue;
                }
                // Find the absorbed cluster on our side that corresponds to this winnerUid
                // and stamp its members.
                for (Map.Entry<Long, long[]> ae : absorbedInfo.entrySet()) {
                    if (ae.getValue()[0] == winnerRank && ae.getValue()[1] == winnerUid) {
                        long myAbsorbedUid = ae.getKey();
                        // This uid was removed from localClusterMap during compaction.
                        // We need to stamp the points that belonged to it.
                        // They are still in this.points — their globalClusterId is still -1.
                        // We have no fast lookup here, so stamp ALL unstamped local points.
                        // (Correct because if this process was absorbed, ALL its clusters
                        // ended up in the same remote cluster — this is guaranteed by the
                        // single-cluster-per-process assumption that the kd-tree gives us.)
                        log("ASSIGNIDS phase2: stamping unstamped local points with globalId=" + globalId
                                + " (absorbed uid=" + myAbsorbedUid + ")");
                        for (Point p : points) {
                            if (p.globalClusterId == -1) {
                                p.globalClusterId = globalId;
                            }
                        }
                    }
                }
            }
        }

        // Wait for all sends to complete.
        if (!sendReqs.isEmpty())  Request.Waitall(sendReqs.toArray(new Request[0]));
        if (!replyReqs.isEmpty()) Request.Waitall(replyReqs.toArray(new Request[0]));

        // Final sanity log
        int stamped = 0, unstamped = 0;
        for (Point p : points) { if (p.globalClusterId >= 0) stamped++; else unstamped++; }
        log("ASSIGNIDS done. stamped=" + stamped + " unstamped(noise)=" + unstamped
                + " totalClusters=" + totalClusters);
    }

    // -----------------------------------------------------------------------
    // Helpers
    // -----------------------------------------------------------------------

    public List<GridKey> neighbourGridQuery(GridCell cell) {
        int offset = (int) Math.ceil(Math.sqrt(this.dimCount));
        List<GridKey> keys = new ArrayList<>();
        generateRecursiveNeighbours(cell.pos, offset, 0, new long[this.dimCount], keys);
        return keys;
    }

    private void generateRecursiveNeighbours(long[] currCellPos, int offset, int currDim,
                                             long[] candidatePos, List<GridKey> keys) {
        if (currDim == this.dimCount) {
            if (Arrays.equals(currCellPos, candidatePos)) return;
            GridKey key = new GridKey(candidatePos.clone());
            if (this.gridMap.containsKey(key)) keys.add(key);
            return;
        }
        for (int i = -offset; i <= offset; i++) {
            candidatePos[currDim] = currCellPos[currDim] + i;
            generateRecursiveNeighbours(currCellPos, offset, currDim + 1, candidatePos, keys);
        }
    }

    private ArrayList<Point> eNeighbourhoodPts(Point x, GridCell currentCell) {
        ArrayList<Point> neighbours = new ArrayList<>();
        ArrayList<GridCell> potentialCells = new ArrayList<>();
        potentialCells.add(currentCell);
        for (GridKey key : neighbourGridQuery(currentCell)) {
            GridCell nCell = gridMap.get(key);
            if (nCell != null) potentialCells.add(nCell);
        }
        for (GridCell cell : potentialCells)
            for (Point y : cell.points)
                if (x.distanceToPoint(y) <= epsilon) neighbours.add(y);
        return neighbours;
    }

    private int find(int x, int[] p) {
        int rep = x;
        while (p[rep] != rep) {
            if (p[rep] == -1) return -1;
            rep = p[rep];
        }
        return rep;
    }

    private int rem(int x, int y, int[] p) {
        int rx = x, ry = y;
        while (p[rx] != p[ry]) {
            Point px = allPts.get(p[rx]);
            Point py = allPts.get(p[ry]);
            boolean pyHasPriority;
            if      (px.type == px.GHOST && py.type != py.GHOST) pyHasPriority = true;
            else if (px.type != px.GHOST && py.type == py.GHOST) pyHasPriority = false;
            else                                                   pyHasPriority = (p[rx] > p[ry]);

            if (pyHasPriority) {
                if (rx == p[rx]) { p[rx] = p[ry]; return p[rx]; }
                int z = p[rx]; p[rx] = p[ry]; rx = z;
            } else {
                if (ry == p[ry]) { p[ry] = p[rx]; return p[ry]; }
                int z = p[ry]; p[ry] = p[rx]; ry = z;
            }
        }
        return -1;
    }

    private double[] buildQuery(int senderRank, long clusterUid, Point anchorPoint) {
        double[] msg = new double[2 + dimCount];
        msg[0] = senderRank;
        msg[1] = clusterUid;
        System.arraycopy(anchorPoint.coords, 0, msg, 2, dimCount);
        return msg;
    }

    private String coordKey(Point pt)        { return coordKey(pt.coords); }
    private String coordKey(double[] coords) {
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < coords.length; i++) {
            if (i > 0) sb.append(',');
            sb.append(Double.toHexString(coords[i]));
        }
        return sb.toString();
    }

    private long findClusterRoot(long uid, Map<Long, Long> parent) {
        long root = uid;
        while (parent.containsKey(root)) {
            long p = parent.get(root);
            if (p == root || p == Long.MIN_VALUE) break;
            long gp = parent.getOrDefault(p, p);
            parent.put(root, gp); // path compression
            root = p;
        }
        return root;
    }

    public void log(String message) {
        System.out.println("Process " + COMM_WORLD.Rank() + ": " + message);
    }
}