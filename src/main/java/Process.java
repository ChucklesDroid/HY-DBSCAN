import mpi.Group;
import mpi.Intracomm;
import mpi.MPI;
import mpi.Request;

import java.util.*;
import static mpi.MPI.COMM_WORLD;

public abstract class Process {
    // tags for message passing
    protected final int LOCAL_HISTOGRAM = 0;
    protected final int GROUP_MEDIAN = 1;
    protected final int POINT_EXCHANGE = 3;
    protected final int BOUNDING_BOXES = 4;
    protected final int GHOST_POINTS = 5;

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
    protected int minPts = 0; // set when localDBScan is called; used for result writing

    int numberOfProcessesInGroup; // number of processes in the current node of the kd-tree
    protected int rank; // rank in the current group
    protected Group group;
    protected Intracomm communicator;
    protected double median;
    protected int currentDimension;

    public Process(ArrayList<Point> data, double epsilon) {
        this.points = data;
        this.rank = COMM_WORLD.Rank();
        this.numberOfProcessesInGroup = COMM_WORLD.Size();
        this.group = COMM_WORLD.Group();
        this.communicator = COMM_WORLD;
        this.currentDimension = 0;
        this.epsilon = epsilon;

        if (data != null && !data.isEmpty()) {
            this.dimCount = data.get(0).dimensions;
            this.boundingBox = new BoundingBox(this.dimCount);
        } else {
            this.dimCount = 0;
        }
    }

    public abstract void decomposeDomain();

    public void exchangePoints() {
        log("Exchanging points");

        ArrayList<Point> upperPoints = new ArrayList<>();
        ArrayList<Point> lowerPoints = new ArrayList<>();

        log("Looking for points to exchange");
        for (Point point : points) {
            if (point.coords[currentDimension] > median) {
                upperPoints.add(point);
            } else {
                lowerPoints.add(point);
            }
        }

        log("Sending and receiving points");
        ArrayList<Point> sendList;
        if (rank < numberOfProcessesInGroup /2) {
            log("Sending higher points");
            sendList = upperPoints;
            points = lowerPoints;
            boundingBox.setMax(currentDimension, median);
        }else {
            log("Sending lower points");
            sendList = lowerPoints;
            points = upperPoints;
            boundingBox.setMin(currentDimension, median);
        }
        ArrayList<Point> receivedPoints;
        PointBuffer sendBuffer = new PointBuffer(sendList);

        int[] processAddressList = new int[numberOfProcessesInGroup];
        for (int i = 0; i < numberOfProcessesInGroup; i++) {
            processAddressList[i] = i;
        }

        int partnerProcess = numberOfProcessesInGroup - (rank + 1); // processes on the edges exchange with each other
        if (rank < numberOfProcessesInGroup /2) { // lower process sends first, then receives. Prevents interlocking
            log("Sending to and then receiving from group partner " + partnerProcess);
            sendBuffer.send(communicator, partnerProcess, POINT_EXCHANGE);
            log("Sent " + sendList.size() + " points");
            receivedPoints = PointBuffer.receive(communicator, partnerProcess, POINT_EXCHANGE).toPointList();
            log("Received " + receivedPoints.size() + " points");

            //Preparation for next round
            group = group.Incl(Arrays.copyOfRange(processAddressList, 0, numberOfProcessesInGroup /2)); // define new communication group
        } else {
            log("Receiving from and then sending to group partner " + partnerProcess);
            receivedPoints = PointBuffer.receive(communicator, partnerProcess, POINT_EXCHANGE).toPointList();
            log("Received " + receivedPoints.size() + " points");
            sendBuffer.send(communicator, partnerProcess, POINT_EXCHANGE);
            log("Sent " + sendList.size() + " points");

            //Preparation for next round
            group = group.Incl(Arrays.copyOfRange(processAddressList, numberOfProcessesInGroup /2, numberOfProcessesInGroup)); // define new communication group
        }
        points.addAll(receivedPoints); // gather all received points

        // Preparing next round

        numberOfProcessesInGroup /= 2; // can now only see half of the previous processes
        currentDimension = (currentDimension + 1) % dimCount;
        communicator = COMM_WORLD.Create(group);
        rank = group.Rank();

        log("Point exchange done");
    }

    public void exchangeBoundingBoxes(){
        log("Exchanging bounding boxes. My own is " + boundingBox.toString());
        for (int address = 0; address < COMM_WORLD.Size(); address++) {
            if (address == rank) {
                log("Gathering bounding boxes");
                for (int sender = 0; sender < COMM_WORLD.Size(); sender++) {
                    if (sender != rank) {
                        BoundingBox other = BoundingBox.receive(sender, BOUNDING_BOXES);
                        otherBoundingBoxes.add(other);
                    }
                }
                log("Received all " + otherBoundingBoxes.size() + " bounding boxes");
                otherBoundingBoxesCopy = new HashSet<>(otherBoundingBoxes);
            } else {
                log("Sending bounding box to process " + address);
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
                int neighbourAddress = neighbour.globalCommGroupAddress;
                sendMap.putIfAbsent(neighbourAddress, new ArrayList<>());

                for (Point point : points) {
                    if (neighbour.distanceToPoint(point) <= epsilon) {
                        sendMap.get(neighbourAddress).add(point);
                        newQueued = true;
                    }
                }
            }
            if (!newQueued) {
                break;
            }
            Set<BoundingBox> newNeighbours = new HashSet<>();
            for (BoundingBox neighbour: neighbours) {
                newNeighbours.addAll(neighbour.neighbourSet(otherBoundingBoxes));
            }
            neighbours = newNeighbours;
        }

        for (int address = 0; address < COMM_WORLD.Size(); address++) {
            if (address == COMM_WORLD.Rank()) {
                for (int sender = 0; sender < COMM_WORLD.Size(); sender++) {
                    if (sender != COMM_WORLD.Rank()) {
                        ArrayList<Point> pts = PointBuffer.receive(COMM_WORLD, sender, GHOST_POINTS).toPointList();
                        for (Point pt: pts) {
                            pt.sourceRank = sender;
                        }
                        ghostPoints.addAll(pts);
                    }
                }
            } else {
                new PointBuffer(sendMap.getOrDefault(address, new ArrayList<>())).send(COMM_WORLD, address, GHOST_POINTS);
            }
        }

        for (Point pt : ghostPoints) {
            pt.type = pt.GHOST;
        }

        log("Finished exchanging ghost points. Received: " + ghostPoints.size());
        log("Epsilon is " + epsilon);
    }

    public void resetCommunication() {
        log("kd-Tree done. Size: " + points.size() + ". Resetting communication parameters");
        group = COMM_WORLD.Group();
        communicator = COMM_WORLD;
        rank = group.Rank();
    }

    public void localDBScan(int minPts) {
        this.minPts = minPts;
        log("Starting localDBScan. Local points: " + points.size() + ", Ghost Points: " + ghostPoints.size());

        // double invSideWidth = 1.0 / (epsilon / Math.sqrt(dimCount));
        double invSideWidth = 1.0 / epsilon;

        this.allPts = new ArrayList<>(points);
        this.allPts.addAll(ghostPoints);
        this.gridMap = new HashMap<>();
        long[] gridPts = new long[dimCount];

        // Assigning localId to points and making the points point to itself.
        // pArray:- acts as an intermediate union find tree before creation of actual clusters.
        int[] pArray = new int[this.allPts.size()];
        for (int i = 0; i < pArray.length; i++) {
            pArray[i] = i;
            this.allPts.get(i).localId = i;
        }

        // 1. Creating Grids and allocating the points into it.
        for (Point p: this.allPts) {
            for(int d = 0; d < dimCount; d++) {
                // subtracting from boundingBox min coordinate to normalize it against dataset constraints
                // gridPts[d] = (long) Math.floor((p.coords[d] - boundingBox.minMaxPerDimension[d][0]) * invSideWidth);
                gridPts[d] = (long) Math.floor((p.coords[d]) * invSideWidth);
            }

            GridKey key = new GridKey(gridPts);
            // long key = (long) Arrays.hashCode(gridPts);
            if (gridMap.putIfAbsent(key, new GridCell(p, gridPts)) != null) {
                gridMap.get(key).points.add(p);
            }
        }

        log("localdbscan: Created " + gridMap.size() + " grid cell(s).");

        // 2. Assigning Core Cells and Core Points.
        int coreCellCount = 0; // for logging
        for (Map.Entry<GridKey, GridCell> mp : gridMap.entrySet()) {
            GridCell cell = mp.getValue();
            if (cell.Size() >= minPts) {
                cell.isCoreCell = true;
                coreCellCount++;
                for (Point pt : cell.points) {
                    if (pt.type != pt.GHOST) {
                        pt.type = pt.CORE;
                    }
                }
                cell.updateReptToCore();
            }
        }
        log("localdbscan: Identified " + coreCellCount + " core cell(s).");

        // 3. Merging core points within core cells.
        int coreMergeCnt = 0;
        for (GridCell cell: gridMap.values()) {
            if (cell.isCoreCell) {
                coreMergeCnt++;
                log("Core cell(" + coreMergeCnt + "): " + cell.reptId);
                for (Point pt: cell.points) {
                    rem(find(cell.reptId, pArray), find(pt.localId, pArray), pArray);
                }
                cell.reptId = find(cell.reptId, pArray);
            }
        }
        log("localdbscan: " + coreMergeCnt + " core cell(s) intra-merged.");

        // 4. Merging points with the surounding grids to form clusters.
        int bcpMerges = 0;
        int nonCoreExpansions = 0;
        for (Map.Entry<GridKey, GridCell> mp : gridMap.entrySet()) {
            GridCell cell = mp.getValue();

            if (cell.isCoreCell) {
                List<GridKey> neighbourKeys = neighbourGridQuery(cell);
                log("localdbscan: core cell under merge: " + cell.reptId);

                for (GridKey key: neighbourKeys) {
                    GridCell nCell = gridMap.get(key);
                    if (nCell != null && nCell.isCoreCell) {
                        log("localdbscan: neighbouring core cell under merge: " + nCell.reptId);
                        if (cell.Bcp(nCell, this.epsilon)) {
                            rem(find(cell.reptId, pArray), find(nCell.reptId, pArray), pArray);
                            bcpMerges++;
                        }
                    }
                }
            } else {
                if (cell.isGhost()) {
                    continue;
                }
                for (Point x: cell.points) {
                    if (x.type == x.GHOST) {
                        continue;
                    }

                    ArrayList<Point> neighbourPts = eNeighbourhoodPts(x, cell);
                    if (neighbourPts.size() >= minPts) {
                        x.type = x.CORE;
                        nonCoreExpansions++;

                        for (Point y: neighbourPts) {
                            if (y.type == y.CORE) {
                                rem(find(x.localId, pArray), find(y.localId, pArray), pArray);
                            } else if (y.type == y.NOISE) {
                                    y.type = y.BOUNDARY;
                                    pArray[y.localId] = find(x.localId, pArray);
                            }
                        }
                    } else if (x.type == x.NOISE) {
                        for (Point y: neighbourPts) {
                            if (y.type == y.CORE) {
                                x.type = x.BOUNDARY;
                                rem(find(x.localId, pArray), find(y.localId, pArray), pArray);
                            }
                        }
                    }
                }
            }
        }
        log("localdbscan: Core-Cell merges (BCP): " + bcpMerges + ", Non-core core discovery: " + nonCoreExpansions);

        //4.5 Remove ghostPoints which point to itself and did not take part in cluster formation at all
        //i.e trees having just one element.
        //This happens because we point every element to itself but since they
        //don't take part in cluster formation they are never updated it.
        int ghostClusters = 0;
        for (int i = 0; i < allPts.size(); i++) {
            Point pt = allPts.get(i);

            if (pt.type == pt.GHOST && pArray[i] == i) {
                pArray[i] = -1;
                ghostClusters++;
            }
        }
        log("localdbscan: Ghost clusters removed: " + ghostClusters);

        //5. Form local clusters from pArray
        // Set<Long> uniqueGhostRoots = new HashSet<>();
        // for (Point pt: this.allPts) {
        //     long rootId = (long) find(pt.localId, pArray);
        //
        //     if (rootId == -1) {
        //         continue;
        //     }
        //     Point rootPt = this.allPts.get((int) rootId);
        //     if (rootPt.type == rootPt.GHOST) {
        //         uniqueGhostRoots.add((long)rootId);
        //     }
        //
        //     //NOTE: using rootId as uid for cluster, should be replaced by globalUid after merging
        //     // checks if it points to itself i.e its not part of any cluster
        //     if (rootId != pt.localId) {
        //         Cluster cluster = localClusterMap.computeIfAbsent(rootId, id -> new Cluster(id));
        //
        //         if (pt.type == pt.CORE) {
        //             cluster.corePts.add(pt);
        //         } else if (pt.type == pt.BOUNDARY) {
        //             cluster.boundaryPts.add(pt);
        //         } else if (pt.type == pt.GHOST) {
        //             //TODO subject to change
        //             cluster.remoteProcessingNeighbours.add(pt);
        //         }
        //     }
        // }
        // log("localdbscan Finished. Found " + localClusterMap.size() + " local clusters. Ghost roots: " + uniqueGhostRoots.size());
        
        for (Point pt: this.allPts) {
            int rootId = find(pt.localId, pArray);
            if (rootId == -1) {
                continue;
            }
            if (rootId != pt.localId || pt.type == pt.CORE) {
                Cluster cluster = localClusterMap.computeIfAbsent((long)rootId, id -> new Cluster(id));
                if (pt.type == pt.CORE) {
                    // log("localdbscan: Final core points - " + pt.localId);
                    cluster.corePts.add(pt);
                } else if (pt.type == pt.BOUNDARY) {
                    // log("localdbscan: Final boundary points - " + pt.localId);
                    cluster.boundaryPts.add(pt);
                } else if (pt.type == pt.GHOST) {
                    //TODO subject to change
                    // log("localdbscan: Final ghost points - " + pt.localId);
                    cluster.remoteProcessingNeighbours.add(pt);
                }
            }
        }
        log("localdbscan Finished. Found " + localClusterMap.size() + " local clusters.");
    }

    //returns keys(hashes) for neighbouring grid cells
    public List<GridKey> neighbourGridQuery(GridCell cell) {
        int offset = (int) Math.ceil(Math.sqrt(this.dimCount));
        List<GridKey> keys = new ArrayList<>();

        generateRecursiveNeighbours(cell.pos, offset, 0, new long[this.dimCount], keys);
        return keys;
    }

    //helper
    private void generateRecursiveNeighbours(long[] currCellPos, int offset, int currDim, long[] candidatePos, List<GridKey> keys) {
        if (currDim == this.dimCount) {
            // skips the cell itself
            if (Arrays.equals(currCellPos, candidatePos))
                return;

            // long key = (long) Arrays.hashCode(candidatePos);
            GridKey key = new GridKey(candidatePos.clone());

            // checks if its a valid key
            if (this.gridMap.containsKey(key)) {
                keys.add(key);
            }
            return;
        }

        for (int i = -offset; i <= offset; i++) {
            candidatePos[currDim] = currCellPos[currDim] + i;
            generateRecursiveNeighbours(currCellPos, offset, currDim+1, candidatePos, keys);
        }
    }

    // returns: epsilon neighbourhood points by checking its neighbouring grid cells.
    private ArrayList<Point> eNeighbourhoodPts(Point x, GridCell currentCell) {
        ArrayList<Point> neighbours = new ArrayList<>();
        List<GridKey> neighbourKeys = neighbourGridQuery(currentCell);

        ArrayList<GridCell> potentialCells = new ArrayList<>();
        potentialCells.add(currentCell);
        for (GridKey key: neighbourKeys) {
            GridCell nCell = gridMap.get(key);
            if (nCell != null) {
                potentialCells.add(nCell);
            }
        }

        for (GridCell cell: potentialCells) {
            for (Point y: cell.points) {
                if (x.distanceToPoint(y) <= epsilon) {
                    neighbours.add(y);
                }
            }
        }
        return neighbours;
    }

    // returns: root of the union find tree
    private int find(int x, int[] p) {
        int rep = x;
        while (p[rep] != rep) {
            if (p[rep] == -1) {
                return -1;
            }
            rep = p[rep];
        }
        return rep;
    }

    // for merging two clusters using parray
    private int rem(int x, int y, int[] p) {
        int rx = x;
        int ry = y;
        while (p[rx] != p[ry]) {
            Point px = allPts.get(p[rx]);
            Point py = allPts.get(p[ry]);

            //Priority logic
            // 1. if types are different, non ghost point is prioritised.
            // 2. if types are same, the lower id is prioritised.

            boolean pyHasPriority;
            if (px.type == px.GHOST && py.type != py.GHOST) {
                pyHasPriority = true;
            } else if (px.type != px.GHOST && py.type == py.GHOST) {
                pyHasPriority = false;
            } else {
                pyHasPriority = (p[rx] > p[ry]);
            }

            if (pyHasPriority) {
                if (rx == p[rx]) {
                    p[rx] = p[ry];
                    return p[rx];
                }
                int z = p[rx];
                p[rx] = p[ry];
                rx = z;
            } else {
                if (ry == p[ry]) {
                    p[ry] = p[rx];
                    return p[ry];
                }
                int z = p[ry];
                p[ry] = p[rx];
                ry = z;
            }
        }
        return -1;
    }



    /**
     * Distributed Rem's Union-Find merge (Algorithm 5 from HY-DBSCAN paper).
     *
     * After localDBScan, each process has a set of local clusters. Clusters that
     * span process boundaries are connected via ghost points. This method
     * iteratively exchanges "Union query" messages until all cross-boundary
     * clusters have been merged.
     *
     * A Union query has the form: (senderRank, senderClusterUid, coords[])
     * meaning: "My cluster senderClusterUid touches the local point at coords[]
     * on your side — please union them."
     *
     * Message wire format (per query, packed into a flat double[]):
     *   [0]           senderRank       (cast to double)
     *   [1]           senderClusterUid (cast to double)
     *   [2 .. 2+d-1]  coords of the ghost point as seen by the sender
     *
     * We use a coordinator-style round: every process sends all pending queries,
     * then receives all incoming queries, processes them (possibly generating new
     * forwarded queries), and repeats until a global AllReduce confirms no process
     * has pending work.
     */
    public void mergeClusters() {
        log("Starting distributed cluster merge. Local clusters: " + localClusterMap.size());

        final int UNION_QUERY = 6;
        final int FIELDS = 2 + dimCount; // senderRank + senderClusterUid + coords
        int worldSize = COMM_WORLD.Size();

        // -----------------------------------------------------------------
        // Build a coord -> Point lookup for OUR OWN (non-ghost) points so we
        // can identify which local point a received ghost coord refers to.
        // We use a string key "x0,x1,...,xd" — exact double comparison is fine
        // because the ghost points are byte-for-byte copies of the originals.
        // -----------------------------------------------------------------
        Map<String, Point> coordToLocalPoint = new HashMap<>();
        for (Point pt : points) {   // 'points' contains only local (non-ghost) points
            coordToLocalPoint.put(coordKey(pt), pt);
        }

        // -----------------------------------------------------------------
        // Build the initial queue of Union queries to send.
        // For every cluster that has ghost neighbours, send one query per
        // ghost point to its source process.
        // -----------------------------------------------------------------
        // pending.get(destRank) -> list of flat FIELDS-element arrays to send
        Map<Integer, List<double[]>> pending = new HashMap<>();
        for (int r = 0; r < worldSize; r++) pending.put(r, new ArrayList<>());

        // localRootId -> cluster uid map (the uid IS the localRootId right now)
        // We need a mutable map from localRootId to "current representative uid"
        // so that after merges the root can change.
        // We represent this as a simple int[] parent array over cluster uids;
        // since uids are localIds (indices into allPts), we reuse a new pArray.
        // For simplicity we keep a Map<Long,Long> clusterParent that starts as
        // identity and gets updated when two clusters are unioned.
        Map<Long, Long> clusterParent = new HashMap<>();
        for (Long uid : localClusterMap.keySet()) {
            clusterParent.put(uid, uid);
        }

        // Generate initial outgoing queries
        for (Map.Entry<Long, Cluster> entry : localClusterMap.entrySet()) {
            long localRootUid = entry.getKey();
            Cluster cluster   = entry.getValue();
            for (Point ghost : cluster.remoteProcessingNeighbours) {
                int dest = ghost.sourceRank;
                double[] msg = buildQuery(rank, localRootUid, ghost);
                pending.get(dest).add(msg);
            }
        }

        // -----------------------------------------------------------------
        // Iterative round loop (terminates when AllReduce of total pending = 0)
        // -----------------------------------------------------------------
        int round = 0;
        while (true) {
            // Count total pending messages across all processes
            int localPending = pending.values().stream().mapToInt(List::size).sum();
            int[] globalPending = new int[]{localPending};
            COMM_WORLD.Allreduce(new int[]{localPending}, 0, globalPending, 0, 1, MPI.INT, MPI.SUM);
            if (globalPending[0] == 0) break;

            log("Merge round " + round + ": " + localPending + " outgoing queries (global: " + globalPending[0] + ")");

            // --- Send phase (non-blocking) ---
            // Post all sends first before receiving anything, so no process blocks
            // waiting for a receiver that is itself stuck trying to send.
            // We must keep the send buffers alive until Waitall completes.
            List<Request> sendRequests = new ArrayList<>();
            List<int[]>    headerBufs  = new ArrayList<>();
            List<double[]> dataBufs    = new ArrayList<>();

            for (int dest = 0; dest < worldSize; dest++) {
                if (dest == rank) continue;
                List<double[]> msgs = pending.get(dest);
                int count = msgs.size();

                int[] header = new int[]{count};
                headerBufs.add(header);
                sendRequests.add(COMM_WORLD.Isend(header, 0, 1, MPI.INT, dest, UNION_QUERY));

                if (count > 0) {
                    double[] flat = new double[count * FIELDS];
                    for (int i = 0; i < count; i++) {
                        System.arraycopy(msgs.get(i), 0, flat, i * FIELDS, FIELDS);
                    }
                    dataBufs.add(flat);
                    sendRequests.add(COMM_WORLD.Isend(flat, 0, flat.length, MPI.DOUBLE, dest, UNION_QUERY));
                }
            }

            // Clear pending — we'll repopulate from forwarded queries this round
            for (int r = 0; r < worldSize; r++) pending.get(r).clear();

            // --- Receive phase (blocking Recv is safe now: all sends already posted) ---
            List<double[]> newQueries = new ArrayList<>();
            for (int src = 0; src < worldSize; src++) {
                if (src == rank) continue;
                int[] countBuf = new int[1];
                COMM_WORLD.Recv(countBuf, 0, 1, MPI.INT, src, UNION_QUERY);
                int count = countBuf[0];
                if (count > 0) {
                    double[] flat = new double[count * FIELDS];
                    COMM_WORLD.Recv(flat, 0, flat.length, MPI.DOUBLE, src, UNION_QUERY);
                    for (int i = 0; i < count; i++) {
                        double[] q = new double[FIELDS];
                        System.arraycopy(flat, i * FIELDS, q, 0, FIELDS);
                        newQueries.add(q);
                    }
                }
            }

            // Wait for all non-blocking sends to complete before we reuse buffers
            Request.Waitall(sendRequests.toArray(new Request[0]));

            // --- Process received queries ---
            for (double[] query : newQueries) {
                int senderRank     = (int) query[0];
                long senderCluster = (long) query[1];
                double[] coords    = new double[dimCount];
                System.arraycopy(query, 2, coords, 0, dimCount);

                // The sender embedded coords of a ghost point they hold, which is
                // a byte-for-byte copy of one of OUR local points.
                String key = coordKey(coords);
                Point localPt = coordToLocalPoint.get(key);
                if (localPt == null) {
                    // We don't own this point — forward to whoever does.
                    for (Point gp : ghostPoints) {
                        if (coordKey(gp).equals(key)) {
                            pending.get(gp.sourceRank).add(query);
                            break;
                        }
                    }
                    continue;
                }

                Long localClusterUid = findLocalClusterUid(localPt, clusterParent);
                if (localClusterUid == null) {
                    continue; // noise point — nothing to merge
                }
                long localRoot = findRoot(localClusterUid, clusterParent);

                // Lower rank wins (paper's cross-process ID rule).
                if (senderRank < rank) {
                    // Sender wins. Send back the same coords — the sender owns this
                    // point locally and can find and merge their cluster on their side.
                    double[] reverseQuery = new double[FIELDS];
                    reverseQuery[0] = rank;
                    reverseQuery[1] = localRoot;
                    System.arraycopy(coords, 0, reverseQuery, 2, dimCount);
                    pending.get(senderRank).add(reverseQuery);
                } else {
                    // We win. Send back a ghost coord that the sender owns locally.
                    Cluster ourCluster = localClusterMap.get(localRoot);
                    if (ourCluster != null) {
                        for (Point gp : ourCluster.remoteProcessingNeighbours) {
                            if (gp.sourceRank == senderRank) {
                                pending.get(senderRank).add(buildQuery(rank, localRoot, gp));
                                break;
                            }
                        }
                    }
                }
            }

            round++;
        }

        // -----------------------------------------------------------------
        // Compact localClusterMap: remove entries that have been absorbed
        // into another cluster (their root in clusterParent no longer points
        // to themselves).
        // -----------------------------------------------------------------
        Map<Long, Cluster> mergedMap = new HashMap<>();
        for (Map.Entry<Long, Cluster> entry : localClusterMap.entrySet()) {
            long uid  = entry.getKey();
            long root = findRoot(uid, clusterParent);
            if (!mergedMap.containsKey(root)) {
                mergedMap.put(root, new Cluster(root));
            }
            mergedMap.get(root).merge(entry.getValue());
        }
        localClusterMap = mergedMap;

        log("Merge done. Clusters after merge: " + localClusterMap.size());
    }

    // -----------------------------------------------------------------------
    // Helpers for mergeClusters
    // -----------------------------------------------------------------------

    /** Builds a Union query message: [senderRank, clusterUid, coords...] */
    private double[] buildQuery(int senderRank, long clusterUid, Point anchorPoint) {
        double[] msg = new double[2 + dimCount];
        msg[0] = senderRank;
        msg[1] = clusterUid;
        System.arraycopy(anchorPoint.coords, 0, msg, 2, dimCount);
        return msg;
    }

    /** Coord-based map key for a point. */
    private String coordKey(Point pt) {
        return coordKey(pt.coords);
    }

    private String coordKey(double[] coords) {
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < coords.length; i++) {
            if (i > 0) sb.append(',');
            sb.append(Double.toHexString(coords[i])); // exact bit representation
        }
        return sb.toString();
    }

    /**
     * Walks clusterParent to find the current root uid for the cluster that
     * contains localPt, or null if the point isn't a root/member of any cluster.
     */
    private Long findLocalClusterUid(Point localPt, Map<Long, Long> clusterParent) {
        // localPt.localId is the key if the point is itself a cluster root;
        // otherwise we search all clusters for one containing this point.
        long id = localPt.localId;
        // Check if this point is a root
        if (clusterParent.containsKey(id)) {
            return findRoot(id, clusterParent);
        }
        // Otherwise search (linear — only needed for boundary/non-root local points)
        for (Map.Entry<Long, Cluster> e : localClusterMap.entrySet()) {
            Cluster c = e.getValue();
            for (Point p : c.corePts)     { if (p.localId == id) return findRoot(e.getKey(), clusterParent); }
            for (Point p : c.boundaryPts) { if (p.localId == id) return findRoot(e.getKey(), clusterParent); }
        }
        return null;
    }

    /** Path-compressed find on the clusterParent map. */
    private long findRoot(long uid, Map<Long, Long> clusterParent) {
        long root = uid;
        while (clusterParent.containsKey(root) && clusterParent.get(root) != root) {
            long grandparent = clusterParent.getOrDefault(clusterParent.get(root), clusterParent.get(root));
            clusterParent.put(root, grandparent); // path compression
            root = clusterParent.get(root);
        }
        return root;
    }

    /**
     * Assigns globally unique integer cluster IDs to every local (non-ghost) point.
     *
     * Algorithm (mirrors Algorithm 6 from the HY-DBSCAN paper):
     *  1. Each process labels its local clusters 0..n-1 locally.
     *  2. An AllGather collects each process's cluster count so every process can
     *     compute its own offset = sum of counts of lower-ranked processes.
     *  3. Each process adds the offset to its local labels → globally unique IDs.
     *  4. Every local (non-ghost) point in a cluster gets stamped with the global ID.
     *     Points not in any cluster (noise) keep globalClusterId = -1.
     *
     * After this call, point.globalClusterId is valid for all points in this.points.
     */
    public void assignClusterIds() {
        log("Assigning global cluster IDs. Local cluster count: " + localClusterMap.size());

        int worldSize = COMM_WORLD.Size();

        // Step 1: assign local sequential IDs to each cluster
        Map<Long, Integer> localIdMap = new HashMap<>(); // uid -> local sequential id
        int localCount = 0;
        for (long uid : localClusterMap.keySet()) {
            localIdMap.put(uid, localCount++);
        }

        // Step 2: AllGather cluster counts to compute per-process offsets
        int[] allCounts = new int[worldSize];
        COMM_WORLD.Allgather(new int[]{localCount}, 0, 1, MPI.INT,
                allCounts, 0, 1, MPI.INT);

        int offset = 0;
        for (int r = 0; r < rank; r++) {
            offset += allCounts[r];
        }

        // Step 3 & 4: stamp every local (non-ghost) point
        // Build a reverse lookup: localId of point -> globalClusterId
        // We walk localClusterMap and stamp each member point directly.
        for (Map.Entry<Long, Cluster> entry : localClusterMap.entrySet()) {
            long uid = entry.getKey();
            int globalId = offset + localIdMap.get(uid);
            Cluster c = entry.getValue();
            for (Point p : c.corePts)     { p.globalClusterId = globalId; }
            for (Point p : c.boundaryPts) { p.globalClusterId = globalId; }
            // ghost points are not stamped — they belong to another process
        }

        int totalClusters = 0;
        for (int c : allCounts) totalClusters += c;
        log("Cluster ID assignment done. Global cluster count: " + totalClusters);
    }

    public void log(String message) {
        System.out.println("Process " + COMM_WORLD.Rank() + ": " + message);
    }
}
