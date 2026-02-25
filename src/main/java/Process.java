import mpi.Group;
import mpi.Intracomm;
import mpi.MPI;

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
    protected double epsilon;
    protected int dimCount;
    protected ArrayList<Point> points;
    protected ArrayList<Point> ghostPoints = new ArrayList<>();
    protected ArrayList<Point> allPts;

    protected NavigableMap<Long, GridCell> gridMap;
    protected Map<Long, Cluster> localClusterMap = new HashMap<>();
    
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
                log("Received all bounding boxes");
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
        log("kd-Tree done. Size: " + points.size() + ".Resetting communication parameters");
        group = COMM_WORLD.Group();
        communicator = COMM_WORLD;
        rank = group.Rank();
    }

    //FIX: have clusters where ghost point is the root
    public void localDBScan(int minPts) {
        log("Starting localDBScan. Local points: " + points.size() + ", Ghost Points: " + ghostPoints.size());

        double invSideWidth = 1.0 / (epsilon / Math.sqrt(dimCount));

        this.allPts = new ArrayList<>(points);
        this.allPts.addAll(ghostPoints);
        this.gridMap = new TreeMap<>();
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
                gridPts[d] = (long) Math.floor((p.coords[d] - boundingBox.minMaxPerDimension[d][0]) * invSideWidth);
            }
            
            long key = (long) Arrays.hashCode(gridPts);
            if (gridMap.putIfAbsent(key, new GridCell(p, gridPts)) != null) {
                gridMap.get(key).points.add(p);
            }
        }

        log("localdbscan: Created " + gridMap.size() + " grid cell(s).");

        // 2. Assigning Core Cells and Core Points.
        int coreCellCount = 0; // for logging
        for (Map.Entry<Long, GridCell> mp : gridMap.entrySet()) {
            GridCell cell = mp.getValue();
            if (cell.Size() > minPts) {
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
        for (Map.Entry<Long, GridCell> mp : gridMap.entrySet()) {
            GridCell cell = mp.getValue();

            if (cell.isCoreCell) {
                List<Long> neighbourKeys = neighbourGridQuery(cell);

                for (Long key: neighbourKeys) {
                    GridCell nCell = gridMap.get(key);
                    if (nCell != null && nCell.isCoreCell) {
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
                    if (neighbourPts.size() > minPts) {
                        x.type = x.CORE;
                        nonCoreExpansions++;
                        
                        for (Point y: neighbourPts) {
                            if (y.type == y.CORE) {
                                rem(x.localId, y.localId, pArray);
                            } else if (y.type == y.NOISE || y.type == y.GHOST) {
                                if (y.type == y.NOISE) {
                                    y.type = y.BOUNDARY;
                                }
                                rem(x.localId, y.localId, pArray);
                            }
                        }
                    } else if (x.type == x.NOISE) {
                        for (Point y: neighbourPts) {
                            if (y.type == y.CORE) {
                                x.type = x.BOUNDARY;
                                rem(x.localId, y.localId, pArray);
                            }
                        }
                    }
                }
            }
        }
        log("localdbscan: Core-Cell merges (BCP): " + bcpMerges + ", Non-core core discovery: " + nonCoreExpansions);

        //5. Form local clusters from pArray
        Set<Long> uniqueGhostRoots = new HashSet<>();
        for (Point pt: this.allPts) {
            long rootId = (long) find(pt.localId, pArray);

            Point rootPt = this.allPts.get((int) rootId);
            if (rootPt.type == rootPt.GHOST) {
                uniqueGhostRoots.add((long)rootId);
            }

            //NOTE: using rootId as uid for cluster, should be replaced by globalUid after merging
            // checks if it points to itself i.e its not part of any cluster
            if (rootId != pt.localId) { 
                Cluster cluster = localClusterMap.computeIfAbsent(rootId, id -> new Cluster(id));

                if (pt.type == pt.CORE) {
                    cluster.corePts.add(pt);
                } else if (pt.type == pt.BOUNDARY) {
                    cluster.boundaryPts.add(pt);
                } else if (pt.type == pt.GHOST) {
                    //TODO subject to change 
                    cluster.remoteProcessingNeighbours.add(pt);
                }
            }
        }
        log("DBSCAN Finished. Found " + localClusterMap.size() + " local clusters. Ghost roots: " + uniqueGhostRoots.size());
    }

    //returns keys(hashes) for neighbouring grid cells
    public List<Long> neighbourGridQuery(GridCell cell) {
        int offset = (int) Math.ceil(Math.sqrt(this.dimCount));
        List<Long> keys = new ArrayList<>();

        generateRecursiveNeighbours(cell.pos, offset, 0, new long[this.dimCount], keys);
        return keys;
    }

    //helper
    private void generateRecursiveNeighbours(long[] currCellPos, int offset, int currDim, long[] candidatePos, List<Long> keys) {
        if (currDim == this.dimCount) {
            // skips the cell itself
            if (Arrays.equals(currCellPos, candidatePos))
                return;
            
            long key = (long) Arrays.hashCode(candidatePos);

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
        List<Long> neighbourKeys = neighbourGridQuery(currentCell);

        ArrayList<GridCell> potentialCells = new ArrayList<>();
        potentialCells.add(currentCell);
        for (Long key: neighbourKeys) {
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

    public void log(String message) {
        System.out.println("Process " + COMM_WORLD.Rank() + ": " + message);
    }
}
