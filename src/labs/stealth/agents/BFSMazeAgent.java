package src.labs.stealth.agents;

// SYSTEM IMPORTS
import edu.bu.labs.stealth.agents.MazeAgent;
import edu.bu.labs.stealth.graph.Vertex;
import edu.bu.labs.stealth.graph.Path;


import edu.cwru.sepia.environment.model.state.State.StateView;


import java.util.HashSet;       // will need for bfs
import java.util.Queue;         // will need for bfs
import java.util.LinkedList;    // will need for bfs
import java.util.Set;           // will need for bfs


// JAVA PROJECT IMPORTS


public class BFSMazeAgent
    extends MazeAgent
{

    public BFSMazeAgent(int playerNum)
    {
        super(playerNum);
    }

    @Override
    public Path search(Vertex src,
                       Vertex goal,
                       StateView state)
    {
        Set<Vertex> visited = new HashSet<Vertex>();
        Queue<Vertex> queue = new LinkedList<Vertex>();
        visited.add(src);
        // while (!queue.isEmpty()) {
        //     Vertex current = queue.poll();
            
        //     for (Vertex neighbor : current.getNeighbors()) {
        //         if (!neighbor.isVisited()) {
        //             neighbor.setVisited(true);
        //             queue.add(neighbor);
        //         }
        //     }
        // }
        return recursiveSearch(null , src, goal, visited, queue, state);
    }

    // * Here is an example to build a path from coordinate (0,1) to adjacent coordinate (1,1):
    // * Path p = new Path(new Vertex(0, 1));     // tail of the linked list
    // * p = new Path(new Vertex(1, 1), 1.0, p);  // add edge (0, 1) -> (1, 1) with cost 1.0 to the head of the path.
    // how to undo
    // path = path.getParentPath();
    private Path recursiveSearch(Path path, Vertex src, Vertex goal, Set<Vertex> visited, Queue<Vertex> queue, StateView state)
    {
        if (src.equals(goal))
            return path.getParentPath();

        if (src.equals(null))
        {
            // no path is found, queue ran out
            System.err.println("no path found, src is null");
            return null;
        }

        // add this vertex to path
        // Path pNew = new Path(new Vertex(src.getXCoordinate(), src.getYCoordinate()), 1.0f, path);

        // add this vertex to visited
        // visited.add(src);
        
        // add non visited adjacent tiles to queue 
        // should check if theyre in the map bounds prob
        for (Vertex v : getAdjacent(src))
        {
            if (visited.contains(v))
                continue;
            // if (state.isUnitAt(v.getXCoordinate(), v.getYCoordinate()))
            //     continue;
            if (!state.inBounds(v.getXCoordinate(), v.getYCoordinate()))
                continue;
            if (state.isResourceAt(v.getXCoordinate(), v.getYCoordinate()))
                continue;

            System.out.println(v.toString());
            queue.add(v);
            visited.add(v);
        }
        System.out.println(queue.toString());
        Vertex next = queue.poll();
        // if (next.equals(null))
        // {
        //     // no path is found, queue ran out
        //     System.err.println("no path found, src is null");
        //     return null;
        // }
        return recursiveSearch(pNew, next, goal, visited, queue, state);
    }

    private Vertex[] getAdjacent(Vertex v)
    {
        Vertex[] adjacent = {
            new Vertex(v.getXCoordinate()+1, v.getYCoordinate()), 
            new Vertex(v.getXCoordinate()-1, v.getYCoordinate()),
            new Vertex(v.getXCoordinate(), v.getYCoordinate()+1),
            new Vertex(v.getXCoordinate(), v.getYCoordinate()-1)
        };
        return adjacent;
    }

    @Override
    public boolean shouldReplacePlan(StateView state)
    {
        // int xCoord = this.getNextVertexToMoveTo().getXCoordinate();
        // int yCoord = this.getNextVertexToMoveTo().getYCoordinate();
        // if (state.isUnitAt(xCoord, yCoord) || state.isResourceAt(xCoord, yCoord))
        // {
        //     return true;
        // }

        return false;
    }

}
