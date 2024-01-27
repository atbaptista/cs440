package src.labs.stealth.agents;

// SYSTEM IMPORTS
import edu.bu.labs.stealth.agents.MazeAgent;
import edu.bu.labs.stealth.graph.Vertex;
import edu.bu.labs.stealth.graph.Path;


import edu.cwru.sepia.environment.model.state.State.StateView;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;   // will need for dfs
import java.util.LinkedList;
import java.util.List;
import java.util.Stack;     // will need for dfs
import java.util.Set;       // will need for dfs


// JAVA PROJECT IMPORTS


public class DFSMazeAgent
    extends MazeAgent
{

    public DFSMazeAgent(int playerNum)
    {
        super(playerNum);
    }

    @Override
    public Path search(Vertex src,
                       Vertex goal,
                       StateView state)
    {
        Set<Vertex> visited = new HashSet<Vertex>();
        Stack<Vertex> stack = new Stack<Vertex>();
        HashMap<Vertex, Vertex> previous = new HashMap<Vertex, Vertex>();

        stack.push(src);

        while (!stack.isEmpty()) {
            Vertex current = stack.pop();
            
            if (current.equals(goal))
            {
                break;
            }

            for (Vertex neighbor : getAdjacent(current, goal, state)) {
                if (visited.contains(neighbor))
                    continue;

                previous.put(neighbor, current);
                visited.add(neighbor);
                stack.push(neighbor);
            }
        }

        Vertex prev = previous.get(goal);
        Path p = new Path(goal);
        while (prev != null)
        {
            p = new Path(prev, 1.0f, p);
            prev = previous.get(prev);
        }
        return p.getParentPath();
    }

    private boolean isValidVertex(Vertex v, Vertex goal, StateView state) 
    {
        int x = v.getXCoordinate();
        int y = v.getYCoordinate();

        if (!state.inBounds(x, y)) 
            return false;
        if (state.isResourceAt(x, y)) 
            return false;
        if (state.isUnitAt(x, y) && !(x == goal.getXCoordinate() && y == goal.getYCoordinate()))
            return false;

        return true;
    }

    private Vertex[] getAdjacent(Vertex v, Vertex goal, StateView state) 
    {
        List<Vertex> adjacent = new ArrayList<>();
        int[][] deltas = {{1, 0}, {-1, 0}, {0, 1}, {0, -1}, {1, 1}, {1, -1}, {-1, -1}, {-1, 1}};
        for (int[] delta : deltas) 
        {
            Vertex neighbor = new Vertex(v.getXCoordinate() + delta[0], v.getYCoordinate() + delta[1]);
            if (isValidVertex(neighbor, goal, state)) 
                adjacent.add(neighbor);
        }
        return adjacent.toArray(new Vertex[0]);
    }

    @Override
    public boolean shouldReplacePlan(StateView state)
    {
        if (this.getNextVertexToMoveTo() == null)
            return false;
        int xCoord = this.getNextVertexToMoveTo().getXCoordinate();
        int yCoord = this.getNextVertexToMoveTo().getYCoordinate();
        // if (state.isUnitAt(xCoord, yCoord))
        // {
        //     Integer unit = state.unitAt(xCoord, yCoord);
        //     if (unit == null)
        //         return false;
        //     if(state.getUnit(unit).getTemplateView().getName().toLowerCase().equals("footman"))
        //         System.out.println(state.getUnit(state.unitAt(xCoord, yCoord)).getTemplateView().getName().toLowerCase());
        //         return true;
        // }
        if (!state.inBounds(xCoord, yCoord))
            return true;
        if (state.isResourceAt(xCoord, yCoord))
            return true;

        return false;
    }

}
