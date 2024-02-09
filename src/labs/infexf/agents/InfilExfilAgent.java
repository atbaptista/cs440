package src.labs.infexf.agents;

import java.util.Set;

// SYSTEM IMPORTS
import edu.bu.labs.infexf.agents.SpecOpsAgent;
import edu.bu.labs.infexf.distance.DistanceMetric;
import edu.bu.labs.infexf.graph.Vertex;
import edu.bu.labs.infexf.graph.Path;


import edu.cwru.sepia.environment.model.state.State.StateView;
import edu.cwru.sepia.environment.model.state.Unit.UnitView;

// JAVA PROJECT IMPORTS


public class InfilExfilAgent
    extends SpecOpsAgent
{
    private boolean debug = false;

    public InfilExfilAgent(int playerNum)
    {
        super(playerNum);
    }

    // if you want to get attack-radius of an enemy, you can do so through the enemy unit's UnitView
    // Every unit is constructed from an xml schema for that unit's type.
    // We can lookup the "range" of the unit using the following line of code (assuming we know the id):
    //     int attackRadius = state.getUnit(enemyUnitID).getTemplateView().getRange();
    @Override
    public float getEdgeWeight(Vertex src,
                               Vertex dst,
                               StateView state)
    {
        // System.out.print("(" + src.getXCoordinate() + ", " + src.getYCoordinate() + "), ");
        // System.out.println("(" + dst.getXCoordinate() + ", " + dst.getYCoordinate() + ")");
        // get distance to nearest enemy unit and return weight inversely proportional to distance
        int minDist = 1000;
        int tempDist = 0;

        // int eX = -1;
        // int eY = -1;
        Set<Integer> enemies = this.getOtherEnemyUnitIDs();
        if (enemies == null) return 1;
        for (int id : enemies)
        {
            // check if enemy unit died
            UnitView enemyView = state.getUnit(id);
            if (enemyView == null) continue;

            int x = dst.getXCoordinate();
            int y = dst.getYCoordinate();
            
            int enemyX = enemyView.getXPosition();
            int enemyY = enemyView.getYPosition();

            // for (int i = 5; i >= 3; i--)
            // {
            //     if (!isWithinDist(x, y, enemyX, enemyY, i)) continue;

            // }
            tempDist = getDistance(x,y,enemyX,enemyY);
            if (tempDist < minDist)
            {
                minDist = tempDist;
                // eX = enemyX;
                // eY = enemyY;
            }
        }

        switch (minDist)
        {
            case 8:
                return 2.5f;
            case 7:
                return 10;
            case 6:
                return 50;
            case 5:
                return 100;
            case 4:
                return 200;
            case 3:
                return 400;
            case 2:
                return 800;
            default:
                return 1;
        }

        // if (debug)
        // {
        //     System.out.print(minDist);
        //     System.out.print("| coord: (" + Integer.toString(dst.getXCoordinate()) + "," + Integer.toString(dst.getYCoordinate())+ ") | ");
        //     System.out.print("enemy coord: (" + Integer.toString(eX) + "," + Integer.toString(eY)+ ")\n");
        // }
        // if (minDist >= 6.25)
        // {
        //     return 1f;
        // }
        // else if (minDist >= 5.75)
        // {
        //     if (debug) System.out.println("1\n");
        //     return 10f;
        // }
        // else if (minDist >= 5.25)
        // {
        //     if (debug) System.out.println("5\n");
        //     return 20f;
        // }
        // else if (minDist >= 4.75)
        // {
        //     if (debug) System.out.println("10\n");
        //     return 40;
        // }
        // else if (minDist >= 4.26)
        // {
        //     if (debug) System.out.println("25\n");
        //     return 100;
        // }
        // else if (minDist >= 3.5)
        // {
        //     if (debug) System.out.println("50\n");
        //     return 200;
        // }
        // else if (minDist >= 2.9)
        // {
        //     if (debug) System.out.println("100\n");
        //     return 400;
        // }
        // else if (minDist >= 2)
        // {
        //     return 800;
        // }
        // else
        // {
        //     return 800;
        // }
    }

    // return manhattan distance? 
    private int getDistance(int x1, int y1, int x2, int y2)
    {
        // expensive?
        // return (float)(Math.sqrt(Math.pow((x1 - x2), 2) + Math.pow((y1 - y2), 2)));
        return Math.abs(x1-x2) + Math.abs(y1 - y2);
    }

    private boolean isWithinDist(int x1, int y1, int x2, int y2, int dist)
    {
        return Math.abs(x1-x2) <= dist && Math.abs(y1-y2) <= dist;
    }

    @Override
    public boolean shouldReplacePlan(StateView state)
    {
        int counter = 0;
        for (Vertex v : this.getCurrentPlan())
        {
            // check if any enemies are x units away from this point
            Set<Integer> enemies = this.getOtherEnemyUnitIDs();
            if (enemies == null) continue;
            for (int id : enemies)
            {
                UnitView enemyView = state.getUnit(id);
                if (enemyView == null) continue;

                int x = v.getXCoordinate();
                int y = v.getYCoordinate();

                int enemyX = enemyView.getXPosition();
                int enemyY = enemyView.getYPosition();

                int attackRadius = enemyView.getTemplateView().getRange();
                attackRadius += 3;

                // within square distance
                if (Math.abs(x-enemyX) <= attackRadius && Math.abs(y-enemyY) <= attackRadius)
                {
                    if (debug) System.out.println("ENEMY NEAR\n");
                    return true;
                }
            }
            counter++;
            // stop at 6 steps ahead
            if (counter >= 6)
            {
                return false;
            }
        }
        return false;
    }

}
