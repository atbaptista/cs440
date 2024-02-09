package src.labs.infexf.agents;

// SYSTEM IMPORTS
import edu.bu.labs.infexf.agents.SpecOpsAgent;
import edu.bu.labs.infexf.distance.DistanceMetric;
import edu.bu.labs.infexf.graph.Vertex;
import edu.bu.labs.infexf.graph.Path;


import edu.cwru.sepia.environment.model.state.State.StateView;

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
        float minDist = 1000;
        float tempDist = 0;

        int eX = -1;
        int eY = -1;
        if (this.getOtherEnemyUnitIDs() == null) return 1;
        for (int id : this.getOtherEnemyUnitIDs())
        {
            int x = dst.getXCoordinate();
            int y = dst.getYCoordinate();
            // check if enemy unit died
            if (state.getUnit(id) == null) continue;

            int enemyX = state.getUnit(id).getXPosition();
            int enemyY = state.getUnit(id).getYPosition();
            tempDist = getDistance(x,y,enemyX,enemyY);
            if (tempDist < minDist)
            {
                minDist = tempDist;
                eX = enemyX;
                eY = enemyY;
            }
        }
        if (debug)
        {
            System.out.print(minDist);
            System.out.print("| coord: (" + Integer.toString(dst.getXCoordinate()) + "," + Integer.toString(dst.getYCoordinate())+ ") | ");
            System.out.print("enemy coord: (" + Integer.toString(eX) + "," + Integer.toString(eY)+ ")\n");
        }
        if (minDist >= 6.25)
        {
            return 1f;
        }
        else if (minDist >= 5.75)
        {
            if (debug) System.out.println("1\n");
            return 10f;
        }
        else if (minDist >= 5.25)
        {
            if (debug) System.out.println("5\n");
            return 20f;
        }
        else if (minDist >= 4.75)
        {
            if (debug) System.out.println("10\n");
            return 40;
        }
        else if (minDist >= 4.26)
        {
            if (debug) System.out.println("25\n");
            return 100;
        }
        else if (minDist >= 3.5)
        {
            if (debug) System.out.println("50\n");
            return 200;
        }
        else if (minDist >= 2.9)
        {
            if (debug) System.out.println("100\n");
            return 400;
        }
        else if (minDist >= 2)
        {
            return 800f;
        }
        else
        {
            return 800;
        }
    }

    // return manhattan distance? 
    private float getDistance(int x1, int y1, int x2, int y2)
    {
        // expensive?
        return (float)(Math.sqrt(Math.pow((x1 - x2), 2) + Math.pow((y1 - y2), 2)));
        // return Math.abs(x1-x2) + Math.abs(y1 - y2);
    }

    @Override
    public boolean shouldReplacePlan(StateView state)
    {
        int counter = 0;
        for (Vertex v : this.getCurrentPlan())
        {
            // check if any enemies are x units away from this point
            if (this.getOtherEnemyUnitIDs() == null) continue;
            for (int id : this.getOtherEnemyUnitIDs())
            {
                int x = v.getXCoordinate();
                int y = v.getYCoordinate();
                if (state.getUnit(id) == null) continue;
                int enemyX = state.getUnit(id).getXPosition();
                int enemyY = state.getUnit(id).getYPosition();

                int attackRadius = state.getUnit(id).getTemplateView().getRange();
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
