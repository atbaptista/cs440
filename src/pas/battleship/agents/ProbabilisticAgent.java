package src.pas.battleship.agents;


import edu.bu.battleship.game.ships.Ship;
import edu.bu.battleship.game.ships.Ship.ShipType;

// SYSTEM IMPORTS
import java.util.*; 
// JAVA PROJECT IMPORTS
import edu.bu.battleship.agents.Agent;
import edu.bu.battleship.game.Game.GameView;
import edu.bu.battleship.game.Game;
import edu.bu.battleship.game.EnemyBoard.Outcome;
import edu.bu.battleship.utils.Coordinate;


public class ProbabilisticAgent
    extends Agent
{


    public ProbabilisticAgent(String name)
    {
        super(name);
        System.out.println("[INFO] ProbabilisticAgent.ProbabilisticAgent: constructed agent");
    }

    @Override
    public Coordinate makeMove(final GameView game)
    {
        // Coordinate result = new Coordinate(0, 0);
        // Outcome[][] board = game.getEnemyBoardView();
        // for (int i = 0; i < board.length; i++)
        // {
        //     for (int j = 0; j < board[i].length; j++)
        //     {
        //         if (board[i][j] == Outcome.UNKNOWN)
        //         {
        //             result = new Coordinate(i, j);
        //         }
        //     }
        // }
        return calculateProbs(game);
        // int xMax = game.getGameConstants().getNumCols();
        // int yMax = game.getGameConstants().getNumRows();
        // int x = randInt(0, xMax-1);
        // int y = randInt(0, yMax-1);
        // result = new Coordinate(x, y);
    }   
    class PositionResult {
        float totalPositions;
        float[][] positions;
        
        public PositionResult(float totalPositions, float[][] positions) {
            this.totalPositions = totalPositions;
            this.positions = positions;
        }
    }
    private Coordinate calculateProbs(final GameView game)
    {
        List<PositionResult> probabilities = new ArrayList<>();
        Map<ShipType, Integer> shipsLeft = game.getEnemyShipTypeToNumRemaining();
        for (Map.Entry<ShipType, Integer> entry : shipsLeft.entrySet()) {
            ShipType ship = entry.getKey();
            Integer numLeft = entry.getValue();
            
            if (numLeft != 1)
            {
                continue;
            }
            PositionResult result;
            switch(ship)
            {
                case AIRCRAFT_CARRIER:
                    result = enumerate(game, 5);
                    break;
                case BATTLESHIP:
                    result = enumerate(game, 4);
                    break;
                case DESTROYER:
                    result = enumerate(game, 3);
                    break;
                case PATROL_BOAT:
                    result = enumerate(game, 3);
                    break;
                case SUBMARINE:
                    result = enumerate(game, 2);
                    break;
                default:
                    throw new UnsupportedOperationException("Unknown ship type: " + ship);
            }
            probabilities.add(result);
        }
        Outcome[][] board = game.getEnemyBoardView();
        float[][] finalProbs = new float[board.length][board[0].length];
        for (int i = 0; i < board.length; i++) 
        {
            for (int j = 0; j < board[i].length; j++) 
            {
                float prob = 0;
                if (board[i][j] == Outcome.UNKNOWN)
                {
                    // add up probabilities for each ship type being in that spot
                    for (PositionResult data : probabilities)
                    {
                        // System.out.println(data.positions[i][j]);
                        // System.out.println(data.totalPositions);
                        prob += (data.positions[i][j] / data.totalPositions);
                    }
                    
                    finalProbs[i][j] = prob;
                }
                else 
                {
                    finalProbs[i][j] = 0;
                }
            }
        }
        for (int i = 0; i < board.length; i++) {
            for (int j = 0; j < board[i].length; j++) {
                if (board[i][j] == Outcome.HIT) {
                    int validSpots = 0;
                    // right
                    if (j + 1 < board[i].length && board[i][j + 1] == Outcome.UNKNOWN) {
                        validSpots++;
                    }
                    // left
                    if (j - 1 >= 0 && board[i][j - 1] == Outcome.UNKNOWN) {
                        validSpots++;
                    }
                    // down
                    if (i + 1 < board.length && board[i + 1][j] == Outcome.UNKNOWN) {
                        validSpots++;
                    }
                    // up
                    if (i - 1 >= 0 && board[i - 1][j] == Outcome.UNKNOWN) {
                        validSpots++;
                    }
                    ////////////////////////////////////////////////////////////////////////////////
                    if (j + 1 < board[i].length && board[i][j+1] == Outcome.UNKNOWN) {
                        finalProbs[i][j+1] += 1/validSpots + 1;
                    }
                    // left
                    if (j - 1 >= 0 && board[i][j-1] == Outcome.UNKNOWN) {
                        finalProbs[i][j-1] += 1/validSpots+1;
                    }
                    // down
                    if (i + 1 < board.length && board[i+1][j] == Outcome.UNKNOWN) {
                        finalProbs[i+1][j] += 1/validSpots+1;
                    }
                    // up
                    if (i - 1 >= 0 && board[i - 1][j] == Outcome.UNKNOWN) {
                        finalProbs[i-1][j] += 1/validSpots+1;
                    }
                    
                }
            }
        }
        float highestProbability = 0;
        int x = 0;
        int y = 0;
        for (int i = 0; i < board.length; i++) 
        {
            for (int j = 0; j < board[i].length; j++) 
            {
                System.out.print(finalProbs[i][j]);
                System.out.print(",");
                if (finalProbs[i][j] > highestProbability)
                {
                    highestProbability = finalProbs[i][j];
                    x = i;
                    y = j;
                }
            }
            System.out.println();
        }
        System.out.println(highestProbability);
        System.out.println(x);
        System.out.println(y);
        System.out.println();
        return new Coordinate(x, y);
    }

    private PositionResult enumerate(final GameView game, int size) 
    {
        Outcome[][] board = game.getEnemyBoardView();
        float[][] positions = new float[board.length][board[0].length];
        float validSpots = 0;
        
        // enumerate 
        for (int i = 0; i < board.length; i++) 
        {
            for (int j = 0; j < board[i].length; j++) 
            {
                int isValidCountRight = 0;
                int isValidCountDown = 0;
                
                // check k right and down
                for(int k = 0; k < size; k++) 
                {
                    // check out of bounds
                    if (j + k < board[i].length) 
                    {
                        // check right
                        if (board[i][j+k] != Outcome.MISS && board[i][j+k] != Outcome.SUNK) 
                        {
                            isValidCountRight++;
                            positions[i][j+k]++;
                        }
                    }
                    // check out of bounds
                    if (i + k < board.length) 
                    {
                        // check down
                        if (board[i+k][j] != Outcome.MISS && board[i+k][j] != Outcome.SUNK) 
                        {
                            isValidCountDown++;
                            positions[i+k][j]++;
                        }
                    }
                }
                
                // line can be placed right
                if (isValidCountRight == size) 
                {
                    validSpots++;
                }
                // line can be placed down
                if (isValidCountDown == size) 
                {
                    validSpots++;
                }
            }
        }
        
        return new PositionResult(validSpots, positions);
    }

    private int randInt(int min, int max) 
    {
        Random rand = new Random();
        int randomNum = rand.nextInt((max - min) + 1) + min;
        return randomNum;
    } 
    @Override
    public void afterGameEnds(final GameView game) {}

}

