package src.pas.tetris.agents;

// 0.2 (218.7045/300) + 0.4 (145/200) + 0.4 (110/150)
// java -cp lib/*:. edu.bu.tetris.Main -q src.pas.tetris.agents.TetrisQAgent -p 5000 -t 100 -v 50 -n 0.01 -b 5000 -s | tee run.log
// java -cp "./lib/*;." edu.bu.tetris.Main -q src.pas.tetris.agents.TetrisQAgent -p 5000 -t 100 -v 50 -n 0.01 -b 10000 -s -o params/newReward | tee run.log
// java -cp "./lib/*;." edu.bu.tetris.Main -q src.pas.tetris.agents.TetrisQAgent -p 5000 -t 100 -v 50 -n 0.01 -b 5000 -s -o param/10feat/ | tee run10.log
// LONG
// java -cp "./lib/*;." edu.bu.tetris.Main -q src.pas.tetris.agents.TetrisQAgent -p 5000 -t 250 -v 50 -n 0.0001 -g 0.99 -u 25 -b 50000 -c 1000000000 -s | tee runDim.log
// javac -cp "./lib/*;." @tetris.srcs
// java -cp "./lib/*;." edu.bu.tetris.Main -a src.pas.tetris.agents.TetrisQAgent -i ./params/trash/36.model

// run10.log - params/qFunction700.model
// 14 features, reward is just points, slow 

// run4s.log - param/800.model
// 4 features, reward is that function, slow

// run.log - /param/10feat/4868.model
// 4 features, reward is that function, quick

// SYSTEM IMPORTS
import java.util.Iterator;
import java.util.List;
import java.util.Random;
import java.util.concurrent.TimeUnit;

// JAVA PROJECT IMPORTS
import edu.bu.tetris.agents.QAgent;
import edu.bu.tetris.agents.TrainerAgent.GameCounter;
import edu.bu.tetris.game.Board;
import edu.bu.tetris.game.Game.GameView;
import edu.bu.tetris.game.minos.Mino;
import edu.bu.tetris.linalg.Matrix;
import edu.bu.tetris.nn.Model;
import edu.bu.tetris.nn.LossFunction;
import edu.bu.tetris.nn.Optimizer;
import edu.bu.tetris.nn.models.Sequential;
import edu.bu.tetris.nn.layers.Dense; // fully connected layer
import edu.bu.tetris.nn.layers.ReLU;  // some activations (below too)
import edu.bu.tetris.nn.layers.Tanh;
import edu.bu.tetris.nn.layers.Sigmoid;
import edu.bu.tetris.training.data.Dataset;
import edu.bu.tetris.utils.Pair;


public class TetrisQAgent
    extends QAgent
{

    long prevPhaseId = 0;
    long prevGameId = 0;
    int phase = 0;
    int rounds = 0;
    public static final double EXPLORATION_PROB = 0.05;
    boolean four = true;

    private Random random;

    public TetrisQAgent(String name)
    {
        super(name);
        this.random = new Random(12345); // optional to have a seed
    }

    public Random getRandom() { return this.random; }

    @Override
    public Model initQFunction()
    {
        rounds = 0;
        // build a single-hidden-layer feedforward network
        // this example will create a 3-layer neural network (1 hidden layer)
        // in this example, the input to the neural network is the
        // image of the board unrolled into a giant vector
        int inputFeatures;
        if (four) {
            inputFeatures = 4;
        }
        else {
            inputFeatures = 14;
        }
        final int hiddenDim = inputFeatures * 2;
        final int outDim = 1;

        Sequential qFunction = new Sequential();
        qFunction.add(new Dense(inputFeatures, hiddenDim));
        qFunction.add(new Tanh());
        qFunction.add(new Dense(hiddenDim, outDim));

        return qFunction;
    }

    /**
        This function is for you to figure out what your features
        are. This should end up being a single row-vector, and the
        dimensions should be what your qfunction is expecting.
        One thing we can do is get the grayscale image
        where squares in the image are 0.0 if unoccupied, 0.5 if
        there is a "background" square (i.e. that square is occupied
        but it is not the current piece being placed), and 1.0 for
        any squares that the current piece is being considered for.
        
        We can then flatten this image to get a row-vector, but we
        can do more than this! Try to be creative: how can you measure the
        "state" of the game without relying on the pixels? If you were given
        a tetris game midway through play, what properties would you look for?
     */
    @Override
    public Matrix getQFunctionInput(final GameView game,
                                    final Mino potentialAction)
    {
        int inputFeatures;
        if (four) {
            inputFeatures = 4;
        }
        else {
            inputFeatures = 14;
        }
        Matrix features = Matrix.zeros(1, inputFeatures);
        if (!four){
            for (int col = 0; col < Board.NUM_COLS; col++) {
                for (int row = 0; row < Board.NUM_ROWS; row++) {
                    if (game.getBoard().isCoordinateOccupied(col, row)) {
                        features.set(0, col, Board.NUM_ROWS-row);
                        break;
                    }
                }
            }
        }
        
        Matrix greyScale = null;
        try
        {
            greyScale = game.getGrayscaleImage(potentialAction);
        } catch(Exception e)
        {
            e.printStackTrace();
            System.exit(-1);
        }

        if (four){
            features.set(0, 0, calculateLinesCleared(greyScale));
            features.set(0, 1, calculateHoles(greyScale));
            features.set(0, 2, calculateBumpiness(greyScale));
            features.set(0, 3, calculateTotalHeight(greyScale));
        }
        else {
            features.set(0, 10, calculateLinesCleared(greyScale));
            features.set(0, 11, calculateHoles(greyScale));
            features.set(0, 12, calculateBumpiness(greyScale));
            features.set(0, 13, calculateTotalHeight(greyScale));
        }
        
        // System.out.println(features.toString() + "\n\n\n");
        return features;
    }

    // Calculate the number of lines cleared
    public static int calculateLinesCleared(Matrix board) {
        int linesCleared = 0;
        for (int row = 0; row < board.getShape().getNumRows(); row++) {
            boolean isFull = true;
            for (int col = 0; col < board.getShape().getNumCols(); col++) {
                if (board.get(row, col) == 0.0) {
                    isFull = false;
                    break;
                }
            }
            if (isFull) linesCleared++;
        }
        return linesCleared;
    }

    // Calculates the number of holes in the board
    public static int calculateHoles(Matrix board) {
        int holes = 0;
        for (int col = 0; col < board.getShape().getNumCols(); col++) {
            boolean blockFound = false;
            for (int row = 0; row < board.getShape().getNumRows(); row++) {
                double cell = board.get(row, col);
                if (cell == 0.5 || cell == 1.0) {
                    blockFound = true;
                } else if (blockFound && cell == 0.0) {
                    holes++;
                }
            }
        }
        return holes;
    }
    // Calculates the bumpiness of the board
    public static int calculateBumpiness(Matrix board) {
        int bumpiness = 0;
        int previousHeight = getColumnHeight(board, 0);
        for (int col = 1; col < board.getShape().getNumCols(); col++) {
            int currentHeight = getColumnHeight(board, col);
            bumpiness += Math.abs(currentHeight - previousHeight);
            previousHeight = currentHeight;
        }
        return bumpiness;
    }

    // Calculates the total height of the columns on the board
    public static int calculateTotalHeight(Matrix board) {
        int totalHeight = 0;
        for (int col = 0; col < board.getShape().getNumCols(); col++) {
            totalHeight += getColumnHeight(board, col);
        }
        return totalHeight;
    }

    // Helper to get the height of a column
    private static int getColumnHeight(Matrix board, int col) {
        for (int row = 0; row < board.getShape().getNumRows(); row++) {
            if (board.get(row, col) != 0.0) {
                return board.getShape().getNumRows() - row;
            }
        }
        return 0;
    }

    /**
     * This method is used to decide if we should follow our current policy
     * (i.e. our q-function), or if we should ignore it and take a random action
     * (i.e. explore).
     *
     * Remember, as the q-function learns, it will start to predict the same "good" actions
     * over and over again. This can prevent us from discovering new, potentially even
     * better states, which we want to do! So, sometimes we should ignore our policy
     * and explore to gain novel experiences.
     *
     * The current implementation chooses to ignore the current policy around 5% of the time.
     * While this strategy is easy to implement, it often doesn't perform well and is
     * really sensitive to the EXPLORATION_PROB. I would recommend devising your own
     * strategy here.
     */
    @Override
    public boolean shouldExplore(final GameView game,
                                 final GameCounter gameCounter)
    {
        // new game
        if (prevGameId != gameCounter.getCurrentGameIdx()) {
            prevGameId = gameCounter.getCurrentGameIdx();
            rounds = 0;
        }
        
        double phaseFactor = Math.max(0.01, 1.0 - (double) gameCounter.getCurrentPhaseIdx() / gameCounter.getNumPhases());
        if (gameCounter.getNumPhases() == 0){
            phaseFactor = 1;
        }
        double epsilon = 0.25 * phaseFactor; 
        // new phase
        if (prevPhaseId != gameCounter.getCurrentPhaseIdx()) {
            prevPhaseId = gameCounter.getCurrentPhaseIdx();
            phase += 1;
            // System.out.println("PERCENT: " + epsilon);
        }
        return this.random.nextDouble() < epsilon;
    }

    /**
     * This method is a counterpart to the "shouldExplore" method. Whenever we decide
     * that we should ignore our policy, we now have to actually choose an action.
     *
     * You should come up with a way of choosing an action so that the model gets
     * to experience something new. The current implemention just chooses a random
     * option, which in practice doesn't work as well as a more guided strategy.
     * I would recommend devising your own strategy here.
     */
    @Override
    public Mino getExplorationMove(final GameView game)
    {
        int randIdx = this.getRandom().nextInt(game.getFinalMinoPositions().size());
        return game.getFinalMinoPositions().get(randIdx);
    }

    /**
     * This method is called by the TrainerAgent after we have played enough training games.
     * In between the training section and the evaluation section of a phase, we need to use
     * the exprience we've collected (from the training games) to improve the q-function.
     *
     * You don't really need to change this method unless you want to. All that happens
     * is that we will use the experiences currently stored in the replay buffer to update
     * our model. Updates (i.e. gradient descent updates) will be applied per minibatch
     * (i.e. a subset of the entire dataset) rather than in a vanilla gradient descent manner
     * (i.e. all at once)...this often works better and is an active area of research.
     *
     * Each pass through the data is called an epoch, and we will perform "numUpdates" amount
     * of epochs in between the training and eval sections of each phase.
     */
    @Override
    public void trainQFunction(Dataset dataset,
                               LossFunction lossFunction,
                               Optimizer optimizer,
                               long numUpdates)
    {
        for(int epochIdx = 0; epochIdx < numUpdates; ++epochIdx)
        {
            dataset.shuffle();
            Iterator<Pair<Matrix, Matrix> > batchIterator = dataset.iterator();

            while(batchIterator.hasNext())
            {
                Pair<Matrix, Matrix> batch = batchIterator.next();

                try
                {
                    Matrix YHat = this.getQFunction().forward(batch.getFirst());

                    optimizer.reset();
                    this.getQFunction().backwards(batch.getFirst(),
                                                  lossFunction.backwards(YHat, batch.getSecond()));
                    optimizer.step();
                } catch(Exception e)
                {
                    e.printStackTrace();
                    System.exit(-1);
                }
            }
        }
    }

    /**
     * This method is where you will devise your own reward signal. Remember, the larger
     * the number, the more "pleasurable" it is to the model, and the smaller the number,
     * the more "painful" to the model.
     *
     * This is where you get to tell the model how "good" or "bad" the game is.
     * Since you earn points in this game, the reward should probably be influenced by the
     * points, however this is not all. In fact, just using the points earned this turn
     * is a **terrible** reward function, because earning points is hard!!
     *
     * I would recommend you to consider other ways of measuring "good"ness and "bad"ness
     * of the game. For instance, the higher the stack of minos gets....generally the worse
     * (unless you have a long hole waiting for an I-block). When you design a reward
     * signal that is less sparse, you should see your model optimize this reward over time.
     */
    @SuppressWarnings("unused")
    @Override
    public double getReward(GameView game) {
        double reward = 0.0;
        // if (game.getTotalScore() >= 8){
        //     System.out.println(game.getTotalScore());
        // }

        // reward
        // reward += game.getScoreThisTurn() * 10;

        // punish
        double holes = calculateHoles(game.getBoard());
        double maxHeight = calculateMaxHeight(game.getBoard());
        double emptyBelowMax = emptySpacesBelowMaxHeight(game.getBoard());
        // double currentAverageHeight = calculateAverageHeight(game.getBoard());
        
        // max height
        // reward -= (maxHeight / 5);

        // holes
        // if (holes <= 5) {
        //     reward += 1;
        // }
        // else if (holes <= 10) {
        //     reward += 0.5;
        // }
        // else {
        //     reward -= 2;
        // }

        // score
        reward += (Math.pow(game.getScoreThisTurn(), 2)/5);  
        reward = (-0.510066) * calculateAggregateHeight(game.getBoard()) + (0.760666) * game.getScoreThisTurn() + (-0.35663) * holes * (-0.184483) * calculateBumpiness(game.getBoard());
        // if (phase > 1000) {
        //     reward += (Math.pow(game.getScoreThisTurn(), 2)/5); 
        // }
        // rounds exponential points
        rounds += 1;
        //reward += rounds/10;
        //reward += Math.exp(0.2 * (rounds - 15));
        // if (rounds < 15) {
        //     reward -= 0.5;
        // }
        // else if (rounds < 20) {
        //     reward += 0.5;
        // }
        // else if (rounds <25) {
        //     reward += 1.0;
        // }
        // else {
        //     reward += (rounds / 20);
        // }
        // System.out.println(rounds);
        // if (game.didAgentLose()) {
        //     rounds = 0;
        // }

        // // empty below max
        // if (emptyBelowMax <= 6) {
        //     reward += 2.0;
        // }
        // else if (emptyBelowMax <= 12) {
        //     reward -= 1;
        // }
        // else if (emptyBelowMax <= 20){
        //     reward -= 2.0;
        // }
        // else{
        //     reward -= 4;
        // }

        // // max height
        // if (maxHeight <= 6) {
        //     // low and few holes
        //     if (holes < 5){
        //         reward += 2;
        //     }
        //     else{
        //         reward += 0.7;
        //     }
        // }
        // else if (maxHeight <= 12){
        //     if (holes > 7){
        //         reward -= 0.3;
        //     }
        //     else {
        //         reward += 1;
        //     }
        // }
        // else {
        //     if (holes > 10) {
        //         reward -= 2;
        //     }
        //     else {
        //         reward -= 0.5;
        //     }
        // }

        // // holes
        // if (holes <= 5) {
        //     reward += 1;
        // }
        // else if (holes <= 10) {
        //     reward += 0;
        // }
        // else if (holes <= 20){
        //     reward -= 2;
        // }
        // else {
        //     reward -= 3;
        // }
        
        // reward -= currentAverageHeight;
        // reward -= holes;
        // reward += game.getScoreThisTurn()*5;
        if (false) {
            System.out.println("SCORE x 10: " + game.getScoreThisTurn() * 10);
            System.out.println("CURRENT HOLES: " + holes);
            // System.out.println("AVERAGE HEIGHT: " + currentAverageHeight);
            System.out.println("TOTAL SCORE: " + game.getTotalScore());
            System.out.println("DID AGENT LOSE: " + game.getScoreThisTurn());
            System.out.println("MAX HEIGHT: " + maxHeight);
            if (game.didAgentLose()) {
                System.out.println("GAME LOST\n\n\n");
                reward -= 100; 
            }
            System.out.println("REWARD: " + reward);
            System.out.println();
            try {
                TimeUnit.SECONDS.sleep(2);
            } catch (InterruptedException e) {
                // TODO Auto-generated catch block
                e.printStackTrace();
            }
        }
        return reward;
    }

    private double calculateMaxHeight(Board board) {
        int maxHeight = 0;
        for (int col = 0; col < Board.NUM_COLS; col++) {
            for (int row = 0; row < Board.NUM_ROWS; row++) {
                if (board.isCoordinateOccupied(col, row)) {
                    int currentHeight = Board.NUM_ROWS - row; 
                    maxHeight = Math.max(maxHeight, currentHeight);
                    break; 
                }
            }
        }
        return maxHeight;
    }

    // private double calculateMaxWidth(Board board) {
    //     int maxWidth = 0;
    //     for (int col = 0; col < Board.NUM_COLS; col++) {
    //         for (int row = 0; row < Board.NUM_ROWS; row++) {
    //             if (board.isCoordinateOccupied(col, row)) {
    //                 int currentHeight = Board.NUM_ROWS - row; 
    //                 maxWidth = Math.max(maxWidth, currentHeight);
    //                 break; 
    //             }
    //         }
    //     }
    //     return maxHeight;
    // }

    private double calculateHoles(Board board) {
        int totalHoles = 0;
        for (int col = 0; col < Board.NUM_COLS; col++) {
            boolean blockFound = false;
            int holesInColumn = 0;
            for (int row = 0; row < Board.NUM_ROWS; row++) {
                if (board.isCoordinateOccupied(col, row)) {
                    blockFound = true;
                } else if (blockFound) {
                    holesInColumn++;
                }
            }
            totalHoles += holesInColumn;
        }
        return totalHoles;
    }

    public int emptySpacesBelowMaxHeight(Board board) {
        int totalEmptySpaces = 0;
        int numColumns = Board.NUM_COLS;
        double maxHeight = calculateMaxHeight(board);
    
        // Iterate over each column
        for (int col = 0; col < numColumns; col++) {
            int columnHeight = 0;
            for (int row = 0; row < Board.NUM_ROWS; row++) {
                if (board.isCoordinateOccupied(col, row)) {
                    columnHeight = Board.NUM_ROWS - row;
                    break;
                }
            }

            for (int row = Board.NUM_ROWS - columnHeight; row < Board.NUM_ROWS - maxHeight; row++) {
                if (!board.isCoordinateOccupied(col, row)) {
                    totalEmptySpaces++;
                }
            }
        }
    
        return totalEmptySpaces;
    }
        // Method to calculate the aggregate height of the board
        public static int calculateAggregateHeight(Board board) {
            int totalHeight = 0;
            for (int col = 0; col < Board.NUM_COLS; col++) {
                totalHeight += getColumnHeight(board, col);
            }
            return totalHeight;
        }
    
        // Helper method to get the height of a column
        private static int getColumnHeight(Board board, int col) {
            for (int row = 0; row < Board.NUM_ROWS; row++) {
                if (board.isCoordinateOccupied(col, row)) {
                    return Board.NUM_ROWS - row;  // Height is total rows minus the current row index
                }
            }
            return 0;  // Return 0 if the column is empty
        }

    // Method to calculate the bumpiness of the board
    public static int calculateBumpiness(Board board) {
        int bumpiness = 0;
        int previousHeight = getColumnHeight(board, 0);
        for (int col = 1; col < Board.NUM_COLS; col++) {
            int currentHeight = getColumnHeight(board, col);
            bumpiness += Math.abs(currentHeight - previousHeight);
            previousHeight = currentHeight;
        }
        return bumpiness;
    }

}
