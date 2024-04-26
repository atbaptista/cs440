package src.pas.tetris.agents;

import java.util.Arrays;

// 0.2 (218.7045/300) + 0.4 (145/200) + 0.4 (110/150)
// java -cp lib/*:. edu.bu.tetris.Main -q src.pas.tetris.agents.TetrisQAgent -p 5000 -t 100 -v 50 -n 0.01 -b 5000 -s | tee run.log
// java -cp "./lib/*;." edu.bu.tetris.Main -q src.pas.tetris.agents.TetrisQAgent -p 5000 -t 100 -v 50 -n 0.01 -b 10000 -s -c 1000000000 -o params/newReward
// javac -cp "./lib/*;." @tetris.srcs

// SYSTEM IMPORTS
import java.util.Iterator;
import java.util.List;
import java.util.Random;


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

    public static final double EXPLORATION_PROB = 0.05;

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
        // build a single-hidden-layer feedforward network
        // this example will create a 3-layer neural network (1 hidden layer)
        // in this example, the input to the neural network is the
        // image of the board unrolled into a giant vector
        // final int numPixelsInImage = Board.NUM_ROWS * Board.NUM_COLS;
        // final int inputSize = numPixelsInImage + 2;
        final int inputSize = 8;
        final int hiddenDim = 2 * inputSize;
        final int outDim = 1;

        Sequential qFunction = new Sequential();
        qFunction.add(new Dense(inputSize, hiddenDim));
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
        // Attempt to get the grayscale image of the board with the potential mino action and flatten it
        // Matrix flattenedImage = null;
        // try {
        //     flattenedImage = game.getGrayscaleImage(potentialAction).flatten();
        // } catch (Exception e) {
        //     e.printStackTrace();
        //     System.exit(-1);
        // }

        // double maxColumnHeight = calculateMaxHeight(game.getBoard());
        // double holes = calculateHoles(game.getBoard());
        // Matrix extendedFeatures = Matrix.zeros(1, flattenedImage.numel() + 2);

        // for (int i = 0; i < flattenedImage.numel(); i++) {
        //     double value = 0;
        //     try {
        //         value = flattenedImage.get(0, i);  
        //     } catch (IndexOutOfBoundsException e) {
        //         e.printStackTrace();
        //         System.exit(-1);
        //     }
        //     extendedFeatures.set(0, i, value);
        // }

        // extendedFeatures.set(0, flattenedImage.numel(), maxColumnHeight);
        // extendedFeatures.set(0, flattenedImage.numel() + 1, holes);

        // return extendedFeatures;

        // Calculate strategic features based on the board state
        BoardFeatures features = calculateBoardFeatures(game.getBoard());

        // There are eight strategic features: 
        // cumulative height, total hole depth, rows with holes, row transitions, 
        // column transitions, max well depth, max height, and total holes.
        // We exclude well depths from this feature set.
        int numFeatures = 8;
        Matrix extendedFeatures = Matrix.zeros(1, numFeatures);

        int idx = 0;
        extendedFeatures.set(0, idx++, features.cumulativeHeight);
        extendedFeatures.set(0, idx++, features.totalHoleDepth);
        extendedFeatures.set(0, idx++, features.rowsWithHoles);
        extendedFeatures.set(0, idx++, features.rowTransitions);
        extendedFeatures.set(0, idx++, features.columnTransitions);
        extendedFeatures.set(0, idx++, features.maxWellDepth);
        extendedFeatures.set(0, idx++, features.maxHeight);
        extendedFeatures.set(0, idx++, features.totalHoles);

        return extendedFeatures;

    }

    private BoardFeatures calculateBoardFeatures(Board board) {
        int numRows = Board.NUM_ROWS;
        int numCols = Board.NUM_COLS;
    
        double cumulativeHeight = 0;
        double totalHoleDepth = 0;
        int rowsWithHoles = 0;
        int rowTransitions = 0;
        int columnTransitions = 0;
        double maxWellDepth = 0;
        double maxHeight = 0; 
        int totalHoles = 0;   
    
        double[] columnHeights = new double[numCols];
        double[] wellDepths = new double[numCols];
        boolean[] rowHasHole = new boolean[numRows];
    
        // Initialize column heights and well depths
        Arrays.fill(columnHeights, numRows); // Assume initially all columns are full
        Arrays.fill(wellDepths, 0);
    
        for (int row = 0; row < numRows; row++) {
            boolean previousCellOccupied = board.isCoordinateOccupied(0, row);
            for (int col = 0; col < numCols; col++) {
                boolean currentCellOccupied = board.isCoordinateOccupied(col, row);
                
                // Calculate column heights and max height
                if (currentCellOccupied && columnHeights[col] == numRows) {
                    columnHeights[col] = row; // Set height to the first occupied cell from top
                    int currentHeight = numRows - row;
                    maxHeight = Math.max(maxHeight, currentHeight); // Update max height
                }
    
                // Check row transitions
                if (col > 0 && currentCellOccupied != previousCellOccupied) {
                    rowTransitions++;
                }
                previousCellOccupied = currentCellOccupied;
    
                // Check column transitions
                if (row > 0 && currentCellOccupied != board.isCoordinateOccupied(col, row - 1)) {
                    columnTransitions++;
                }
    
                // Check for holes and hole depth
                if (!currentCellOccupied && col < numCols && row < numRows - 1 && board.isCoordinateOccupied(col, row + 1)) {
                    totalHoleDepth += 1; // Increment depth for each hole
                    rowHasHole[row] = true;
                    totalHoles++; // Counting holes
                }
            }
            // Update row transition count for row end
            if (board.isCoordinateOccupied(numCols - 1, row) != board.isCoordinateOccupied(numCols - 1, 0)) {
                rowTransitions++;
            }
        }
    
        // Calculate well depths and cumulative heights
        for (int col = 0; col < numCols; col++) {
            double wellDepth = 0;
            boolean foundWell = false;
            for (int row = (int)columnHeights[col] + 1; row < numRows && !board.isCoordinateOccupied(col, row); row++) {
                wellDepth++;
                foundWell = true;
            }
            if (foundWell) {
                wellDepths[col] = wellDepth;
                maxWellDepth = Math.max(maxWellDepth, wellDepth);
            }
            cumulativeHeight += numRows - columnHeights[col];
        }
    
        // Count rows with holes
        for (boolean hasHole : rowHasHole) {
            if (hasHole) rowsWithHoles++;
        }
    
        // Return an object encapsulating all the calculated features
        return new BoardFeatures(cumulativeHeight, wellDepths, totalHoleDepth, rowsWithHoles, rowTransitions,
                                 columnTransitions, maxWellDepth, maxHeight, totalHoles);
    }
    
    private static class BoardFeatures {
        double cumulativeHeight;
        double[] wellDepths;
        double totalHoleDepth;
        int rowsWithHoles;
        int rowTransitions;
        int columnTransitions;
        double maxWellDepth;
        double maxHeight;
        int totalHoles;
    
        public BoardFeatures(double cumulativeHeight, double[] wellDepths, double totalHoleDepth, int rowsWithHoles,
                             int rowTransitions, int columnTransitions, double maxWellDepth, double maxHeight, int totalHoles) {
            this.cumulativeHeight = cumulativeHeight;
            this.wellDepths = wellDepths;
            this.totalHoleDepth = totalHoleDepth;
            this.rowsWithHoles = rowsWithHoles;
            this.rowTransitions = rowTransitions;
            this.columnTransitions = columnTransitions;
            this.maxWellDepth = maxWellDepth;
            this.maxHeight = maxHeight;
            this.totalHoles = totalHoles;
        }
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
    
    private double calculateHoles(Board board) {
        int totalHoles = 0;
        for (int col = 0; col < Board.NUM_COLS; col++) {
            boolean blockFound = false;
            for (int row = 0; row < Board.NUM_ROWS; row++) {
                if (board.isCoordinateOccupied(col, row)) {
                    blockFound = true; 
                } else if (blockFound) {
                    totalHoles++;
                }
            }
        }
        return totalHoles;
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
        return this.getRandom().nextDouble() <= EXPLORATION_PROB;
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
    @Override
    public double getReward(GameView game) {
        double reward = 0.0;
        if (game.getTotalScore() >= 10){
            reward += 100;
            System.out.println(game.getTotalScore());
        }
        // // Reward based on score achieved this turn - consider smoothing this
        // reward += game.getScoreThisTurn() * 100;  // Scaling the score to make it more significant
        // Reward exponentially based on the score achieved this turn
        // reward += Math.exp((double) game.getScoreThisTurn() / 10.0) * 10;

        // // Additional bonus for increasing the total score
        // reward += Math.exp((double) game.getTotalScore() / 10.0) * 10;
        //reward += game.getTotalScore() / 100.0;  // Normalize total score to manage scale
        // Penalize for high stack heights - encourage lower heights
        double maxHeight = calculateMaxHeight(game.getBoard());
        if (maxHeight > Board.NUM_ROWS / 2) {
            reward -= (maxHeight - Board.NUM_ROWS / 2) * 20;  // Increasing penalty for heights above half the board
        }
    
        // Reward for keeping the board lower than certain thresholds
        reward += (Board.NUM_ROWS - maxHeight) * 5;  // Reward for each row below the max height
    
        // Penalize for holes
        double holes = calculateHoles(game.getBoard());
        reward -= holes * 30;  // Strong penalty for each hole
    
        // Check if the game is lost
        if (game.didAgentLose()) {
            reward -= 1000;  // Large penalty for losing
        }
    
        return reward;
    }
}
