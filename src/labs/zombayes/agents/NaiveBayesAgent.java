package src.labs.zombayes.agents;


// SYSTEM IMPORTS
import java.io.InputStream;
import java.io.OutputStream;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;


// JAVA PROJECT IMPORTS
import edu.bu.labs.zombayes.agents.SurvivalAgent;
import edu.bu.labs.zombayes.features.Features.FeatureType;
import edu.bu.labs.zombayes.linalg.Matrix;
import edu.bu.labs.zombayes.utils.Pair;



public class NaiveBayesAgent
    extends SurvivalAgent
{

    public static class NaiveBayes
        extends Object
    {
        private Map<Integer, Double> classPriors = new HashMap<>();
        private Map<Integer, Map<Integer, Double>> featureMeans = new HashMap<>();
        private Map<Integer, Map<Integer, Double>> featureVariances = new HashMap<>();
        private Map<Integer, Map<Integer, Map<Integer, Integer>>> featureFrequencies = new HashMap<>();
        private Map<Integer, Integer> classCounts = new HashMap<>(); // Proper declaration
        private int numFeatures;

        public static final FeatureType[] FEATURE_HEADER = {FeatureType.CONTINUOUS,
                                                            FeatureType.CONTINUOUS,
                                                            FeatureType.DISCRETE,
                                                            FeatureType.DISCRETE};

        // TODO: complete me!
        public NaiveBayes()
        {
            this.numFeatures = FEATURE_HEADER.length;
        }

        public void fit(Matrix features, Matrix labels) {
            int numRows = features.getShape().getNumRows();
            // Determine number of classes
            HashSet<Double> uniqueClasses = new HashSet<>();
            for (int i = 0; i < numRows; i++) {
                uniqueClasses.add(labels.get(i, 0));
            }
            int numClasses = uniqueClasses.size();
    
            // Initialize for each class
            for (Double classVal : uniqueClasses) {
                int classIntVal = classVal.intValue();
                classCounts.put(classIntVal, classCounts.getOrDefault(classVal, 0) + 1);
                classPriors.put(classIntVal, 0.0);
                for (int j = 0; j < numFeatures; j++) {
                    if (FEATURE_HEADER[j] == FeatureType.CONTINUOUS) {
                        featureMeans.computeIfAbsent(classIntVal, k -> new HashMap<>()).put(j, 0.0);
                        featureVariances.computeIfAbsent(classIntVal, k -> new HashMap<>()).put(j, 0.0);
                    } else {
                        featureFrequencies.computeIfAbsent(classIntVal, k -> new HashMap<>()).put(j, new HashMap<>());
                    }
                }
            }
    
            // Populate counts and sums for features
            for (int i = 0; i < numRows; i++) {
                int classVal = (int) labels.get(i, 0);
                classPriors.put(classVal, classPriors.get(classVal) + 1); // Count class occurrences
    
                for (int j = 0; j < numFeatures; j++) {
                    double featureVal = features.get(i, j);
                    if (FEATURE_HEADER[j] == FeatureType.CONTINUOUS) {
                        // Sum up for mean calculation later
                        double currentSum = featureMeans.get(classVal).get(j);
                        featureMeans.get(classVal).put(j, currentSum + featureVal);
                    } else {
                        // Count frequencies for discrete features
                        featureFrequencies.get(classVal).get(j).merge((int)featureVal, 1, Integer::sum);
                    }
                }
            }
    
            // Calculate means, variances, and normalize class priors
            for (int classVal : classPriors.keySet()) {
                double classCount = classPriors.get(classVal);
                classPriors.put(classVal, classCount / numRows);
    
                for (int j = 0; j < numFeatures; j++) {
                    if (FEATURE_HEADER[j] == FeatureType.CONTINUOUS) {
                        double sum = featureMeans.get(classVal).get(j);
                        double mean = sum / classCount;
                        featureMeans.get(classVal).put(j, mean);
    
                        // Calculate variance
                        double varianceSum = 0.0;
                        for (int i = 0; i < numRows; i++) {
                            if ((int)labels.get(i, 0) == classVal) {
                                double diff = features.get(i, j) - mean;
                                varianceSum += diff * diff;
                            }
                        }
                        double variance = varianceSum / classCount;
                        featureVariances.get(classVal).put(j, variance);
                    }
                }
            }
        }

        public int predict(Matrix features) {
            double bestLogProb = Double.NEGATIVE_INFINITY;
            int bestClass = -1;
            
            for (Integer classVal : classPriors.keySet()) {
                double logProb = Math.log(classPriors.get(classVal));
                
                for (int j = 0; j < numFeatures; j++) {
                    double featureVal = features.get(0, j);
                    
                    if (FEATURE_HEADER[j] == FeatureType.CONTINUOUS) {
                        // Continuous feature, use Gaussian probability
                        double mean = featureMeans.get(classVal).get(j);
                        double variance = featureVariances.get(classVal).get(j);
                        double logGaussianProb = Math.log(calculateGaussianProbability(featureVal, mean, variance));
                        logProb += logGaussianProb;
                    } else {
                        // Discrete feature, use frequency-based probability
                        // Note: Adding 1 for Laplace smoothing
                        Map<Integer, Integer> frequencies = featureFrequencies.get(classVal).get(j);
                        int totalClassCount = classCounts.get(classVal);
                        int featureFrequency = frequencies.getOrDefault((int) featureVal, 0) + 1; // Laplace smoothing
                        int possibleValuesCount = frequencies.size() + 1; // Considering unseen values
                        
                        double logFrequencyProb = Math.log((double) featureFrequency / (totalClassCount + possibleValuesCount));
                        logProb += logFrequencyProb;
                    }
                }
                
                if (logProb > bestLogProb) {
                    bestLogProb = logProb;
                    bestClass = classVal;
                }
            }
            
            return bestClass;
        }
        private double calculateGaussianProbability(double x, double mean, double variance) {
            double exponent = Math.exp(-(Math.pow(x - mean, 2) / (2 * variance)));
            return (1 / (Math.sqrt(2 * Math.PI * variance))) * exponent;
        }
    }
    
    private NaiveBayes model;

    public NaiveBayesAgent(int playerNum, String[] args)
    {
        super(playerNum, args);
        this.model = new NaiveBayes();
    }

    public NaiveBayes getModel() { return this.model; }

    @Override
    public void train(Matrix X, Matrix y_gt)
    {
        System.out.println(X.getShape() + " " + y_gt.getShape());
        this.getModel().fit(X, y_gt);
    }

    @Override
    public int predict(Matrix featureRowVector)
    {
        return this.getModel().predict(featureRowVector);
    }

}