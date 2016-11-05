package samples;

import java.util.Arrays;
import org.neuroph.core.NeuralNetwork;
import org.neuroph.core.data.DataSet;
import org.neuroph.nnet.MultiLayerPerceptron;
import org.neuroph.core.data.DataSetRow;
import org.neuroph.nnet.learning.MomentumBackpropagation;

/**
* This sample shows how to create, train, save and load simple Multi Layer Perceptron
*/
public class XORNet {

    public static void main(String[] args) {

        // create training set (logical XOR function)
        DataSet trainingSet = new DataSet(2, 1);
        trainingSet.addRow(new DataSetRow(new double[]{0, 0}, new double[]{0}));
        trainingSet.addRow(new DataSetRow(new double[]{0, 1}, new double[]{1}));
        trainingSet.addRow(new DataSetRow(new double[]{1, 0}, new double[]{1}));
        trainingSet.addRow(new DataSetRow(new double[]{1, 1}, new double[]{0}));

        NeuralNetwork myMlPerceptron = new MultiLayerPerceptron(2, 3, 1);
        
        MomentumBackpropagation b = new MomentumBackpropagation();
        b.setMomentum(0.7);
        b.setMaxIterations(10000);
        b.setMaxError(0.0001);
        b.setLearningRate(0.2);
        
        myMlPerceptron.setLearningRule(b);
        myMlPerceptron.learn(trainingSet);
        myMlPerceptron.save("myMlPerceptron.nnet");
        
        NeuralNetwork loadedMlPerceptron = NeuralNetwork.createFromFile("myMlPerceptron.nnet"); 
        loadedMlPerceptron.setInput(1, 0);
        loadedMlPerceptron.calculate();
        System.out.println("Resultado do teste: " + loadedMlPerceptron.getOutput()[0]);

    }

    public static void testNeuralNetwork(NeuralNetwork nnet, DataSet tset) {
        for (DataSetRow dataRow : tset.getRows()) {
            nnet.setInput(dataRow.getInput());
            nnet.calculate();
            double[] networkOutput = nnet.getOutput();
            System.out.print("Input: " + Arrays.toString(dataRow.getInput()));
            System.out.println(" Output: " + Arrays.toString(networkOutput));
        }

    }
 

}
