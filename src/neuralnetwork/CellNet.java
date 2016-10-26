/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

package neuralnetwork;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.text.DecimalFormat;
import java.util.Arrays;
import org.neuroph.core.NeuralNetwork;
import org.neuroph.core.data.DataSet;
import org.neuroph.core.data.DataSetRow;
import org.neuroph.nnet.MultiLayerPerceptron;
import org.neuroph.nnet.learning.MomentumBackpropagation;
import org.neuroph.util.TransferFunctionType;

/**
 * @author Jhony Luiz de Almeida
 */
public class CellNet {

    enum LocalizationSite {CYT, NUC, MIT, ME3, ME2, ME1, EXC, VAC, POX, ERL}
    
    public static void main(String[] args) {
        
        DataSet trainingSet = new DataSet(8, 10);
        File file = new File("cells.txt");
        try(
            BufferedReader reader = new BufferedReader(new FileReader(file))
        ) {
            int cont = 0;
            for(String s = reader.readLine(); s != null; s = reader.readLine()) {
                //Pega valores da string separados por vírgula
                String[] values = s.split("  ");
                if(values.length != 10) break; //pula linhas com problema
                //Criação do array de entradas
                double[] inputs = new double[values.length - 2];
                for(int i = 1; i < inputs.length + 1; i++) {
                    inputs[i - 1] = Double.parseDouble(values[i]);
                }
                //Criação do array de saídas
                double[] output = new double[10]; 
                int activation = LocalizationSite.valueOf(values[values.length - 1]).ordinal();
                output[activation] = 1.0;
                System.out.println(Arrays.toString(inputs) + "  - " + Arrays.toString(output));
                //Adição do registro de treinamento
                trainingSet.addRow(new DataSetRow(inputs, output));
                cont++;
            }
        } catch(IOException ex) {
            
        }

        MomentumBackpropagation backPropagation = new MomentumBackpropagation();
        backPropagation.setMomentum(0.2);
        backPropagation.setMaxIterations(100000);
        backPropagation.setMaxError(0.01);
        backPropagation.setLearningRate(0.1);
        
        NeuralNetwork perceptron = new MultiLayerPerceptron(TransferFunctionType.TANH, 8, 10, 10);
        perceptron.setLearningRule(backPropagation);
        perceptron.learn(trainingSet);
        perceptron.save("cellNet.nnet");
        
        //NeuralNetwork perceptron = NeuralNetwork.createFromFile("cellNet.nnet"); 
        testNeuralNetwork(perceptron);
    }
    
    public static void testNeuralNetwork(NeuralNetwork nnet) {
        nnet.setInput(new double[] {0.50,  0.34,  0.55,  0.21,  0.50,  0.00,  0.49,  0.22}); //ME2
        nnet.calculate();
        for(int i = 0; i < 10; i++) {
            System.out.println(LocalizationSite.values()[i].name() + ": " + 
                DecimalFormat.getNumberInstance().format(nnet.getOutput()[i]));
        }
        
    }
    
}
