package samples;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import org.neuroph.core.data.DataSet;
import org.neuroph.core.data.DataSetRow;
import org.neuroph.nnet.MultiLayerPerceptron;
import org.neuroph.nnet.learning.MomentumBackpropagation;
import org.neuroph.util.TransferFunctionType;

public class LISNet {

    public static void main(String[] args) {
        int inputSize = 4;
        int outputSize = 3;
        int erros = 0;
        
        DataSet trainingSet = new DataSet(inputSize, outputSize);
        File file = new File("LIS.TRN");
        
        try (BufferedReader reader = new BufferedReader(new FileReader(file))) {
            for (int i = 0; i < 90; i ++) {
                
                //Criação do array de entradas
                double[] inputs = new double[inputSize];
                inputs[0] = Double.parseDouble(reader.readLine());
                inputs[1] = Double.parseDouble(reader.readLine());
                inputs[2] = Double.parseDouble(reader.readLine());
                inputs[3] = Double.parseDouble(reader.readLine());
                
                //Criação do array de saídas
                double[] outputs = new double[outputSize];
                outputs[0] = Double.parseDouble(reader.readLine());
                outputs[1] = Double.parseDouble(reader.readLine());
                outputs[2] = Double.parseDouble(reader.readLine());
                
                System.out.println(Arrays.toString(inputs) + "  - " + Arrays.toString(outputs));
                
                //Adição do registro de treinamento
                trainingSet.addRow(new DataSetRow(inputs, outputs));
            }
        } catch(IOException ex) {
            System.out.println("Error reading file.");
        }
        
        System.out.println("Lines read: " + trainingSet.getRows().size());
        
        MomentumBackpropagation backPropagation = new MomentumBackpropagation();
        backPropagation.setMomentum(0.1);
        backPropagation.setMaxIterations(10000);
        backPropagation.setMaxError(0.01);
        backPropagation.setLearningRate(0.1);

        MultiLayerPerceptron perceptron = new MultiLayerPerceptron(TransferFunctionType.SIGMOID, inputSize, 4, outputSize);

        perceptron.setLearningRule(backPropagation);
        perceptron.learn(trainingSet);
//        
        perceptron.save("lis.nnet");
        System.out.println("Network saved");
        
        
        File trainFile = new File("LISEXE.TRN");
        double[] exOut;
        int[] highest = new int[10];
        
        try (BufferedReader reader = new BufferedReader(new FileReader(trainFile))) {
            for (int i = 0; i < 60; i ++) {
                
                //Criação do array de entradas
                double[] inputs = new double[inputSize];
                inputs[0] = Double.parseDouble(reader.readLine());
                inputs[1] = Double.parseDouble(reader.readLine());
                inputs[2] = Double.parseDouble(reader.readLine());
                inputs[3] = Double.parseDouble(reader.readLine());
                
                //Criação do array de saídas
                exOut = new double[outputSize];
                exOut[0] = Double.parseDouble(reader.readLine());
                exOut[1] = Double.parseDouble(reader.readLine());
                exOut[2] = Double.parseDouble(reader.readLine());
                
                perceptron.setInput(inputs);
                perceptron.calculate();
                double[] actNr = perceptron.getOutput();
                
                double maxExpected;
                int expectedIndex;
                if (exOut[0] > exOut[1] && exOut[0] > exOut[2]) {
                    maxExpected = exOut[0];
                    expectedIndex = 0;
                } else if (exOut[1] > exOut[2]) {
                    maxExpected = exOut[1];
                    expectedIndex = 1;
                } else {
                    maxExpected = exOut[2];
                    expectedIndex = 2;
                }
                
                double maxActive;
                int activeIndex;
                if (actNr[0] > actNr[1] && actNr[0] > actNr[2]) {
                    maxActive = actNr[0];
                    activeIndex = 0;
                } else if (actNr[1] > actNr[2]) {
                    maxActive = actNr[1];
                    activeIndex = 1;
                } else {
                    maxActive = actNr[2];
                    activeIndex = 2;
                }
                
                if (Math.round(maxExpected) != Math.round(maxActive)) {
                    erros++;
                    System.out.println("Era esperado " + exOut[expectedIndex] + " e foi obtido " + actNr[activeIndex]);
                }
                
                int indiceAdd = (((int) Math.round(maxActive * 100)) / 10) - 1;
                highest[indiceAdd]++;
                
//                System.out.println("Entradas: " + Arrays.toString(inputs));
//                System.out.println("Saída esperada: " + Arrays.toString(exOut));
//                System.out.println("Saída Obtida: " + Arrays.toString(perceptron.getOutput()));
            }
        } catch(IOException ex) {
            System.out.println("Error reading file.");
        }
        
        System.out.println("\nErros: " + erros);
        for (int z = 0; z < 10; z++) {
            System.out.println((z * 10) + "% a " + ((z + 1) * 10) + "% \t" + Math.round(highest[z]  / 60.0 * 100.0) + "%");
        }
//        
        
        //Utilizar a rede depois de salva
        //NeuralNetwork perceptron = MultiLayerPerceptron.createFromFile("pokerNet5.nnet"); 
        
//        perceptron.calculate();
//        System.out.println(" Output: " + Arrays.toString(perceptron.getOutput()));
    }
    
}
