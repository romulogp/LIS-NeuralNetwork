package samples;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.util.Arrays;
import org.neuroph.core.data.DataSet;
import org.neuroph.core.data.DataSetRow;
import org.neuroph.nnet.MultiLayerPerceptron;
import org.neuroph.nnet.learning.MomentumBackpropagation;
import org.neuroph.util.TransferFunctionType;

public class LISNet {
    
    static interface Naipe {
        int QUANTIDADE = 4;
        double OUROS = 1;
        double ESPADAS = 2; 
        double COPAS = 3;
        double PAUS = 4;
    }
    
    static interface Carta {
        int QUANTIDADE = 13;
        double AS = 1;
        double DOIS = 2;
        double TRES = 3;
        double QUATRO = 4;
        double CINCO = 5;
        double SEIS = 6;
        double SETE = 7; 
        double OITO = 8; 
        double NOVE = 9;
        double DEZ = 10; 
        double VALETE = 11; 
        double DAMA = 12; 
        double REI = 13;
    }
    
    public static void main(String[] args) {
        int inputSize = 4;
        int outputSize = 3;
        
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
        backPropagation.setMomentum(0.03);
        backPropagation.setMaxIterations(10000);
        backPropagation.setMaxError(0.01);
        backPropagation.setLearningRate(0.01);
        
        MultiLayerPerceptron perceptron = new MultiLayerPerceptron(TransferFunctionType.SIN, inputSize, 4, outputSize);
        perceptron.setLearningRule(backPropagation);
        perceptron.learn(trainingSet);
//        
        perceptron.save("lis.nnet");
        System.out.println("Network saved");
        
        
        //Utilizar a rede depois de salva
        //NeuralNetwork perceptron = MultiLayerPerceptron.createFromFile("pokerNet5.nnet"); 
        perceptron.setInput(5.5, 2.4, 3.7, 1.0);
        perceptron.calculate();
        System.out.println(" Output: " + Arrays.toString(perceptron.getOutput()));
    }
    
}
