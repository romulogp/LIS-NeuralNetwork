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

/**
 * @author Jhony Luiz de Almeida
 */
public class PokerNet2 {
    
    
    
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
        
        int inputSize = (Naipe.QUANTIDADE + Carta.QUANTIDADE) * 5;
        DataSet trainingSet = new DataSet(inputSize, 9);
        File file = new File("poker.txt");
        try(
            BufferedReader reader = new BufferedReader(new FileReader(file))
        ) {
            for(String s = reader.readLine(); s != null; s = reader.readLine()) {
                //Pega valores da string separados por vírgula
                String[] values = s.split(",");
                //Criação do array de entradas
                double[] inputs = new double[inputSize];
                int cont = 0;
                for(int i = 0; i < values.length - 1; i++) {
                    inputs[cont + Integer.parseInt(values[i]) - 1] = 1.0;
                    cont += i % 2 == 0 ? Naipe.QUANTIDADE : Carta.QUANTIDADE;
                }
                //Criação do array de saídas
                double[] output = new double[9]; 
                int activation = Integer.parseInt(values[values.length - 1]);
                if(activation > 0) { output[activation - 1] = 1.0; }
                //System.out.println(Arrays.toString(inputs) + "  - " + Arrays.toString(output));
                //Adição do registro de treinamento
                trainingSet.addRow(new DataSetRow(inputs, output));
            }
        } catch(IOException ex) {
            
        }

        MomentumBackpropagation backPropagation = new MomentumBackpropagation();
        backPropagation.setMomentum(0.3);
        backPropagation.setMaxIterations(5000);
        backPropagation.setMaxError(0.01);
        backPropagation.setLearningRate(0.1);
        
        MultiLayerPerceptron perceptron = new MultiLayerPerceptron(TransferFunctionType.SIGMOID, 85, 20, 9);
        perceptron.setLearningRule(backPropagation);
        perceptron.learn(trainingSet);
        
        perceptron.save("pokerNet5.nnet");
        System.out.println("saved");
        
        
        //Utilizar a rede depois de salva
        //NeuralNetwork perceptron = MultiLayerPerceptron.createFromFile("pokerNet5.nnet"); 
        perceptron.setInput(
            0, 0, 0, 1,
            1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 1, 0,
            1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 1,
            0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 1,
            0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 1, 0, 0,
            1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
        );
        perceptron.calculate();
        System.out.println(" Output: " + Arrays.toString(perceptron.getOutput()));
    }
    
}
