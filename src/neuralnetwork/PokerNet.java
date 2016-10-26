package neuralnetwork;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.util.Arrays;
import static neuralnetwork.PokerNet.Carta.*;
import static neuralnetwork.PokerNet.Naipe.*;
import org.neuroph.core.data.DataSet;
import org.neuroph.core.data.DataSetRow;
import org.neuroph.nnet.MultiLayerPerceptron;
import org.neuroph.nnet.learning.MomentumBackpropagation;
import org.neuroph.util.TransferFunctionType;

/**
 * @author Jhony Luiz de Almeida
 */
public class PokerNet {
    
    
    
    static interface Naipe {
        double ESCALA = 1.0 / 4.0;
        double OUROS = 1;
        double ESPADAS = 2; 
        double COPAS = 3;
        double PAUS = 4;
    }
    
    static interface Carta {
        double ESCALA = 1.0 / 13.0;
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
        
        DataSet trainingSet = new DataSet(10, 9);
        File file = new File("poker.txt");
        try(
            BufferedReader reader = new BufferedReader(new FileReader(file))
        ) {
            for(String s = reader.readLine(); s != null; s = reader.readLine()) {
                //Pega valores da string separados por vírgula
                String[] values = s.split(",");
                //Criação do array de entradas
                double[] inputs = new double[values.length - 1];
                for(int i = 0; i < inputs.length; i++) {
                    inputs[i] = Double.parseDouble(values[i]); // * (i % 2 == 0 ? Naipe.ESCALA : Carta.ESCALA);
                }
                //Criação do array de saídas
                double[] output = new double[9]; 
                int activation = Integer.parseInt(values[values.length - 1]);
                if(activation > 0) { output[activation - 1] = 1.0; }
                System.out.println(Arrays.toString(inputs) + "  - " + Arrays.toString(output));
                //Adição do registro de treinamento
                trainingSet.addRow(new DataSetRow(inputs, output));
            }
        } catch(IOException ex) {
            
        }

        //Definição do algoritmo de treinamento
        MomentumBackpropagation backPropagation = new MomentumBackpropagation();
        backPropagation.setMomentum(0.2);
        backPropagation.setMaxIterations(3000);
        backPropagation.setMaxError(0.01);
        backPropagation.setLearningRate(0.1);
        
        MultiLayerPerceptron perceptron = new MultiLayerPerceptron(TransferFunctionType.SIGMOID, 10, 7, 9);
        perceptron.setLearningRule(backPropagation);
        perceptron.learn(trainingSet);
        
        perceptron.save("pokerNet3.nnet");
        System.out.println("saved");
        
        //Utilizar a rede depois de salva
        //NeuralNetwork perceptron = MultiLayerPerceptron.createFromFile("pokerNet.nnet"); 
        perceptron.setInput(
            COPAS, DOIS,
            COPAS, AS,
            PAUS, DOIS,
            ESPADAS, DOIS,
            OUROS, DAMA
        );
        perceptron.calculate();
        System.out.println(" Output: " + Arrays.toString(perceptron.getOutput()));
    }
    
}
