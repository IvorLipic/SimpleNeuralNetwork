package ui;
import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.nio.file.Path;
import java.util.*;

public class Solution {
	public static Random random = new Random();
	public static Path trainSetCSV;
	public static Path testSetCSV;
	public static int[] architecture;
	public static int popSize;
	public static int elitism;
	public static double probabilityOfMutation;
	public static double K;
	public static int iterations;
	public static ArrayList<String> features = new ArrayList<>();
	public static ArrayList<ArrayList<Double>> trainSet = new ArrayList<>();
	public static ArrayList<ArrayList<Double>> testSet = new ArrayList<>();

	public static class GeneticAlgorithm{
		ArrayList<NeuralNetwork> population = new ArrayList<>(popSize);
		public GeneticAlgorithm() {
			/*
			Stvori inicijalnu populaciju i evaluiraj (sortiraj po fitnessu silazno)
			*/
			for(int i = 0; i < popSize; i++){
				NeuralNetwork nn = new NeuralNetwork();
				nn.generateWeightsAndThresholds();
				nn.setTrainError();
				population.add(nn);
			}
			population.sort(Comparator.comparing(NeuralNetwork::getFitness).reversed());
			int x = 0;
			while (iterations != x){
				/*
				U svakoj iteraciji:
					1. Ostavi onoliko najboljih iz trenutne populacije koliko je definirano elitizmom
					2. Izaberi 2 roditelja, krizaj ih i mutiraj i stvori 2 djeteta, dodaj ih u novu populaciju i ponavljaj dok populacija nije odredene velicine
					3. Izračunaj error i sortiraj novu populaciju po dobroti
				 */
				x++;
				ArrayList<NeuralNetwork> newPopulation = new ArrayList<>();
				for(int i = 0; i < elitism; i++) newPopulation.add(population.get(i)); //1.
				NeuralNetwork parent1;
				NeuralNetwork parent2;
				while(newPopulation.size() < popSize){ //2.
					parent1 = rouletteWheelSelection(population);
					if(parent1 == null){
						System.exit(0);
					}
					population.remove(parent1);
					parent2 = rouletteWheelSelection(population);
					if(parent2 == null){
						System.exit(0);
					}
					population.add(parent1);
					newPopulation.addAll(crossAndMutate(parent1, parent2));
				}
				population = newPopulation;
				for(int i = 0; i < popSize; i++){//3.
					population.get(i).setTrainError();
				}
				population.sort(Comparator.comparing(NeuralNetwork::getFitness).reversed());
				if(x % 2000 == 0){
					System.out.println("[Train error @" + x + "]: " + population.get(0).getTrainError());
				}
			}
			population.get(0).setTestError();
			System.out.println("[Test error]: " + population.get(0).getTestError());
		}

		private NeuralNetwork rouletteWheelSelection(ArrayList<NeuralNetwork> population) {
			double totalFitness = 0.0;
			for(NeuralNetwork nn : population){
				totalFitness += nn.getFitness();
			}
			double randomValue = totalFitness * Math.random(); //Random vrijednost izmedu 0 i zbroja svih fitnessa
			double addedFitness = 0.0;
			for(NeuralNetwork nn : population){ //Zbrajaj fitnesse i kad prijedes random vrijednost vrati jedinku koja ju je pregazila
				addedFitness += nn.getFitness();
				if(addedFitness >= randomValue) return nn;
			}
			return null;
		}
		private ArrayList<NeuralNetwork> crossAndMutate(NeuralNetwork nn1, NeuralNetwork nn2) {
			//Krizanje
			/*
			Uzmi tezine i pragove roditelja i izracunaj njihovu aritmeticku sredinu
			 */
			double[][][] weights1 = nn1.getWeights();
			double[][][] weights2 = nn2.getWeights();
			double[][] threshold1 = nn1.getThreshold();
			double[][] threshold2 = nn2.getThreshold();

			double[][][] newWeights = new double[weights1.length][][];
			double[][] newThreshold = new double[threshold1.length][];

			for(int i = 0; i < weights1.length; i++){
				newWeights[i] = new double[weights1[i].length][];
				for(int j = 0; j < weights1[i].length; j++){
					newWeights[i][j] = new double[weights1[i][j].length];
					for(int k = 0; k < weights1[i][j].length; k++){
						newWeights[i][j][k] = (weights1[i][j][k] + weights2[i][j][k])/2; //Aritmeticka sredina
					}
				}
			}
			for(int i = 0; i < threshold1.length; i++){
				newThreshold[i] = new double[threshold1[i].length];
				for(int j = 0; j < threshold1[i].length; j++){
					newThreshold[i][j] = (threshold1[i][j] + threshold2[i][j])/2;
				}
			}
			//Mutacija
			/*
			Uzmi novi tensor tezina i novu matricu pragova, 2 puta mutiraj i tako stvori 2 djeteta
			 */
			ArrayList<NeuralNetwork> children = new ArrayList<>();
			for(int y = 0; y < 2; y++) {
				NeuralNetwork d = new NeuralNetwork();
				double[][][] mutatedWeights = new double[weights1.length][][];
				for (int i = 0; i < weights1.length; i++) {
					mutatedWeights[i] = new double[weights1[i].length][];
					for (int j = 0; j < weights1[i].length; j++) {
						mutatedWeights[i][j] = new double[weights1[i][j].length];
						for (int k = 0; k < weights1[i][j].length; k++) {
							if (random.nextDouble() <= probabilityOfMutation)
								//Pribroji tezini uzorak iz normalne razdiobe sa standardnom devijacijom K
								mutatedWeights[i][j][k] = newWeights[i][j][k] + random.nextGaussian() * K;
							else
								mutatedWeights[i][j][k] = newWeights[i][j][k];

						}
					}
				}
				double[][] mutatedThresholds = new double[threshold1.length][];
				for (int i = 0; i < threshold1.length; i++) {
					mutatedThresholds[i] = new double[threshold1[i].length];
					for (int j = 0; j < threshold1[i].length; j++) {
						if(random.nextDouble() <= probabilityOfMutation)
							mutatedThresholds[i][j] = newThreshold[i][j] + random.nextGaussian() * K;
						else
							mutatedThresholds[i][j] = newThreshold[i][j];

					}
				}
				d.setThreshold(mutatedThresholds);
				d.setWeights(mutatedWeights);
				d.setTrainError();
				children.add(d);
			}
			return children;
		}

		public static class NeuralNetwork{
			private static final int inputSize = features.size() - 1; //velicina ulaza (placeholder neurona)
			public double[][][] weights = new double[architecture.length + 1][][];
			public double[][] threshold = new double[architecture.length + 1][];
			public double testError;
			public double trainError;
			public double fitness;
			public double getTestError() {
				return this.testError;
			}
			public double getTrainError() {
				return this.trainError;
			}
			public double[][][] getWeights() {
				return this.weights;
			}
			public void setWeights(double[][][] weights) {
				this.weights = weights;
			}
			public double[][] getThreshold() {
				return this.threshold;
			}
			public void setThreshold(double[][] threshold) {
				this.threshold = threshold;
			}
			public double getFitness() {
				return fitness;
			}
			public void setFitness(){
				this.fitness = 1/this.trainError;
			}
			public NeuralNetwork(){}
			public void generateWeightsAndThresholds(){
				for(int i = 0; i < architecture.length + 1; i++){
				/*
				int[] architecture ima spremljene velicine SAMO SKRIVENIH slojeva,
				radi toga velicinu iduceg sloja mogu dohvatiti sa architecture[i],
				a velicinu trenutnog sloja sa architecture[i-1].

				*/
				// Postavljanje trenutnog/prijasnjeg i iduceg sloja (za određeni sloj tezina)
					int sizeOfCurrentLayer;
					if(i == 0) sizeOfCurrentLayer = inputSize;
					else sizeOfCurrentLayer = architecture[i - 1];

					int sizeOfNextLayer;
					if(i == architecture.length) sizeOfNextLayer = 1;
					else sizeOfNextLayer = architecture[i];
				/*
				Za tezine izmedu 2 odredena sloja imamo pripadnu matricu tezina cije su dimanzije odredene
				broju neurona u ta 2 sloja.
				Pravimo ju tako da za svaki neuron u trenutnom sloju izgeneriramo random tezine prema svim
				neuronima u iducem sloju.

				Pragove generiramo za sve neurone u iducem sloju (jer ih ne generiramo u prvom trenutnom - ulazu).
				*/
					weights[i] = new double[sizeOfCurrentLayer][sizeOfNextLayer];
					threshold[i] = new double[sizeOfNextLayer];

					//Popuni sve tezine i pragove random po norm. razdiobi s devijacijom 0.01
					for(int k = 0; k < sizeOfNextLayer; k++) threshold[i][k] = random.nextGaussian() * 0.01;
					for(int j = 0; j < sizeOfCurrentLayer; j++){
						for(int k = 0; k < sizeOfNextLayer; k++){
							weights[i][j][k] = random.nextGaussian() * 0.01;
						}
					}
				}
			}
			public void setTrainError() {
				this.trainError = calculateMeanSquaredError(trainSet);
				setFitness();
			}
			public void setTestError() {
				this.testError = calculateMeanSquaredError(testSet);
			}
			private double calculateMeanSquaredError(ArrayList<ArrayList<Double>> set){
				//Srednje kvadratno istupanje - err = 1/N * sum(for every s in set)((ys - NN(s))^2), N - broj uzoraka
				double MSE = 0.0;
				for(ArrayList<Double> sample : set){
					double expectedValue = sample.get(sample.size() - 1);
					ArrayList<Double> input = new ArrayList<>(sample);
					input.remove(sample.size() - 1);
					double squaredError = Math.pow(expectedValue - propagate(input),2);
					MSE += squaredError;
				}
				return MSE / set.size();
			}
			private double propagate(ArrayList<Double> input){ //Unaprijedni prolaz
				ArrayList<Double> currentLayerOutput = new ArrayList<>(input); //Propusti ciste podatke kao input trenutnog/prijasnjeg sloja (inicijalno za 1. sloj)
				for(int i = 0; i < architecture.length + 1; i++){

					int sizeOfCurrentLayer;
					if(i == 0) sizeOfCurrentLayer = input.size();
					else sizeOfCurrentLayer = architecture[i - 1];

					int sizeOfNextLayer;
					if(i == architecture.length) sizeOfNextLayer = 1;
					else sizeOfNextLayer = architecture[i];

					ArrayList<Double> nextLayerOutput = new ArrayList<>();
				/*
				Za svaki neuron u sljedećem sloju izračunaj net.
				net[i][j] = sum{for each neuron k in current layer}(sigmoid(net)[i-1][k] * weight[i][k][j]) + threshold[i][j]
				*/
					for(int j = 0; j < sizeOfNextLayer; j++){
						double net = threshold[i][j];
						for(int k = 0; k < sizeOfCurrentLayer; k++) net += currentLayerOutput.get(k) * weights[i][k][j];
						if(sizeOfNextLayer != 1) nextLayerOutput.add(sigmoid(net)); //Sigmoidu ne primjenjujemo na izlazni sloj
						else nextLayerOutput.add(net);
					}
					currentLayerOutput = nextLayerOutput; //U idoucoj iteraciji samo si postavimo izlaz bivseg iduceg sloja kao novog trenutnog
				}
				return currentLayerOutput.get(0); //Izlazni sloj ima 1 neuron pa vracamo jednu vrijednost
			}
			private double sigmoid(double x) {
				return 1 / (1 + Math.exp(-x));
			} //Sigmoidalna funkcija f(x) = 1/(1 + e^-x)
		}
	}
	public static void setParameters(String ... args){
		for(int i = 0; i < args.length - 1; i++){
			switch (args[i]) {
				case "--train" -> trainSetCSV = Path.of(args[i + 1]);
				case "--test" -> testSetCSV = Path.of(args[i + 1]);
				case "--nn" -> {
					String[] buffer = args[i + 1].split("s");
					architecture = new int[buffer.length];
					for (int j = 0; j < buffer.length; j++) architecture[j] = Integer.parseInt(buffer[j]);
				}
				case "--popsize" -> popSize = Integer.parseInt(args[i + 1]);
				case "--elitism" -> elitism = Integer.parseInt(args[i + 1]);
				case "--p" -> probabilityOfMutation = Double.parseDouble(args[i + 1]);
				case "--K" -> K = Double.parseDouble(args[i + 1]);
				case "--iter" -> iterations = Integer.parseInt(args[i + 1]);
			}
		}
	}
	public static void loadData() throws IOException {
		BufferedReader reader = new BufferedReader(new FileReader(trainSetCSV.toFile()));
		String line = reader.readLine();
		Collections.addAll(features, line.split(","));
		line = reader.readLine();
		while(line != null){
			ArrayList<String> buffer = new ArrayList<>();
			Collections.addAll(buffer, line.split(","));
			ArrayList<Double> buffer2 = new ArrayList<>();
			for(String a : buffer){
				buffer2.add(Double.valueOf(a));
			}
			trainSet.add(buffer2);
			line = reader.readLine();
		}
		reader = new BufferedReader(new FileReader(testSetCSV.toFile()));
		line = reader.readLine();
		line = reader.readLine();
		while(line != null){
			ArrayList<String> buffer = new ArrayList<>();
			Collections.addAll(buffer, line.split(","));
			ArrayList<Double> buffer2 = new ArrayList<>();
			for(String a : buffer){
				buffer2.add(Double.valueOf(a));
			}
			testSet.add(buffer2);
			line = reader.readLine();
		}
	}
	public static void main(String ... args) throws IOException {
		setParameters(args);
		loadData();
		GeneticAlgorithm ga = new GeneticAlgorithm();
	}
}
