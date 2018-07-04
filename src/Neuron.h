/*
 * Neuron.h
 *
 *  Created on: Oct 2, 2016
 *      Author: PedroBuarque
 */

#ifndef NEURON_H_
#define NEURON_H_

#include <vector>
#include <iostream>
#include <cmath>

using namespace std;
class Neuron;

// Definindo o tipo auxiliar para representar uma camada inteira de neuronios
typedef vector<Neuron> Layer;

// Estrutura de conexao entre neuronio e os outros da prox camada
struct Connection {
	double weight;
	double delta;
};

class Neuron {

public:
	// Static Memebers
	static double activationFunction(double net);				// f(net)  = 1 / 1 + e^(-net)
	static double activationFunctionDerivative(double net);		// f'(net) = f(net)*(1 - f(net))
	// Instance Memebers
	Neuron(unsigned numOutputs, unsigned myIndex);
	void feedForward(const Layer& prevLayer);
	void setOutputVal(double val) {this->outputVal = val;}
	double getOutputVal(void) const {return this->outputVal;}
	void computeSensibility(double targetVal);
	void calculateHiddenSensibility(const Layer& nextLayer);
	void updateWeights(Layer& prevLayer);
	double sumDOW(const Layer& nextLayer) const;

private:
	// Static Memebers
	static double randomWeight(void) {return rand()/double(RAND_MAX); } // Gerador de numeros aleatorios entre [0,1]
	static double eta;
	static double alpha;
	// Instance Memebers
	unsigned myIndex;						// A posicao desse neuronio na camada dele
	double outputVal;						// Valor de saida do neuronio
	double sensibility;
	vector<Connection> outputWeigths;		// pesos que ligam esse neuronio a os outros da prox camada

};

#endif
