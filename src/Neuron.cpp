/*
 * Neuron.cpp
 *
 *  Created on: Oct 2, 2016
 *      Author: PedroBuarque
 */

#include "Neuron.h"

double Neuron::eta   = 0.75;
double Neuron::alpha = 0.5;

Neuron::Neuron(unsigned numOutputs, unsigned myIndex) {
    // Para cada conexao existente entre esse neuronio e um pertencente a prox camada FACA
    for (unsigned c = 0; c < numOutputs; c++) {
        outputWeigths.push_back(Connection());
        // Inicia os pesos de forma aleatoria, com valores entre 0 e 1
        outputWeigths.back().weight = randomWeight();
    }
    // Copying the position of this neuron in his layer
    this->myIndex = myIndex;
}

double Neuron::activationFunction(double net) {
    // Funcao tangente Hiperbolica
    return tanh(net);
}

double Neuron::activationFunctionDerivative(double net) {
    // Derivada da funcao tangente hiperbolica
    return 1.0 - (net * net);
}

void Neuron::feedForward(const Layer& prevLayer) {
    // Entrada liquida
    double net = 0.0;
    // Para cada neuronio na camada anterior FACA (incluir o bias)
    for (unsigned neuron = 0; neuron < prevLayer.size(); neuron++) {
        net += prevLayer[neuron].getOutputVal()
        * prevLayer[neuron].outputWeigths[myIndex].weight;
    }
    // Calculo da funcao de saida
    this->outputVal = Neuron::activationFunction(net);
}

void Neuron::computeSensibility(double targetVal) {
    double delta = targetVal - outputVal;
    sensibility = delta * Neuron::activationFunctionDerivative(outputVal);
}

double Neuron::sumDOW(const Layer& nextLayer) const{
    double sum = 0.0;
    // Soma das contribuicoes dos neuronios de cada camada
    for (unsigned neuron = 0; neuron < nextLayer.size() - 1; neuron++) {
        sum += outputWeigths[neuron].weight * nextLayer[neuron].sensibility;
    }
    return sum;
}

void Neuron::calculateHiddenSensibility(const Layer& nextLayer) {
    double delta = sumDOW(nextLayer);
    sensibility = delta * Neuron::activationFunctionDerivative(outputVal);
}

void Neuron::updateWeights(Layer& prevLayer) {
    // Atualiza os pesos de cada conexao entre camadas
    for (unsigned neuron = 0; neuron < prevLayer.size(); neuron++) {
        Neuron& n = prevLayer[neuron];
        double oldDelta = n.outputWeigths[myIndex].delta;
        double newDelta = Neuron::eta * n.getOutputVal() * sensibility + Neuron::alpha * oldDelta;
        n.outputWeigths[myIndex].delta  = newDelta;
        n.outputWeigths[myIndex].weight += newDelta;
    }
}

