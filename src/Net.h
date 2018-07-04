/*
 * Net.h
 *
 *  Created on: Oct 1, 2016
 *      Author: PedroBuarque
 */

#ifndef NET_H_
#define NET_H_

#include <vector>
#include <iostream>
#include <assert.h>
#include "Neuron.h"

using namespace std;

// Definindo o tipo auxiliar para representar uma camada inteira de neuronios
typedef vector<Neuron> Layer;

class Net {

public:
    Net(const vector<unsigned>& topology);
    void feedForward(const vector<double>& inputVals);			// 1 ETAPA - Propagacao do sinal de entrada
    void backPropagation(const vector<double>& outputVals);		// 2 ETAPA - Retropropagacao do erro e reajuste dos pesos
    void getResults(vector<double>& resultVals) const;			// Computando a saida finae copiado para o parametro
    double getError(void) const {return m_error;}
    double getRecentAverageError(void) const {return m_recentAverageError;}

private:
    // Vetor com todas as camadas podendo acessar os neuronios por camada
    vector<Layer> m_Layers;
    double m_error;
    double m_recentAverageError;
    double m_recentAverageSmoothingFactor;
};

#endif
