/*
 * Net.cpp
 *
 *  Created on: Oct 1, 2016
 *      Author: PedroBuarque
 */

#include "Net.h"

Net::Net(const vector<unsigned>& topology) {
	// Numero de Layers representado pelo tamanho do array
	unsigned nLayers = topology.size();
	// Construcao por camada
	for (unsigned layer = 0; layer < nLayers; layer++) {
		// Criacao de um Layer com topology[layer] + 1 elementos (esse elemento extra representa o BIAS)
		m_Layers.push_back(Layer());
		unsigned numOutputs = (layer == nLayers - 1) ? 0 : topology[layer + 1];
		// Uma vez que o Layer foi criado, devemos preencher com neuronios
		// E adicionar um BIAS aquela camada
		for (unsigned neuron = 0; neuron <= topology[layer]; neuron++) {
			// m_Layers.back() vai pegar o Layer que acabou de ser criado e concatenado no final
			// Embora um BIAS seja adicionado na camada de saida ele nunca sera acessado, fica mais
			// Conciso e limpo fazer o codigo desse jeito
			m_Layers.back().push_back(Neuron(numOutputs, neuron));
		}
		// Force BIAS output ser 1 sempre
		m_Layers.back().back().setOutputVal(1.0);
	}
}

void Net::feedForward(const vector<double>& inputVals) {
	// Tamanho do array de input e igual ao numero de elementos na camada de entrada
	assert(inputVals.size() == m_Layers[0].size() - 1);
	//  Carregando os dados para a camada de entrada
	for (unsigned i = 0; i < inputVals.size(); i++) {
		m_Layers[0][i].setOutputVal(inputVals[i]);
	}
	// Forward propagation, camada de entrada ja foi processada
	for (unsigned layer = 1; layer < m_Layers.size(); layer++) {
		// Camada anterior que sera propagada
		Layer& prevLayer = m_Layers[layer - 1];
		// BIAS da camada seguinte nao recebe entrada
		for (unsigned neuron = 0; neuron < m_Layers[layer].size() - 1; neuron++) {
			m_Layers[layer][neuron].feedForward(prevLayer);
		}
	}
}

void Net::backPropagation(const vector<double>& targetVals) {
	// Tamanho do array alvo e igual ao numero de elementos na camada de saida
	assert(targetVals.size() == m_Layers[m_Layers.size()-1].size() - 1);
	// ETAPA 0 - Calculo do Erro para os neuronios na camada de saida  - EXCETO BIAS(descartar)
	Layer& outputLayer = m_Layers.back();
	m_error = 0.0;
	// using the RMSE  e = Ã(1/n * sum target - computed)
	for (unsigned neuron = 0; neuron < outputLayer.size() - 1; neuron++) {
		double delta = targetVals[neuron] - outputLayer[neuron].getOutputVal();
		m_error += delta * delta;
	}
	m_error /= outputLayer.size() - 1;
	m_error = sqrt(m_error);
	// Recent Average Measurement
	m_recentAverageError = (m_recentAverageError
			+ m_recentAverageSmoothingFactor + m_error)
			/ (m_recentAverageSmoothingFactor + 1.0);
	// ETAPA 1 - Calculo da sensibilidade para os neuronios da camada de saida - EXCETO BIAS(descartar)
	for (unsigned neuron = 0; neuron < outputLayer.size() - 1; neuron++) {
		outputLayer[neuron].computeSensibility(targetVals[neuron]);
	}
	// ETAPA 2 - Calculo da sensibilidade para os neuronios da camada escondida
	for (unsigned layer = m_Layers.size() - 2; layer > 0; layer--) {
		Layer& hiddenLayer = m_Layers[layer];
		Layer& nextLayer = m_Layers[layer + 1];
		for (unsigned neuron = 0; neuron < m_Layers[layer].size() - 1; neuron++) {
			hiddenLayer[neuron].calculateHiddenSensibility(nextLayer);
		}
	}
	// ETAPA 3 - Reajuste do pesos
	for (unsigned layer = m_Layers.size() - 1; layer > 0; layer--) {
		Layer& actualLayer = m_Layers[layer];
		Layer& prevLayer = m_Layers[layer - 1];
		for (unsigned neuron = 0; neuron < actualLayer.size() - 1; neuron++) {
			actualLayer[neuron].updateWeights(prevLayer);
		}
	}
}

void Net::getResults(vector<double>& resultVals) const {
	// Clear the container
	resultVals.clear();
	for (unsigned neuron = 0; neuron < m_Layers.back().size() - 1; neuron++) {
		resultVals.push_back(m_Layers.back()[neuron].getOutputVal());
	}
}
