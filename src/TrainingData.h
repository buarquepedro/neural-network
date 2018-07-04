/*
 * TrainingData.h
 *
 *  Created on: Oct 8, 2016
 *      Author: PedroBuarque
 */

#ifndef TRAININGDATA_H_
#define TRAININGDATA_H_

#include <iostream>
#include <vector>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>

using namespace std;
typedef vector<double> t_vals;

class TrainingData
{
public:
    TrainingData(const string filename);
    bool isEof(void) { return m_trainingDataFile.eof(); }
    void getTopology(vector<unsigned> &topology);

    // Returns the number of input values read from the file:
    unsigned getNextInputs(t_vals &inputVals);
    unsigned getTargetOutputs(t_vals &targetOutputVals);

private:
    std::ifstream m_trainingDataFile;
};

#endif /* TRAININGDATA_H_ */
