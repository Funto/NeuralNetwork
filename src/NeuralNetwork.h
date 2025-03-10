#pragma once

struct LabeledImage;

struct Layer
{
	int					nbInputs=0;
	int					nbOutputs=0;	// Note: nbOutputs == nbNeurons
	std::vector<float>	weightsAndBias;

	// Temporary values: neuron outputs written during last feedForward() call
	std::vector<float>	neuronValues;	// size == nbOutputs == nbNeurons
	std::vector<float>	zValues;		// size == nbOutputs == nbNeurons

	// Temporary values: written during last back propagation
	// - List of sum of partial derivatives of Cost function for last backprop'ed image. Same size as weightAndBias.
	//   => [dCost(img)/dWeight0, dCost(img)/dWeight1 ..., dCost(img)/dBias]
	std::vector<float>	backpropSumOfWeightsAndBiasCostPartialDerivative;

	// - List of Delta values issued from chain rule. Used to scale previous neurons influence. Size = number of neurons = nbOutputs.
	std::vector<float>	backpropDelta;

	void initRandom(int nbInputValues, int nbOutputValues)
	{
		nbInputs = nbInputValues;
		nbOutputs = nbOutputValues;
		weightsAndBias.resize((nbInputs+1)*nbOutputs);
		for(float& f : weightsAndBias)
			f = randNormal();
		resizeTemporaryArraysOnInit();
	}

	void saveToFile(FILE* f)
	{
		fwrite(&nbInputs, sizeof(nbInputs), 1, f);
		fwrite(&nbOutputs, sizeof(nbOutputs), 1, f);
		fwrite(weightsAndBias.data(), sizeof(weightsAndBias[0]), weightsAndBias.size(), f);
	}

	void readFromFile(FILE* f)
	{
		fread(&nbInputs, sizeof(nbInputs), 1, f);
		fread(&nbOutputs, sizeof(nbOutputs), 1, f);
		weightsAndBias.resize((nbInputs+1)*nbOutputs);
		fread(weightsAndBias.data(), sizeof(weightsAndBias[0]), weightsAndBias.size(), f);
		resizeTemporaryArraysOnInit();
	}

	void debugPrintNeuronValues(const char* message)
	{
		printf("Neuron values %s:\n", message);
		for(int i=0 ; i < (int)neuronValues.size() ; i++)
			printf("[%d]: %.6f\n", i, neuronValues[i]);
	}

	void feedForward(const float* inData, int nbInputValues, bool bDebugPrint, const char* message);
	void feedForward(const Layer& prevLayer, bool bDebugPrint, const char* message)
	{
		feedForward(prevLayer.neuronValues.data(), (int)prevLayer.neuronValues.size(), bDebugPrint, message);
	}

	void resetBackpropCostGradient();
	void computeBackpropagationValues(const Layer& nextLayer, const float* prevLayerActivations, int nbPrevLayerActivations);
	void computeBackpropagationValuesForLastLayer(float* expectedOutput, int nbExpectedOutputValues, const float* prevLayerActivations, int nbPrevLayerActivations);

private:
	void resizeTemporaryArraysOnInit()
	{
		neuronValues.resize(nbOutputs);
		zValues.resize(nbOutputs);
		backpropSumOfWeightsAndBiasCostPartialDerivative.resize(weightsAndBias.size());
		backpropDelta.resize(nbOutputs);
	}
};

struct NeuralNetwork
{
	Layer layers[3];

	void	initRandom();
	bool	initFromFile(const char* fileName);
	bool	saveToFile(const char* fileName);

	void	resetBackpropCostGradient()
	{
		for(Layer& layer : layers)
			layer.resetBackpropCostGradient();
	}

	void	feedForward(const LabeledImage& img, bool bDebugPrint);
	void	backPropagateImages(const std::vector<const LabeledImage*>& images, std::vector<std::vector<float>>& outCostGradient);
	void	addToWeightAndBiases(const std::vector<std::vector<float>> weightAndBiasesCorrectionPerLayer);
	float	computeCost(const std::vector<LabeledImage>& images);
	//void	computeLabeledImageCostDerivative(const LabeledImage& img, std::vector<float> outDCostPerWeightAndBias[2]);
};
