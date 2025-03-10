#include "NeuralNetwork.h"

#define _USE_SIGMOID	// sigmoid or ReLU?

static float activationFunc(float x)
{
#ifdef _USE_SIGMOID
	return 1.f / (1.f + expf(-x));	// logistic function
#else
	return x > 0.f ? x : 0.f;	// ReLU
#endif
}

static float dActivationFunc(float x)
{
#ifdef _USE_SIGMOID
	// from https://en.wikipedia.org/wiki/Logistic_function
	const float fx = activationFunc(x);
	return fx * (1.f-fx);
#else
	assert(false && "not implemented");
#endif
}

//template<int N>
//void softMax(const float* inData, float* outData)
//{
//	double expData[N];
//	double sumExpData = 0.;
//	for(int i=0 ; i < N ; i++)
//	{
//		expData[i] = exp((double)inData[i]);
//		sumExpData += expData[i];
//	}
//
//	for(int i=0 ; i < N ; i++)
//		outData[i] = (float)(expData[i] / sumExpData);
//}

void Layer::feedForward(const float* inData, int nbInputValues, bool bDebugPrint, const char* message)
{
	assert(nbInputs == nbInputValues);

	const int neuronSize = nbInputs + 1;	// number of weights + 1 for the bias
	const int nbNeurons = nbOutputs;
	for(int idxNeuron = 0 ; idxNeuron < nbNeurons ; idxNeuron++)
	{
		const int neuronOffset = idxNeuron * neuronSize;
		float z = 0.f;
		for(int idxInput=0 ; idxInput < nbInputs ; idxInput++)
			z += weightsAndBias[neuronOffset + idxInput] * inData[idxInput];
		z += weightsAndBias[neuronOffset + neuronSize - 1];	// bias

		zValues[idxNeuron] = z;
		const float output = activationFunc(z);
		neuronValues[idxNeuron] = output;
	}
	if(bDebugPrint)
		debugPrintNeuronValues(message);
}

void Layer::resetBackpropCostGradient()
{
	const size_t size = backpropSumOfWeightsAndBiasCostPartialDerivative.size() * sizeof(backpropSumOfWeightsAndBiasCostPartialDerivative[0]);
	memset(backpropSumOfWeightsAndBiasCostPartialDerivative.data(), 0, size);
}

void Layer::computeBackpropagationValues(const Layer& nextLayer, const float* prevLayerActivations, int nbPrevLayerActivations)
{
	assert(nextLayer.nbInputs == nbOutputs);
	assert(nbPrevLayerActivations == nbInputs);

	const int neuronSize = nbInputs + 1;	// number of weights + 1 for the bias
	const int nbNeurons = nbOutputs;
	const int nextLayerNeuronSize = nextLayer.nbOutputs + 1;	// number of weights + 1 for the bias
	for(int idxNeuron=0 ; idxNeuron < nbOutputs ; idxNeuron++)
	{
		float delta = 0.f;
		for(int idxNextLayerOutput=0 ; idxNextLayerOutput < nextLayer.nbOutputs ; idxNextLayerOutput++)
		{
			const int nextLayerNeuronOffset = idxNextLayerOutput * nextLayerNeuronSize;
			const float nextLayerWeight = nextLayer.weightsAndBias[nextLayerNeuronOffset + idxNeuron];	// idxNeuron is in [0;nextLayer.nbInputs-1]
			const float nextLayerDelta = nextLayer.backpropDelta[idxNextLayerOutput];	// TODO: can probably avoid re-reading same value every time?
			delta += nextLayerDelta * nextLayerWeight;
		}

		const float sigmaPrime = dActivationFunc(zValues[idxNeuron]);
		delta *= sigmaPrime;
		backpropDelta[idxNeuron] = delta;

		const int neuronOffset = idxNeuron * neuronSize;
		for(int idxInput=0 ; idxInput < nbInputs ; idxInput++)
		{
			const float prevLayerActivation = prevLayerActivations[idxInput];
			backpropSumOfWeightsAndBiasCostPartialDerivative[neuronOffset + idxInput] += delta * prevLayerActivation;
		}
		backpropSumOfWeightsAndBiasCostPartialDerivative[neuronOffset + neuronSize - 1] += delta;
	}
}

void Layer::computeBackpropagationValuesForLastLayer(float* expectedOutput, int nbExpectedOutputValues, const float* prevLayerActivations, int nbPrevLayerActivations)
{
	assert(nbExpectedOutputValues == nbOutputs);
	assert(nbPrevLayerActivations == nbInputs);

	const int neuronSize = nbInputs + 1;	// number of weights + 1 for the bias
	const int nbNeurons = nbOutputs;
	for(int idxNeuron=0 ; idxNeuron < nbOutputs ; idxNeuron++)
	{
		const int neuronOffset = idxNeuron * neuronSize;
		const float dCostRelativeToActivation = 2.f * (neuronValues[idxNeuron] - expectedOutput[idxNeuron]);
		const float delta = dCostRelativeToActivation * dActivationFunc(zValues[idxNeuron]);
		backpropDelta[idxNeuron] = delta;
		for(int idxInput=0 ; idxInput < nbInputs ; idxInput++)
		{
			const float prevLayerActivation = prevLayerActivations[idxInput];
			backpropSumOfWeightsAndBiasCostPartialDerivative[neuronOffset + idxInput] += delta * prevLayerActivation;
		}
		backpropSumOfWeightsAndBiasCostPartialDerivative[neuronOffset + neuronSize - 1] += delta;
	}
}

void NeuralNetwork::initRandom()
{
	// https://www.youtube.com/watch?v=aircAruvnKk&t=262s
	// Architecture:
	// - layer 0: 28*28 = 784 outputs, 16 outputs
	// - layer 1: 16 inputs, 16 outputs
	// - layer 2: 16 inputs, 10 outputs
	
	layers[0].initRandom(IMG_SX*IMG_SY, 16);
	layers[1].initRandom(16, 16);
	layers[2].initRandom(16, 10);

	//layers[0].initRandom(IMG_SX*IMG_SY, 32);
	//layers[1].initRandom(32, 32);
	//layers[2].initRandom(32, 10);
}

bool NeuralNetwork::initFromFile(const char* fileName)
{
	FILE* f = fopen(fileName, "rb");
	if(!f)
	{
		fprintf(stderr, "Failed to init neural network from file: %s\n", fileName);
		return false;
	}
	for(Layer& layer : layers)
		layer.readFromFile(f);
	fclose(f);
	return true;
}

bool NeuralNetwork::saveToFile(const char* fileName)
{
	FILE* f = fopen(fileName, "wb");
	if(!f)
	{
		fprintf(stderr, "Failed to save neural network to file: %s\n", fileName);
		return false;
	}
	for(Layer& layer : layers)
		layer.saveToFile(f);
	fclose(f);
	return true;
}

void NeuralNetwork::feedForward(const LabeledImage& img, bool bDebugPrint)
{
	// layers[0] <- img
	layers[0].feedForward(img.floatData, IMG_SX*IMG_SY, bDebugPrint, "layer 0");

	// layers[1] <- layers[0]
	layers[1].feedForward(layers[0], bDebugPrint, "layer 1");

	// layers[2] <- layers[1]
	layers[2].feedForward(layers[1], bDebugPrint, "layer 2");
}

void NeuralNetwork::backPropagateImages(const std::vector<const LabeledImage*>& images, std::vector<std::vector<float>>& outCostGradient)
{
	resetBackpropCostGradient();

	for(const LabeledImage* pImage : images)
	{
		const LabeledImage& img = *pImage;

		//feedForward(img, true);
		feedForward(img, false);

		float expectedOutput[10] = {0};
		expectedOutput[img.label] = 1.f;

		Layer& lastLayer = layers[_countof(layers)-1];
		Layer& prevToLastLayer = layers[_countof(layers)-2];
		lastLayer.computeBackpropagationValuesForLastLayer(expectedOutput, _countof(expectedOutput), prevToLastLayer.neuronValues.data(), (int)prevToLastLayer.neuronValues.size());

		// Compute layer idxLayer with next layer (idxLayer+1) as input
		for(int idxLayer = _countof(layers)-2 ; idxLayer >= 0 ; idxLayer--)
		{
			const Layer& nextLayer = layers[idxLayer+1];
			const float* prevLayerActivations = nullptr;
			int nbPrevLayerActivations = 0;
			if(idxLayer == 0)
			{
				prevLayerActivations = img.floatData;
				nbPrevLayerActivations = IMG_SX*IMG_SY;
			}
			else
			{
				prevLayerActivations = layers[idxLayer-1].neuronValues.data();
				nbPrevLayerActivations = (int)layers[idxLayer-1].neuronValues.size();
			}

			Layer& curLayer = layers[idxLayer];
			curLayer.computeBackpropagationValues(nextLayer, prevLayerActivations, nbPrevLayerActivations);
		}
	}

	// Now that we computed backpropSumOfWeightsAndBiasCostPartialDerivative[], divide by number of images in batch to compute the cost gradient
	if(images.size() > 1)
	{
		const float fInvBatchSize = 1.f / ((float)images.size());
		outCostGradient.reserve(_countof(layers));
		for(const Layer& layer : layers)
		{
			outCostGradient.push_back({});
			std::vector<float>& layerWeightsAndBiasCostPartialDerivative = outCostGradient.back();
			layerWeightsAndBiasCostPartialDerivative = layer.backpropSumOfWeightsAndBiasCostPartialDerivative;
			for(float& f : layerWeightsAndBiasCostPartialDerivative)
				f *= fInvBatchSize;
		}
	}
}

void NeuralNetwork::addToWeightAndBiases(const std::vector<std::vector<float>> weightAndBiasesCorrectionPerLayer)
{
	for(int idxLayer=0 ; idxLayer < _countof(layers) ; idxLayer++)
	{
		Layer& layer = layers[idxLayer];
		const std::vector<float>& weightAndBiasesCorrection = weightAndBiasesCorrectionPerLayer[idxLayer];

		assert(weightAndBiasesCorrection.size() == layer.weightsAndBias.size());
		for(int i=0 ; i < (int)layer.weightsAndBias.size() ; i++)
			layer.weightsAndBias[i] += weightAndBiasesCorrection[i];
	}
}

float NeuralNetwork::computeCost(const std::vector<LabeledImage>& images)
{
	double totalCost = 0.;

	Layer& lastLayer = layers[_countof(layers)-1];
	for(const LabeledImage& img : images)
	{
		feedForward(img, false);
		float imgCost = 0.f;
		for(int i=0 ; i < (int)lastLayer.neuronValues.size() ; i++)
		{
			const float diff = lastLayer.neuronValues[i] - (img.label == i ? 1.f : 0.f);
			imgCost += diff*diff;
		}
		totalCost += (double)imgCost;
	}

	totalCost /= (double)images.size();
	return (float)totalCost;
}

#if 0
// Compute partial derivative of cost relative to each weight and bias
void NeuralNetwork::computeLabeledImageCostDerivative(const LabeledImage& img, std::vector<float> outDCostPerWeightAndBias[2])
{
	assert(outDCostPerWeightAndBias[0].size() == layers[0].weightsAndBias.size());
	assert(outDCostPerWeightAndBias[1].size() == layers[1].weightsAndBias.size());

	/*
	* === Compute Cost partial derivatives for weights and bias in layer 1 ===
	* Per https://www.youtube.com/watch?v=tIeHLnjs5U8&t=224s
	* 
	* L = layer 1, j = neuron index in layer 1
	* 
	* z(L,j) = sum_j(Weight(L,j)*Activation(L-1,j)) + bias(L,j)
	* Activation(L,j) = activationFunc(z(L,j))
	* 
	* Cost = sum_j( (Activation(L,j) - y(j))^2 )
	* dCost/dActivation(L,j) = 2*(Activation(L,j) - y(j))
	* 
	* Chain rule:
	* dCost/dWeight(L,j) = dz(L,j)/dWeight(L,j)       *    dActivation(L,j)/dz(L,j)    *    dCost/dActivation(L,j)
	*                    = sum_k(Activation(L-1,k))   *    dActivationFunc(z(L,j))     *    2*(Activation(L,j) - y(j))
	* 
	* Per https://www.youtube.com/watch?v=tIeHLnjs5U8&t=6m1s
	* dCost/dBias(L,j) = dz(L,j)/dBias(L,j)    *    dActivation(L,j)/dz(L,j)    *    dCost/dActivation(L,j)
	*                  = 1                     *    dActivationFunc(z(L,j))     *    2*(Activation(L,j) - y(j))
	* 
	* === Compute Cost partial derivatives for weights and bias in layer 0 ===
	* 
	* Per https://www.youtube.com/watch?v=tIeHLnjs5U8&t=387s
	* TODO
	*/
}
#endif
