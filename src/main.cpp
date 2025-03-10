#include "NeuralNetwork.h"
#include "GUI.h"

//#pragma optimize("", off)

static void _debugTestImage(NeuralNetwork& nn, LabeledImage& img, int epoch)
{
	Layer& lastLayer = nn.layers[_countof(nn.layers)-1];
	nn.feedForward(img, false);
	int answer = 0;
	for(int i=0 ; i < 10 ; i++)
		answer = lastLayer.neuronValues[i] > lastLayer.neuronValues[answer] ? i : answer;
	printf("Epoch: %d (wanted result: %d): %.6f %.6f %.6f %.6f %.6f %.6f %.6f %.6f %.6f %.6f -> %d\n",
		epoch,
		(int)img.label,
		lastLayer.neuronValues[0], lastLayer.neuronValues[1], lastLayer.neuronValues[2], lastLayer.neuronValues[3], lastLayer.neuronValues[4],
		lastLayer.neuronValues[5], lastLayer.neuronValues[6], lastLayer.neuronValues[7], lastLayer.neuronValues[8], lastLayer.neuronValues[9],
		answer);
}

static void _extractDebugImages(const std::vector<LabeledImage>& trainingImages, const std::vector<LabeledImage>& testImages)
{
	// Debug training
	for(int i=0 ; i < 30 ; i++)
	{
		char strFileName[64];
		sprintf(strFileName, "debug/train_%03d_%d.pgm", i, trainingImages[i].label);
		trainingImages[i].save(strFileName);
	}

	// Debug test
	for(int i=0 ; i < 30 ; i++)
	{
		char strFileName[64];
		sprintf(strFileName, "debug/test_%03d_%d.pgm", i, testImages[i].label);
		testImages[i].save(strFileName);
	}
}

int main(int argc, char* argv[])
{
	gData.pGUI = std::make_unique<GUI>();	// Comment to disable GUI
	if(!gData.pGUI->init())
		return EXIT_FAILURE;

	readLabeledImages(TRAINING_IMAGES_FILENAME, TRAINING_LABELS_FILENAME, gData.trainingImages);
	readLabeledImages(TEST_IMAGES_FILENAME, TEST_LABELS_FILENAME, gData.testImages);

	//_extractDebugImages(gData.trainingImages, gData.testImages);

	gData.pNN = std::make_unique<NeuralNetwork>();
	//gData.pNN->initRandom();
	gData.pNN->initFromFile(DATA_DIR "/weightsAndBiases_30000.bin");
	
	if(gData.pGUI)
	{
		gData.pGUI->mainLoop();

		gData.pGUI->shut();
	}

	const int batchSize = 100;
	
#if 0
	// Debug test with 1 image
	{
		LabeledImage& img = trainingImages[0];

		std::vector<LabeledImage> imagesList = {img};
		std::vector<const LabeledImage*> imagesPtrList = {&img};
		float cost = 0.f;

		cost = nn.computeCost(imagesList);
		printf("Init: cost: %f\n", cost);
		_debugTestImage(nn, img, -1);

		for(int epoch=0 ; epoch < 10000 ; epoch++)
		{
			std::vector<std::vector<float>> weightAndBiasesCorrectionPerLayer;
			nn.backPropagateImages(imagesPtrList, weightAndBiasesCorrectionPerLayer);

			static float s_learningRate = 3.f;
			for(std::vector<float>& weightAndBiasesCorrection : weightAndBiasesCorrectionPerLayer)
				for(float& f : weightAndBiasesCorrection)
					f = -s_learningRate * f;
		
			nn.addToWeightAndBiases(weightAndBiasesCorrectionPerLayer);

			cost = nn.computeCost(imagesList);
			printf("Epoch: %d cost: %f\n", epoch, cost);

			debugTestImage(nn, img, epoch);
		}
	}
#elif 0
	// Debug test with N images
	{
		//const int N = 50;
		const int N = 3;
		LabeledImage& img = trainingImages[0];

		std::vector<LabeledImage> imagesList;
		imagesList.reserve(N);
		for(int i=0 ; i < N ; i++)
			imagesList.push_back(trainingImages[i]);

		std::vector<const LabeledImage*> imagesPtrList;
		imagesPtrList.reserve(N);
		for(int i=0 ; i < N ; i++)
			imagesPtrList.push_back(&imagesList[i]);

		float cost = 0.f;

		cost = nn.computeCost(imagesList);
		printf("Init: cost: %f\n", cost);
		debugTestImage(nn, trainingImages[0], -1);
		debugTestImage(nn, trainingImages[1], -1);
		debugTestImage(nn, trainingImages[2], -1);

		for(int epoch=0 ; epoch < 2000 ; epoch++)
		{
			std::vector<std::vector<float>> weightAndBiasesCorrectionPerLayer;
			nn.backPropagateImages(imagesPtrList, weightAndBiasesCorrectionPerLayer);

			static float s_learningRate = 3.f;
			for(std::vector<float>& weightAndBiasesCorrection : weightAndBiasesCorrectionPerLayer)
				for(float& f : weightAndBiasesCorrection)
					f = -s_learningRate * f;
		
			nn.addToWeightAndBiases(weightAndBiasesCorrectionPerLayer);

			cost = nn.computeCost(imagesList);
			printf("Epoch: %d cost: %f\n", epoch, cost);

			debugTestImage(nn, trainingImages[0], epoch);
			debugTestImage(nn, trainingImages[1], epoch);
			debugTestImage(nn, trainingImages[2], epoch);
		}
	}
#elif 0	// WORKING CASE!!
	// Training
	for(int epoch=0 ; true ; epoch++)
	{
		std::vector<const LabeledImage*> imgBatch;
		imgBatch.reserve(batchSize);
		for(int i=0 ; i < batchSize ; i++)
		{
			const int idx = randInt(0, (int)(trainingImages.size()-1));
			imgBatch.push_back(&trainingImages[idx]);
		}

		std::vector<std::vector<float>> weightAndBiasesCorrectionPerLayer;
		nn.backPropagateImages(imgBatch, weightAndBiasesCorrectionPerLayer);

		// weightAndBiasesCorrection = -learningRate * costGradient
		static float s_learningRate = 3.f;
		for(std::vector<float>& weightAndBiasesCorrection : weightAndBiasesCorrectionPerLayer)
			for(float& f : weightAndBiasesCorrection)
				f = -s_learningRate * f;
		
		nn.addToWeightAndBiases(weightAndBiasesCorrectionPerLayer);

		// Debug test
		static bool s_bDebugTest = false;
		if(s_bDebugTest)
		{
			static int idxImageToTest = 0;
			LabeledImage& img = testImages[idxImageToTest];

			_debugTestImage(nn, img, epoch);
		}

		static int s_nbEpochsBetweenTests = 500;
		if(s_nbEpochsBetweenTests > 0 && epoch % s_nbEpochsBetweenTests == 0)
		{
			int nbGoodAnswers = 0;
			int nbBadAnswers = 0;
			Layer& lastLayer = nn.layers[_countof(nn.layers)-1];
			for(const LabeledImage& img : testImages)
			{
				nn.feedForward(img, false);
				int answer = 0;
				for(int i=0 ; i < 10 ; i++)
					answer = lastLayer.neuronValues[i] > lastLayer.neuronValues[answer] ? i : answer;
				if(answer == img.label)
					nbGoodAnswers++;
				else
					nbBadAnswers++;
			}
			printf("--- Epoch %d: good: %d  bad: %d ---\n", epoch, nbGoodAnswers, nbBadAnswers);
		}

		static int s_nbEpochsBetweenCostEvaluation = 500;
		if(s_nbEpochsBetweenCostEvaluation > 0 && epoch % s_nbEpochsBetweenCostEvaluation == 0)
		{
			const float cost = nn.computeCost(testImages);
			printf("--- Epoch %d: cost: %f ---\n", epoch, cost);
		}

		static bool s_bSaveResults = false;
		if(s_bSaveResults)
		{
			s_bSaveResults = false;
			nn.saveToFile("weightsAndBiases.bin");
		}
	}
#endif

	return EXIT_SUCCESS;
}
