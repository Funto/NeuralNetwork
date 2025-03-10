#pragma once

#define _CRT_SECURE_NO_WARNINGS 1
#include <stdio.h>
#include <assert.h>
#include <vector>
#include <math.h>
#include <functional>
#include <memory>

#define IMGUI_DEFINE_MATH_OPERATORS
#include <imgui.h>

#ifdef __EMSCRIPTEN__
	#define DATA_DIR "emscripten_data"
#else
	#define DATA_DIR "data"
#endif

// Data from http://yann.lecun.com/exdb/mnist/ (now empty!)
#define TRAINING_IMAGES_FILENAME	DATA_DIR "/train-images.idx3-ubyte"
#define TRAINING_LABELS_FILENAME	DATA_DIR "/train-labels.idx1-ubyte"

#define TEST_IMAGES_FILENAME	DATA_DIR "/t10k-images.idx3-ubyte"
#define TEST_LABELS_FILENAME	DATA_DIR "/t10k-labels.idx1-ubyte"

#ifndef _countof
	#define _countof(_Array) (sizeof(_Array) / sizeof(_Array[0]))
#endif

inline void debugSavePxM(const unsigned char *data, int width, int height, const char *name, const char* strMagic, int bpp, int maxVal = 255)
{
	FILE*	f = fopen(name, "wb");
	assert(f);
	fprintf(f, "%s%d %d\n%d\n", strMagic, width, height, maxVal);
	fwrite(data, 1, width*height*bpp*sizeof(unsigned char), f);
	fclose(f);
}

inline void	debugSavePGM(const unsigned char *data, int width, int height, const char *name)		{ debugSavePxM(data, width, height, name, "P5\n", 1); }					// data is grayscale
inline void	debugSavePPM(const unsigned char *data, int width, int height, const char *name)		{ debugSavePxM(data, width, height, name, "P6\n", 3); }					// data is RGB

#include "LabeledImage.h"

union U32Union
{
	uint32_t u32;
	uint8_t  u8[4];
};

float	randNormal();
int		randInt(int minVal, int maxVal);

class ScopeExitTask {
	std::function<void()> func_;
public:
	ScopeExitTask(std::function<void()> func):
		func_(func) {
	}
	~ScopeExitTask() {
		func_();
	}
	ScopeExitTask& operator=(const ScopeExitTask&) = delete;
	ScopeExitTask(const ScopeExitTask&) = delete;
};

#define Defer_Merge(a, b) a##b
#define Defer_ID(a, b) Defer_Merge(a, b)
#define Defer(code) ScopeExitTask Defer_ID(_defer_, __COUNTER__) {[&]{code;}}

[[nodiscard]] const char* formatTempStr(const char* fmt, ...);

struct NeuralNetwork;
class GUI;
struct LabeledImage;

struct Globals
{
	std::unique_ptr<GUI>			pGUI;
	std::unique_ptr<NeuralNetwork>	pNN;
	std::vector<LabeledImage>		trainingImages;
	std::vector<LabeledImage>		testImages;
	bool							bDebugAlwaysRedrawContent = false;
	int								curFrame = 0;
	bool							bExitApp = false;

	~Globals();
};

extern Globals gData;
