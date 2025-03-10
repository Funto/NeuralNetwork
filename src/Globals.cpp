#include "Globals.h"
#include "NeuralNetwork.h"
#include "GUI.h"
#include <random>

#define STB_SPRINTF_IMPLEMENTATION
#include "stb_sprintf.h"

#ifdef _MSC_VER
	#pragma comment(lib, "externals\\glfw-3.3.8.bin.WIN64\\lib-vc2022\\glfw3.lib")
	#pragma comment(lib, "opengl32.lib")
#endif

Globals gData;

Globals::~Globals()
{
}

void testPGM()
{
	int sx = 28;
	int sy = 28;
	unsigned char* data = new unsigned char[sx * sy];
	for(int y=0 ; y < sy ; y++)
	{
		for (int x = 0; x < sx; x++)
		{
			data[x+y*sx] = 255*(x + y) / (28*2);
		}
	}
	debugSavePGM(data, sx, sy, "test.pgm");
	delete [] data;
}

static std::default_random_engine randGenerator;
static std::normal_distribution<double> randNormalDistribution(0.f, 1.f);

float randNormal()
{
	const double d = randNormalDistribution.operator()(randGenerator);
	return (float)d;
}

int randInt(int minVal, int maxVal)
{
	std::uniform_int_distribution<int> distrib(minVal, maxVal);
	return distrib(randGenerator);
}

[[nodiscard]] const char* formatTempStr(const char* fmt, ...)
{
	static thread_local char buffer[2048];

	va_list va;
	va_start(va, fmt);
	stbsp_vsnprintf(buffer, _countof(buffer), fmt, va);
	return buffer;
}
