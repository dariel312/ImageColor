#pragma once
#include <stdio.h>
#include <cmath>
#include "device_launch_parameters.h"
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

#ifdef __CUDACC__
#define cuda_SYNCTHREADS() __syncthreads()
#else
#define cuda_SYNCTHREADS()
#endif


#define COLOR_DEPTH 4
#define MAX_DEGREES 360
#define MIN_BRIGHTNESS 0.10f
#define MAX_BRIGHTNESS 0.80f
#define MIN_SATURATION 0.3f
#define BAD_PIXEL -1


typedef struct RgbColor
{
	float R;
	float G;
	float B;
} RgbColor;
typedef struct HsvColor
{
	float H;
	float S;
	float V;
} HsvColor;
class Image
{
public:
	int Width;
	int Height;
	HsvColor* Data;

	Image(int Width, int Height)
	{
		this->Width = Width;
		this->Height = Height;
		this->Data = new HsvColor[Width * Height];
	}
};

RgbColor HsvToRgb(HsvColor hsv);
HsvColor RgbToHsv(RgbColor rgb);
Image* ReadBMP(char*);
RgbColor AnalyzeColor(HsvColor*, int, int);
int RoundToMultiple(int, int);
__global__
void AnalyzeColorGPU(HsvColor*, int, int, int*);
__global__
void ReductionMaxImage(HsvColor*, int, int);

