#pragma once
#include <stdio.h>
#include <cmath>
#include "device_launch_parameters.h"
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include "math.h"

#define COLOR_DEPTH 32
#define MAX_DEGREES 360.0
#define MIN_BRIGHTNESS 0.10
#define MIN_SATURATION 0.30


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

int RoundToMultiple(int, int);
RgbColor HueInxToRGB(int);
RgbColor HsvToRgb(HsvColor hsv);
HsvColor RgbToHsv(RgbColor rgb);
Image* ReadBMP(char*);
RgbColor AnalyzeColor(HsvColor*, int, int);
__global__
void AnalyzeColorGPU(HsvColor*, int, int, int*);
__global__
void SumBucket(int*, int);

