#include <stdio.h>
#include <iostream>
#include <cmath>
#include "Color.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#define GPU_BLOCKS 16
#define GPU_THREADS 255
#define COLOR_DEPTH 24
using namespace std;

Image* ReadBMP(char* filename)
{
	FILE* f = fopen(filename, "rb");

	if (f == NULL)
		throw "Argument Exception";

	unsigned char info[54];
	fread(info, sizeof(unsigned char), 54, f); // read the 54-byte header

	int width = *(int*)&info[18];
	int height = *(int*)&info[22];

	//declare return data
	Image* img = new Image(width, height);
	HsvColor** body = img->Data;

	int row_padded = (width * 3 + 3) & (~3);
	unsigned char* data = new unsigned char[row_padded];
	unsigned char tmp;

	for (int i = 0; i < height; i++)
	{
		fread(data, sizeof(unsigned char), row_padded, f);
		for (int j = 0; j < width * 3; j += 3)
		{
			int y = j / 3;
			RgbColor color;
			color.B = data[j] / 255.0;
			color.G = data[j + 1] / 255.0;
			color.R = data[j + 2] / 255.0;
			body[i][y] = RgbToHsv(color);
		}
	}

	fclose(f);
	delete data;
	return img;
}

RgbColor AnalyzeImage(HsvColor** img, int countX, int countY) {

	int colors[COLOR_DEPTH] = { 0 };

	for (int x = 0; x < countX; x++)
	{
		for (int y = 0; y < countY; y++)
		{
			HsvColor *color = (img[x] + y);
			int colorIndex = (int)round(color->H * COLOR_DEPTH / 360);

			if (colorIndex == COLOR_DEPTH)
				colorIndex = 0;

			if (color->V >= 0.10f && color->V <= 0.80f && color->S >= 0.3f)
				colors[colorIndex]++;
		}
	}

	int max = 0;
	for (int i = 0; i < COLOR_DEPTH; i++)
		if (colors[i] > colors[max])
			max = i;

	HsvColor currentColor;
	currentColor.S = 1;
	currentColor.V = 1;
	currentColor.H = max * 360 / COLOR_DEPTH;
	return HsvToRgb(currentColor);
}

__global__ 
void AnalyzeColorGPU(HsvColor** Img, int SizeX, int SizeY, int** Buckets) {

}

void doCPU(Image* img)
{
	RgbColor c = AnalyzeImage(img->Data, img->Width, img->Height);
	cout << "The most used color in this image is:" << endl;
	cout << "R: " << c.R * 255 << endl;
	cout << "G: " << c.G * 255 << endl;
	cout << "B: " << c.B * 255 << endl;
}
void doGPU(Image* Img, int Blocks, int Threads) {

	int **buckets = 0;
	int **data= 0;

	//Allocate Img
	cudaMalloc(&data, sizeof(HsvColor*) * Img->Width);
	//for (int x = 0; x < Img->Width; x++)
	//	cudaMalloc(&data[x], sizeof(HsvColor) * Img->Height);

	AnalyzeColorGPU<<<Blocks, Threads>>>(Img->Data, img, buckets);
}
int main() {
	Image* img = ReadBMP("example.bmp");

	//Made as a variable so in the future we can pass thread count as a variable in command line
	int numBlocks = GPU_BLOCKS;
	int numThreads = GPU_THREADS; 

	doCPU(img);
	doGPU(img, numBlocks, numThreads);

	return 0;
}
