#include "ImageColor.h"

int RoundToMultiple(int numToRound, int multiple)
{
	if (multiple == 0)
		return numToRound;

	int remainder = numToRound % multiple;
	if (remainder == 0)
		return numToRound;

	return numToRound + multiple - remainder;
}

RgbColor AnalyzeColor(HsvColor* Img, int Width, int Height) {

	int colors[COLOR_DEPTH] = { 0 };

	for (int y = 0; y < Height; y++)
	{
		HsvColor *color = (Img + y*Width);
		for (int x = 0; x < Width; x++)
		{
			//Compute colors index within bucket
			int colorIndex = (int)round(color->H * COLOR_DEPTH / MAX_DEGREES);

			if (colorIndex == COLOR_DEPTH)
				colorIndex = 0;

			if (color->V >= MIN_BRIGHTNESS && color->V <= MAX_BRIGHTNESS && color->S >= MIN_SATURATION)
				colors[colorIndex]++;

			color++; //Move ptr to next
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
void AnalyzeColorGPU(HsvColor* data, int Width, int Height, int* bucket)
{
	int xId = blockIdx.x * blockDim.x + threadIdx.x;
	int yId = blockIdx.y * blockDim.y + threadIdx.y;
	int blockStartX = Width / gridDim.x * blockIdx.x;
	int blockStartY = Height / gridDim.y * blockIdx.y;
	int threadStartX = threadIdx.x * Width / gridDim.x / blockDim.x + blockStartX;
	int threadStartY = threadIdx.y * Height / gridDim.y / blockDim.y + blockStartY;
	int xLength = Width / gridDim.x / blockDim.x;
	int yLength = Height / gridDim.y / blockDim.y;


	HsvColor *color = NULL;
	printf("X, Y: (%d, %d)\n", threadStartX, threadStartY);

	for (int y = 0; y < yLength; y++)
	{
		color = (data + y*yLength + threadStartX);
		for (int x = 0; x < xLength; x++)
		{
			int colorIndex = (int)round(color->H * COLOR_DEPTH / MAX_DEGREES);

			if (colorIndex == COLOR_DEPTH)
				colorIndex = 0;

			if (color->V >= MIN_BRIGHTNESS && color->V <= MAX_BRIGHTNESS && color->S >= MIN_SATURATION)
			{
				int bucketInx = colorIndex + (x * COLOR_DEPTH) + (yId * blockDim.x  * gridDim.x * COLOR_DEPTH);
				bucket[bucketInx]++;
			}
		

			color++;
		}
	}
}

__global__
void ReductionMaxImage(HsvColor* data, int Width, int Height)
{
	extern __shared__ int sharedData[];
	// each thread loads one element from global to shared mem
	unsigned int threadID = threadIdx.x;
	unsigned int i = blockIdx.x*(blockDim.x * 2) + threadIdx.x;
	sharedData[threadID] = data[i].H + data[i + blockDim.x].H;

	cuda_SYNCTHREADS();
	// do reduction in shared mem
	//for (unsigned int s = 1; s < blockDim.x; s *= 2) {

	//	int index = 2 * s * threadID;
	//	if (index < blockDim.x) {
	//		sharedData[index] += sharedData[index + s];
	//	}
	//	cuda_SYNCTHREADS();
	//}

	for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
		if (threadID < stride) {
			if(sharedData[threadID] < sharedData[threadID + stride])
				sharedData[threadID] = sharedData[threadID + stride];
		}
		cuda_SYNCTHREADS();
	}
	// write result for this block to global mem
	if (threadID == 0)
		data[blockIdx.x].H = sharedData[0];
}