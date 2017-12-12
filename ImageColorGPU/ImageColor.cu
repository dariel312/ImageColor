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
RgbColor HueInxToRGB(int HueInx)
{
	HsvColor color;
	color.S = 1;
	color.V = 1;
	color.H = HueInx * 360 / COLOR_DEPTH;
	return HsvToRgb(color);
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

			if (color->V >= MIN_BRIGHTNESS && color->V && color->S >= MIN_SATURATION)
				colors[colorIndex]++;

			color++; //Move ptr to next
		}
	}

	int maxInx = 0;
	for (int i = 0; i < COLOR_DEPTH; i++)
		if (colors[i] > colors[maxInx])
			maxInx = i;

	return HueInxToRGB(maxInx);
}

__global__
void AnalyzeColorGPU(HsvColor* data, int Width, int Height, int* bucket)
{
	int xId = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x;

	for (int i = xId; i < Width * Height; i += stride)
	{
		HsvColor* color = (data + i);
		long long colorIndex = llrintf(color->H * COLOR_DEPTH / MAX_DEGREES);

		if (colorIndex == COLOR_DEPTH)
			colorIndex = 0;

		int bucketInx = colorIndex + (xId*COLOR_DEPTH);

		if (color->V > MIN_BRIGHTNESS && color->S > MIN_SATURATION)
			bucket[bucketInx]++;
	}

}

__global__
void SumBucket(int* bucket, int BucketsLength)
{
	int xId = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = COLOR_DEPTH;
	int count = 0;

	for (int i = 0; i < BucketsLength; i++)
		count += bucket[xId + i * stride];

	bucket[xId] = count;
}