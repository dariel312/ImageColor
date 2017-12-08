#include "ImageColor.h"

using namespace std;


void doCPU(Image* img)
{
	RgbColor c = AnalyzeColor(img->Data, img->Width, img->Height);
	printf("The most used color in this image is:\n");
	printf("R: %3.1f\nG: %3.1f\nB: %3.1f\n", c.R * 255, c.G * 255, c.B * 255);
}

void doGPU(Image* Img)
{
	HsvColor *data = NULL;
	unsigned long long size = sizeof(HsvColor) * Img->Width * Img->Height;

	//allocate mem
	cudaMalloc<HsvColor>(&data, size);
	cudaMemcpy(data, Img->Data, size, cudaMemcpyHostToDevice);

	//define block size
	int blocksX = RoundToMultiple(Img->Width / 32.0, 2);
	int blocksY = RoundToMultiple(Img->Height / 16.0, 2);
	dim3 threads = dim3(32, 16);
	dim3 gridDim = dim3(blocksX, blocksY);
	
	//execute gpu alg
	AnalyzeColorGPU <<<gridDim, threads>>> (data, Img->Width, Img->Height);
	cudaError_t success = cudaDeviceSynchronize();

	//ReductionMaxImage<<<blocksY, 512>>> (data, Img->Width, Img->Height);

	//Print
	cudaMemcpy(Img->Data, data, size, cudaMemcpyDeviceToHost);

	//Sum block indx
	int bucket[COLOR_DEPTH] = { 0 };
	for (int i = 0; i < blocksX * blocksY; i++)
	{
		int inx = (int)Img->Data[i].H;
		if(inx >= 0)
			bucket[inx]++;
		else {
			int sd = 1;
		}
	}

	int max = 0;
	for (int i = 0; i < COLOR_DEPTH; i++)
	{
		float val = bucket[i];
		if (val > bucket[max])
			max = i;
	}

	float curr = max * 360 / COLOR_DEPTH;
	cudaFree(data);
}

int main()
{
	Image* img = ReadBMP("example-01.bmp");

	doCPU(img);
	doGPU(img);

	system("pause");
	return 0;
}
