#include "ImageColor.h"
#include <time.h>

#define THREAD_COUNT 512
#define BLOCK_COUNT 2 

Image* img = NULL;  //Image read from DISK
HsvColor* imgGPU = NULL; //array of pixels on the gpu
int* bucketGPU = NULL; //Pointer to buckets (Plural) on GPU
int* bucketCPU = NULL; //Pointer to bucket (summed) on cpu
unsigned long long bucketSizeGPU = sizeof(int) * THREAD_COUNT * BLOCK_COUNT * COLOR_DEPTH; //Size of GPU Buckets
unsigned long long bucketSizeCPU = sizeof(int) * COLOR_DEPTH; //Size of CPU Bucket

void doCPU()
{
	clock_t time;
	time = clock();
	RgbColor c = AnalyzeColor(img->Data, img->Width, img->Height);
	time = clock() - time;

	double time_taken = ((double)time) / CLOCKS_PER_SEC * 1000; // in seconds
	printf("The most used color in this image by CPU is: ");
	printf("RGB(%3.1f, %3.1f, %3.1f)\n", c.R * 255, c.G * 255, c.B * 255);
	printf("With time: %.1f milliseconds\n\n", time_taken);
}

//Allocates memory for GPU
void allocateMem()
{
	//bucket cpu
	bucketCPU = (int*)malloc(bucketSizeCPU);

	//bucket GPU
	cudaMalloc<int>(&bucketGPU, bucketSizeGPU);

	//Img GPU
	unsigned long long imgSize = sizeof(HsvColor) * img->Width * img->Height;
	cudaMalloc<HsvColor>(&imgGPU, imgSize);
	cudaMemcpy(imgGPU, img->Data, imgSize, cudaMemcpyHostToDevice);
}

//Runs bucket analysis
void execAnalyzeColorGPU()
{
	dim3 threadsAnalyze(THREAD_COUNT);
	dim3 blocks(BLOCK_COUNT);

	AnalyzeColorGPU << <blocks, threadsAnalyze >> > (imgGPU, img->Width, img->Height, bucketGPU);
	cudaError_t success = cudaDeviceSynchronize();
}

//Sums buckets from all threads
void execSumGPU()
{
	dim3 threadsSum(COLOR_DEPTH);
	dim3 blocksSum(1);
	SumBucket << <blocksSum, threadsSum >> > (bucketGPU, THREAD_COUNT);
	cudaError_t success = cudaDeviceSynchronize();
}

void printBucket()
{
	for (int y = 0; y < COLOR_DEPTH; y++)
	{
		int bucketInx = y;
		printf(" %d", bucketCPU[bucketInx]);
	}
}

//Gets the index of the max bucket
int getMaxBucketInx()
{
	int maxInx = 0;
	for (int i = 0; i < COLOR_DEPTH; i++)
	{
		if (bucketCPU[i] > bucketCPU[maxInx])
			maxInx = i;
	}

	return maxInx;
}

//Performs ImageColor analysis using GPU
void doGPU()
{
	cudaEvent_t gpuStart, gpuStop;
	allocateMem();

	//Setup timer
	cudaEventCreate(&gpuStart);
	cudaEventCreate(&gpuStop);

	//timer start
	cudaEventRecord(gpuStart);
	execAnalyzeColorGPU();
	execSumGPU();
	cudaEventRecord(gpuStop);
	cudaEventSynchronize(gpuStop);
	//timer end
	float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, gpuStart, gpuStop);

	//copy back GPU buckets to CPU summed
	cudaMemcpy(bucketCPU, bucketGPU, bucketSizeCPU, cudaMemcpyDeviceToHost);

	int maxInx = getMaxBucketInx();
	RgbColor c = HueInxToRGB(maxInx);
	printf("The most used color in this image by GPU is: ");
	printf("RGB(%3.1f, %3.1f, %3.1f)\n", c.R * 255, c.G * 255, c.B * 255);
	printf("With time : %.1f milliseconds\n", milliseconds);

	cudaFree(imgGPU);
	cudaFree(bucketGPU);
	free(bucketCPU);
}

int main(int argc, char* argv[])
{
	img = ReadBMP(argv[1]);
	printf("Image %s loaded.\n", argv[1]);
	printf("Img (X,Y): %dpx %dpx.\n\n", img->Width, img->Height);
	doCPU();
	doGPU();

	return 0;
}

