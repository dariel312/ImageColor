#include "ImageColor.h"

using namespace std;
#define BLOCK_DIVISIONS 1
#define THREAD_DIVISIONS 1

Image* img = NULL;
int* bucket = NULL;
unsigned long long bucketSize = sizeof(int) * BLOCK_DIVISIONS*BLOCK_DIVISIONS * THREAD_DIVISIONS*THREAD_DIVISIONS * COLOR_DEPTH;
unsigned long long dataSize = 0;

void doCPU()
{
	RgbColor c = AnalyzeColor(img->Data, img->Width, img->Height);
	printf("The most used color in this image is:\n");
	printf("R: %3.1f\nG: %3.1f\nB: %3.1f\n", c.R * 255, c.G * 255, c.B * 255);
}

//Allocates memory for GPU
//Returns data pointer to GPU
//Returns bucket pointer to GPU
void allocateGPUMem(HsvColor** Data, int** Bucket)
{
	//Img
	unsigned long long size = sizeof(HsvColor) * img->Width * img->Height;
	cudaMalloc<HsvColor>(Data, size);
	cudaMemcpy(*Data, img->Data, size, cudaMemcpyHostToDevice);

	cudaMallocManaged<int>(Bucket, bucketSize);
}

//Copies data back from gpu to cpu
void copyBackToCPU(HsvColor** Data, int** Bucket)
{
	cudaMemcpy(img->Data, Data, dataSize, cudaMemcpyDeviceToHost);


}
void printBucket()
{
	long int total = 0;
	//BLOCK_DIVISIONS*BLOCK_DIVISIONS * THREAD_DIVISIONS*THREAD_DIVISIONS * COLOR_DEPTH
	for (int y = 0; y < THREAD_DIVISIONS*BLOCK_DIVISIONS; y++)
	{
		for (int x = 0; x <  THREAD_DIVISIONS*BLOCK_DIVISIONS; x++)
		{
			for (int c = 0; c < COLOR_DEPTH; c++)
			{
				int bucketInx = c + (x * COLOR_DEPTH) + (y * THREAD_DIVISIONS*BLOCK_DIVISIONS * COLOR_DEPTH);
				total += bucket[bucketInx];
				printf(" %d", bucket[bucketInx]);
			}
			printf("\n");
		}
	}
	printf("TOTAL: %d", total);
}

//Performs ImageColor analysis using GPU
void doGPU()
{
	HsvColor *data = NULL;
	dataSize = sizeof(HsvColor) * img->Width * img->Height;
	allocateGPUMem(&data, &bucket);

	dim3 threads = dim3(THREAD_DIVISIONS, THREAD_DIVISIONS);
	dim3 gridDim = dim3(BLOCK_DIVISIONS, BLOCK_DIVISIONS);

	//execute gpu alg
	AnalyzeColorGPU <<<gridDim, threads>>> (data, img->Width, img->Height, bucket);
	cudaError_t success = cudaDeviceSynchronize();

	copyBackToCPU(&data, &bucket);
	printBucket();
	//Sum block indx
	cudaFree(data);
	cudaFree(bucket);
}

int main()
{
	img = ReadBMP("example-01.bmp");

	doCPU();
	doGPU();

	system("pause");
	return 0;
}
/*int bucket[COLOR_DEPTH] = { 0 };
for (int i = 0; i < BLOCK_DIVISIONS * BLOCK_DIVISIONS; i++)
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

float curr = max * 360 / COLOR_DEPTH;*/
