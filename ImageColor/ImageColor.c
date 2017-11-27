#include <stdio.h>
#include <stdlib.h>
#include "bmplib.h"
#include <getopt.h>

void get2dArray(PIXEL** newBmp, PIXEL* bmp, int rows, int cols)
{
	int x;
	int y;

	for (x = 0; x < cols; x++)
	{
		for (y = 0; y < rows; y++)
		{
			newBmp[x][y].r = bmp[x*rows + y].r;
			newBmp[x][y].g = bmp[x*rows + y].g;
			newBmp[x][y].b = bmp[x*rows + y].b;
			
			printf("%d %d %d", newBmp[x][y].r, newBmp[x][y].g, newBmp[x][y].b);
		}
	}
}

int main(int argc, char** argv)
{
	extern char *optarg;
	extern int optind;
	char *outFile = NULL;
	char *inFile = "example.bmp";

	int row, col;
	int* newRow = &row;
	int* newCol = &col;

	//read file
	PIXEL *bmp, *newbmp;
	readFile(inFile, &row, &col, &bmp);

	//get 2d array
	PIXEL** img = calloc(col, sizeof(PIXEL*));
	int y = 0;
	for (y = 0; y < col; y++)
	{
		img[y] = calloc(row, sizeof(PIXEL));
	}
	get2dArray(img, bmp, row, col);

	printf("Rows: %d Cols: %d\n", row, col);
	//TODO for loop for FREE img
	free(bmp);
	free(newbmp);

	return 0;
}




