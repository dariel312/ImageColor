#include <stdio.h>
#include <iostream>
#include <cmath>
#include "Colors.h"

#define COLOR_DEPTH 24
using namespace std;

class Image
{
public:
	int Width;
	int Height;
	HsvColor** Data;

	Image(int Width, int Height)
	{
		this->Width = Width;
		this->Height = Height;
		this->Data = new HsvColor*[Width];
		for (int z = 0; z < Width; z++)
			this->Data[z] = new HsvColor[Height];
	}
};
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

RgbColor AnalyzeImage(HsvColor** img, int startX, int endX, int countX, int countY) {

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

int main() {
	//Read image from file
	Image* img = ReadBMP("example.bmp");

	//Analyze image
	RgbColor c = AnalyzeImage(img->Data, 0, 0, img->Width, img->Height);
	cout << "The most used color in this image is:" << endl;
	cout << "R: " << c.R * 255 << endl;
	cout << "G: " << c.G * 255 << endl;
	cout << "B: " << c.B * 255 << endl;


	delete img->Data;
	delete img;
	return 0;
}
