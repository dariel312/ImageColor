#include <stdio.h>
#include <iostream>
#include <cmath>
#include "ColorConverter.h"

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
			color.B = data[j];
			color.G = data[j + 1];
			color.R = data[j + 2];
			body[i][y] = rgb2hsv(color);
		}
	}

	fclose(f);
	delete data;
	return img;
}

RgbColor AnalyzeColor(HsvColor** img, int startX, int endX, int countX, int countY) {
	RgbColor currentColor;
	int colorDepth = 24;

	int colors[COLOR_DEPTH] = { 0 };

	for (int x = 0; x < countX; x++)
	{
		for (int y = 0; y < countY; y++)
		{
			HsvColor *color = (img[x]+y);
			float quotient = color->H / 24;
			int range = (int)round(quotient);

			float saturation = scaled.GetPixel(x, y).GetSaturation();
			float brightness = scaled.GetPixel(x, y).GetBrightness();

			if (range == colorDepth)
				range = 0;

			if (brightness >= 0.10f && brightness <= 0.80f && saturation >= 0.3f)
				colors[range]++;

		}
	}

	int max = 0;
	for (int i = 0; i < colors.Length; i++)
		if (colors[i] > colors[max])
			max = i;

	currentColor = ColorTranslator.FromWin32(ColorHLSToRGB(240 * (max * 15) / 360, 132, 240));
	return currentColor;
}

int main() {
	Image* img;
	img = ReadBMP("C:/Users/darie/Source/Repos/ImageColor/Debug/example.bmp");
	//img = ReadBMP("example.bmp");
	int height = img->Height;
	int width = img->Width;
	HsvColor** data = img->Data;



	delete img->Data;
	delete img;
	return 0;
}
