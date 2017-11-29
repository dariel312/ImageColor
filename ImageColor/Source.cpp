#include <stdio.h>
#include <iostream>
#include "ColorConverter.h"

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

//Pixel AnalyzeColor(Pixel** img) {
//	return NULL;
//}

int main() {
	Image* img;
	//img = ReadBMP("C:/Users/darie/Source/Repos/ImageColor/Debug/example.bmp");
	img = ReadBMP("example.bmp");
	int height = img->Height;
	int width = img->Width;
	HsvColor** data = img->Data;
	cout << "H " << (int)data[0][0].H << endl;
	cout << "S " << (int)data[0][0].S << endl;
	cout << "V " << (int)data[0][0].V << endl;
	return 0;
}
