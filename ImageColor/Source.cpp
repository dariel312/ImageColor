#include <stdio.h>
#include <iostream>
using namespace std;
struct Pixel
{
	unsigned char R;
	unsigned char G;
	unsigned char B;
};
class Image
{
public:
	int Width;
	int Height;
	Pixel** Data;

	Image(int Width, int Height)
	{
		this->Width = Width;
		this->Height = Height;
		this->Data = new Pixel*[Width];
		for (int z = 0; z < Width; z++)
			this->Data[z] = new Pixel[Height];
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
	Pixel** body = img->Data;

	int row_padded = (width * 3 + 3) & (~3);
	unsigned char* data = new unsigned char[row_padded];
	unsigned char tmp;

	for (int i = 0; i < height; i++)
	{
		fread(data, sizeof(unsigned char), row_padded, f);
		for (int j = 0; j < width * 3; j += 3)
		{
			int y = j / 3;
			body[i][y].B = data[j];
			body[i][y].G = data[j + 1];
			body[i][y].R = data[j + 2];
		}
	}

	fclose(f);
	delete data;
	return img;
}

int main() {
	Image* img;
	//img = ReadBMP("C:/Users/darie/Source/Repos/ImageColor/Debug/example.bmp");
	img = ReadBMP("example.bmp");
	int height = img->Height;
	int width = img->Width;
	Pixel** data = img->Data;

	for (int i = 0; i < height; i++)
	{
		for (int j = 0; j < width; j++)
		{
			cout << i*height + j << " " << "R: " << (int)data[i][j].R << " G: " << (int)data[i][j].G << " B: " << (int)data[i][j].B << endl;
		}
	}

	cout << "Height: " << height << endl;
	cout << " Width: " << width << endl;

	return 0;
}