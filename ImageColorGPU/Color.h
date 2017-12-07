#pragma once
typedef struct RgbColor
{
	float R;
	float G;
	float B;
} RgbColor;

typedef struct HsvColor
{
	float H;
	float S;
	float V;
} HsvColor;

RgbColor HsvToRgb(HsvColor hsv);
HsvColor RgbToHsv(RgbColor rgb);

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