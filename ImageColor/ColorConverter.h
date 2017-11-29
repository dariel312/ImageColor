#pragma once
typedef struct RgbColor
{
	unsigned char R;
	unsigned char G;
	unsigned char B;
} RgbColor;

typedef struct HsvColor
{
	unsigned char H;
	unsigned char S;
	unsigned char V;
} HsvColor;

RgbColor HsvToRgb(HsvColor hsv);
HsvColor RgbToHsv(RgbColor rgb);
HsvColor rgb2hsv(RgbColor in);