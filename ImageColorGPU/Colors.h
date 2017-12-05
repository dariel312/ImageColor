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
