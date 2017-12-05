#include "Colors.h"

HsvColor RgbToHsv(RgbColor in)
{
	HsvColor out;
	double min, max, delta;

	min = in.R < in.G ? in.R : in.G;
	min = min < in.B ? min : in.B;

	max = in.R > in.G ? in.R : in.G;
	max = max > in.B ? max : in.B;

	out.V = max;                                // v
	delta = max - min;
	if (delta < 0.00001)
	{
		out.S = 0;
		out.H = 0; // undefined, maybe nan?
		return out;
	}
	if (max > 0.0) { // NOTE: if Max is == 0, this divide would cause a crash
		out.S = (delta / max);                  // s
	}
	else {
		// if max is 0, then r = g = b = 0              
		// s = 0, h is undefined
		out.S = 0.0;
		out.H = 0;  // its now undefined
		return out;
	}
	if (in.R >= max) // > is bogus, just keeps compilor happy
		out.H = (in.G - in.B) / delta; // between yellow & magenta
	else
		if (in.G >= max)
			out.H = 2.0 + (in.B - in.R) / delta;  // between cyan & yellow
		else
			out.H = 4.0 + (in.R - in.G) / delta;  // between magenta & cyan

	out.H *= 60.0; // degrees

	if (out.H < 0.0)
		out.H += 360.0;

	return out;
}


RgbColor HsvToRgb(HsvColor in)
{
	double hh, p, q, t, ff;
	long i;
	RgbColor out;

	if (in.S <= 0.0) {       // < is bogus, just shuts up warnings
		out.R = in.V;
		out.G = in.V;
		out.B = in.V;
		return out;
	}
	hh = in.H;
	if (hh >= 360.0) hh = 0.0;
	hh /= 60.0;
	i = (long)hh;
	ff = hh - i;
	p = in.V * (1.0 - in.S);
	q = in.V * (1.0 - (in.S * ff));
	t = in.V * (1.0 - (in.S * (1.0 - ff)));

	switch (i) {
	case 0:
		out.R = in.V;
		out.G = t;
		out.B = p;
		break;
	case 1:
		out.R = q;
		out.G = in.V;
		out.B = p;
		break;
	case 2:
		out.R = p;
		out.G = in.V;
		out.B = t;
		break;

	case 3:
		out.R = p;
		out.G = q;
		out.B = in.V;
		break;
	case 4:
		out.R = t;
		out.G = p;
		out.B = in.V;
		break;
	case 5:
	default:
		out.R = in.V;
		out.G = p;
		out.B = q;
		break;
	}
	return out;
}