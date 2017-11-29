typedef struct RgbColor
{
	unsigned char R;
	unsigned char G;
	unsigned char B;
} RgbColor;

typedef struct HsvColor
{
	float H;
	float S;
	float V;
} HsvColor;

RgbColor HsvToRgb(HsvColor hsv)
{
	RgbColor rgb;
	unsigned char region, remainder, p, q, t;

	if (hsv.S == 0)
	{
		rgb.R = hsv.V;
		rgb.G = hsv.V;
		rgb.B = hsv.V;
		return rgb;
	}

	region = hsv.H / 43;
	remainder = (hsv.H - (region * 43)) * 6;

	p = (hsv.V * (255 - hsv.S)) >> 8;
	q = (hsv.V * (255 - ((hsv.S * remainder) >> 8))) >> 8;
	t = (hsv.V * (255 - ((hsv.S * (255 - remainder)) >> 8))) >> 8;

	switch (region)
	{
	case 0:
		rgb.R = hsv.V; rgb.G = t; rgb.B = p;
		break;
	case 1:
		rgb.R = q; rgb.G = hsv.V; rgb.B = p;
		break;
	case 2:
		rgb.R = p; rgb.G = hsv.V; rgb.B = t;
		break;
	case 3:
		rgb.R = p; rgb.G = q; rgb.B = hsv.V;
		break;
	case 4:
		rgb.R = t; rgb.G = p; rgb.B = hsv.V;
		break;
	default:
		rgb.R = hsv.V; rgb.G = p; rgb.B = q;
		break;
	}

	return rgb;
}

HsvColor RgbToHsv(RgbColor rgb)
{
	HsvColor hsv;
	unsigned char rgbMin, rgbMax;

	rgbMin = rgb.R < rgb.G ? (rgb.R < rgb.B ? rgb.R : rgb.B) : (rgb.G < rgb.B ? rgb.G : rgb.B);
	rgbMax = rgb.R > rgb.G ? (rgb.R > rgb.B ? rgb.R : rgb.B) : (rgb.G > rgb.B ? rgb.G : rgb.B);

	hsv.V = rgbMax;
	if (hsv.V == 0)
	{
		hsv.H = 0;
		hsv.S = 0;
		return hsv;
	}

	hsv.S = 255 * long(rgbMax - rgbMin) / hsv.V;
	if (hsv.S == 0)
	{
		hsv.H = 0;
		return hsv;
	}

	if (rgbMax == rgb.R)
		hsv.H = 0 + 43 * (rgb.G - rgb.B) / (rgbMax - rgbMin);
	else if (rgbMax == rgb.G)
		hsv.H = 85 + 43 * (rgb.B - rgb.R) / (rgbMax - rgbMin);
	else
		hsv.H = 171 + 43 * (rgb.R - rgb.G) / (rgbMax - rgbMin);

	return hsv;
}


HsvColor rgb2hsv(RgbColor in)
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
		out.H = -1;                            // its now undefined
		return out;
	}
	if (in.R >= max)                           // > is bogus, just keeps compilor happy
		out.H = (in.G - in.B) / delta;        // between yellow & magenta
	else
		if (in.G >= max)
			out.H = 2.0 + (in.B - in.R) / delta;  // between cyan & yellow
		else
			out.H = 4.0 + (in.R - in.G) / delta;  // between magenta & cyan

	out.H *= 60.0;                              // degrees

	if (out.H < 0.0)
		out.H += 360.0;

	return out;
}
