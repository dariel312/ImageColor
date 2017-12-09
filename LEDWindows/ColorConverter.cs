using System;
using System.Collections.Generic;
using System.Drawing;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading.Tasks;

namespace LEDController
{
    public class ColorConverter
    {
        public const int COLOR_DEPTH = 255;

        [DllImport("shlwapi.dll")]
        public static extern int ColorHLSToRGB(int H, int L, int S);
        public static Color GetImageColor(Bitmap Image)
        {
            var scaled = Image;
            
            int colorDepth = 256;
            int[] colors = new int[colorDepth];
            float bucketNumber = 360 / colorDepth;

            for (int x = 0; x < scaled.Size.Width; x++)
            {
                for (int y = 0; y < scaled.Size.Height; y++)
                {
                    float hue = scaled.GetPixel(x, y).GetHue();
                    int colorIndex = (int)Math.Round(hue / bucketNumber);

                    float saturation = scaled.GetPixel(x, y).GetSaturation();
                    float brightness = scaled.GetPixel(x, y).GetBrightness();

                    colorIndex = colorIndex % colorDepth;

                    if (brightness >= 0.10f && brightness <= 0.80f && saturation >= 0.3f)
                        colors[colorIndex]++;

                }
            }

            int colorMax = 0;
            for (int i = 0; i < colors.Length; i++)
                if (colors[i] > colors[colorMax])
                    colorMax = i;

            return ColorTranslator.FromWin32(ColorHLSToRGB(255 * (colorMax * (int)bucketNumber) / 360, 132, 240));
        }

    }
}
