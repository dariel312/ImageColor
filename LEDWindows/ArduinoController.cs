using System;
using System.Drawing;
using System.Drawing.Imaging;
using System.Runtime.InteropServices;
using System.Text;
using System.Windows.Forms;
using System.IO.Ports;

namespace LEDController
{
    public class ArduinoController
    {

        [DllImport("shlwapi.dll")]
        public static extern int ColorHLSToRGB(int H, int L, int S);

        public Color CurrentColor { get { return currentColor; } }
        public byte ComPort {get { return 3; } }

        private System.Threading.Thread worker;
        private Color currentColor;
        private Bitmap bmpScreenshot;
        private Bitmap scaled;
        private Graphics gfxScreenshot;
        private Graphics gfxScaled;
        private SerialPort ardPort;
        private bool stopWorker = false;

        public ArduinoController()
        {
            ardPort = new SerialPort("COM" + ComPort, 9600);

        }
        public void Start()
        {
            worker = new System.Threading.Thread(this.workerLoop);

            //try
            //{
            //    ardPort.Open();
            //} 
            //catch (Exception Ex)
            //{
            //    throw new Exception("Could not find a controller on COM3. Try disconnecting then reconnecting the controller.", Ex);
            //}

            worker.Start();
        }
        public void Stop()
        {
            stopWorker = true;
            ardPort.Close();
        }
        private void sendColor()
        {
            ardPort.Write(new byte[] { 100, currentColor.R, currentColor.G, currentColor.B }, 0, 4);
        }
        private void workerLoop()
        {
            bmpScreenshot = new Bitmap(Screen.PrimaryScreen.Bounds.Width, Screen.PrimaryScreen.Bounds.Height, PixelFormat.Format32bppArgb);
            scaled = new Bitmap(160, 90);

            try
            {
               // ardPort.Open();
            }  //We open here because Form Exit event will be called
            catch
            {
                MessageBox.Show("Could not find a controller on COM3. Try disconnecting then reconnecting the controller.");

                //Stop();
                //Application.Exit();
            }

            while (!stopWorker)
            {
                GetScreenColor();

                if (ardPort.IsOpen)
                    sendColor();

                System.Threading.Thread.Sleep(15);
            }

        }
        private void GetScreenColor()
        {
            gfxScreenshot = Graphics.FromImage(bmpScreenshot);

            gfxScreenshot.CopyFromScreen(Screen.PrimaryScreen.Bounds.X, Screen.PrimaryScreen.Bounds.Y, 0, 0, Screen.PrimaryScreen.Bounds.Size,
                                             CopyPixelOperation.SourceCopy);

            gfxScaled = Graphics.FromImage(scaled);
            gfxScaled.DrawImage(bmpScreenshot, new RectangleF(0, 0, scaled.Width, scaled.Height));

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

            currentColor = ColorTranslator.FromWin32(ColorHLSToRGB(255 * (colorMax* (int)bucketNumber) / 360, 132, 240));


        }
    }
}
