using System;
using System.Drawing;
using System.Drawing.Imaging;
using System.Runtime.InteropServices;
using System.Text;
using System.Windows.Forms;
using System.IO.Ports;
using System.Threading.Tasks;

namespace LEDController
{
    public partial class Form1 : Form
    {
        private Color currectColor;
        private Timer timer;
        private Bitmap screen;
        Bitmap scaled;
        private Graphics gfxScreenshot;
        private Graphics gfxScaled;
        public Form1()
        {
            InitializeComponent();

            screen = new Bitmap(Screen.PrimaryScreen.Bounds.Width, Screen.PrimaryScreen.Bounds.Height, PixelFormat.Format32bppArgb);
            scaled = new Bitmap(160, 90);
            gfxScreenshot = Graphics.FromImage(screen);

            timer = new Timer();
            timer.Tick += updateGUI;
            timer.Start();

            Task.Run(async () =>
            {
                while (true)
                {
                    update();
                    Task.Delay(33);
                }
            });
        }
        void updateGUI(object sender, EventArgs e)
        {
            label1.Text = "Memory: " + System.Diagnostics.Process.GetCurrentProcess().WorkingSet64 / 1048576 + "mb";
            Bitmap img = new Bitmap(400, 400, PixelFormat.Format32bppArgb);
            Graphics g = Graphics.FromImage(img);
            g.DrawLine(new Pen(this.currectColor, 300), new Point(0, 0), new Point(300, 300));
            pictureBox1.Image = scaled;

        }
        void update()
        {
            screen = new Bitmap(Screen.PrimaryScreen.Bounds.Width, Screen.PrimaryScreen.Bounds.Height, PixelFormat.Format32bppArgb);
            gfxScreenshot = Graphics.FromImage(screen);

            gfxScreenshot.CopyFromScreen(Screen.PrimaryScreen.Bounds.X, Screen.PrimaryScreen.Bounds.Y, 0, 0, Screen.PrimaryScreen.Bounds.Size,
                                             CopyPixelOperation.SourceCopy);
            gfxScaled = Graphics.FromImage(scaled);
            gfxScaled.DrawImage(screen, new RectangleF(0, 0, scaled.Width, scaled.Height));


            this.currectColor = ColorConverter.GetImageColor(scaled);
        }


    }
}
