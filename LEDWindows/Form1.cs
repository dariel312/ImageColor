using System;
using System.Drawing;
using System.Drawing.Imaging;
using System.Runtime.InteropServices;
using System.Text;
using System.Windows.Forms;
using System.IO.Ports;

namespace LEDController
{
    public partial class Form1 : Form
    {
        public ArduinoController controller;
        private Timer timer = new Timer();


        public Form1()
        {
            InitializeComponent();

            controller = new ArduinoController();
            controller.Start();

            timer.Interval = 33;
            timer.Tick += UpdateLabel;
            timer.Tick += UpdatePictureBox;
            timer.Start();
            
        }

        void CloseWorkerThread(object sender, EventArgs e)
        {
            controller.Stop();
        }
        void UpdateLabel(object sender, EventArgs e)
        {
            label1.Text = "Memory: " + System.Diagnostics.Process.GetCurrentProcess().WorkingSet64 / 1048576 + "mb";
        }
        void UpdatePictureBox(object sender, EventArgs e)
        {
            Bitmap img = new Bitmap(400, 400, PixelFormat.Format32bppArgb);
            Graphics g = Graphics.FromImage(img);
            g.DrawLine(new Pen(controller.CurrentColor, 300), new Point(0, 0), new Point(300, 300));
            pictureBox1.Image = img;
        }


    }
}
