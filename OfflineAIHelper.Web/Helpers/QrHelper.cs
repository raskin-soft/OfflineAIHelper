using QRCoder;
using System.Drawing;
using System.Drawing.Imaging;

namespace OfflineAIHelper.Web.Helpers
{
    public class QrHelper
    {
        public string GenerateQr(string tag, float confidence, string note, string imageUrl = null)
        {
            string content = $"Prediction: {tag}\nConfidence: {confidence:P1}\nNote: {note}";
            if (!string.IsNullOrWhiteSpace(imageUrl))
                content += $"\nImage: {imageUrl}";

            using var qrGenerator = new QRCodeGenerator();
            using var qrData = qrGenerator.CreateQrCode(content, QRCodeGenerator.ECCLevel.Q);
            using var qrCode = new QRCode(qrData);
            using var bitmap = qrCode.GetGraphic(20);

            using var ms = new MemoryStream();
            bitmap.Save(ms, ImageFormat.Png);
            string base64 = Convert.ToBase64String(ms.ToArray());

            return $"data:image/png;base64,{base64}";
        }

        public string GenerateQrFromUrl(string url)
        {
            using var qrGenerator = new QRCodeGenerator();
            using var qrData = qrGenerator.CreateQrCode(url, QRCodeGenerator.ECCLevel.Q);
            using var pngQr = new PngByteQRCode(qrData);
            byte[] qrBytes = pngQr.GetGraphic(20);

            return $"data:image/png;base64,{Convert.ToBase64String(qrBytes)}";
        }
    }
}
