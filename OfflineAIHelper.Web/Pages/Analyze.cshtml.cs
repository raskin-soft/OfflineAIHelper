using Microsoft.AspNetCore.Mvc;
using Microsoft.AspNetCore.Mvc.RazorPages;
using Microsoft.ML;
using OfflineAIHelper.Web.Helper;
using OfflineAIHelper.Web.Helpers;
using OfflineAIHelper.Web.Models;
using PdfSharpCore.Drawing;
using PdfSharpCore.Pdf;
using QuestPDF.Infrastructure;
using SixLabors.Fonts.Tables.AdvancedTypographic;
using System.IO;
using System.Net;
using System.Net.Sockets;
using System.Text.Json;
using Tensorflow.Keras.Engine;

public class AnalyzeModel : PageModel
{

    private readonly IWebHostEnvironment _env;

    public AnalyzeModel(IWebHostEnvironment env)
    {
        _env = env;
    }



    [BindProperty]
    public IFormFile UploadedImage { get; set; }

    [BindProperty]
    public IFormFileCollection BatchImages { get; set; }

    public string ImagePath { get; set; }
    public string PredictionResult { get; set; }
    public List<float> ConfidenceScores { get; set; } = new();
    public List<BatchResultModel> BatchResults { get; set; } = new();
    public string ActiveTab { get; set; } = "single";

    [TempData]
    public string BatchResultsJson { get; set; }

    public string HighlightedImagePath { get; set; }

    public string MarkedImagePath { get; set; }

    [BindProperty]
    public string HumanNote { get; set; }

    public string QrCodeImageUrl { get; set; }

    public string PredictedTag { get; set; }
    public float ConfidenceValue { get; set; }

    public string SavePdfAndGenerateQr(byte[] pdfBytes)
    {
        var filename = $"DamageReport_{DateTime.Now:yyyyMMddHHmmss}.pdf";
        var exportDir = Path.Combine(_env.WebRootPath, "exports");
        Directory.CreateDirectory(exportDir);

        var pdfPath = Path.Combine(exportDir, filename);
        System.IO.File.WriteAllBytes(pdfPath, pdfBytes);

        //var localIp = HttpContext.Connection.LocalIpAddress?.ToString() ?? "127.0.0.1";
        //var port = Request.Host.Port ?? 5000;
        //var pdfUrl = $"http://{localIp}:{port}/exports/{filename}";

        var ip = GetLocalIp();
        var port = Request.Host.Port ?? 5000;
        var pdfUrl = $"http://{ip}:{port}/exports/{filename}";

        var qrGen = new QrHelper(); // Your helper class
        return qrGen.GenerateQrFromUrl(pdfUrl);
    }

    public string GetLocalIp()
    {
        var host = Dns.GetHostEntry(Dns.GetHostName());
        foreach (var ip in host.AddressList)
        {
            if (ip.AddressFamily == AddressFamily.InterNetwork)
                return ip.ToString();
        }
        return "127.0.0.1";
    }

    public void OnPost()
    {
        ActiveTab = "single";

        if (UploadedImage != null)
        {
            var fileName = Path.GetFileName(UploadedImage.FileName);
            var savePath = Path.Combine("wwwroot/uploads", fileName);
            Directory.CreateDirectory("wwwroot/uploads");

            //using var fs = new FileStream(savePath, FileMode.Create);
            //UploadedImage.CopyTo(fs);

            using (var fs = new FileStream(savePath, FileMode.Create))
            {
                UploadedImage.CopyTo(fs);
            }

            ImagePath = "/uploads/" + fileName;

            var modelPath = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "Models", "squeezenet1_1_Opset18.onnx");
            var analyzer = new ImageAnalyzer(modelPath);
            PredictionResult = analyzer.Predict(savePath);

            PredictedTag = PredictionResult.Split('|')[0].Split(':')[1].Trim();         // e.g. "Severe"
            ConfidenceValue = float.Parse(PredictionResult.Split('|')[1].Split(':')[1].Replace("%", "").Trim()) / 100f; // e.g. 0.92f
            //HumanNote = humanNote;
            HighlightedImagePath = analyzer.GenerateDamageOverlay(savePath);
            MarkedImagePath = analyzer.MarkDamageAreas(savePath);

            var qrHelper = new QrHelper();
            QrCodeImageUrl = qrHelper.GenerateQr(PredictedTag, ConfidenceValue, "", MarkedImagePath);
        }
    }

    public void OnPostBatch()
    {
        ActiveTab = "batch";

        if (BatchImages != null && BatchImages.Count > 0)
        {
            var modelPath = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "Models", "squeezenet1_1_Opset18.onnx");
            var analyzer = new ImageAnalyzer(modelPath);
            Directory.CreateDirectory("wwwroot/uploads");

            foreach (var image in BatchImages)
            {
                var fileName = Path.GetFileName(image.FileName);
                var savePath = Path.Combine("wwwroot/uploads", fileName);

                //using var fs = new FileStream(savePath, FileMode.Create);
                //image.CopyTo(fs);

                using (var fs = new FileStream(savePath, FileMode.Create))
                {
                    image.CopyTo(fs);
                }

                var prediction = analyzer.Predict(savePath);

                var markedImagePath = analyzer.MarkDamageAreas(savePath);

                HighlightedImagePath = analyzer.GenerateDamageOverlay(savePath);
                MarkedImagePath = analyzer.MarkDamageAreas(savePath);

                // Parse result (Tag + Confidence)
                var parts = prediction.Split('|');
                var tag = parts[0].Trim().Replace("Tag: ", "");
                var confidence = parts.Length > 1 ? parts[1].Trim().Replace("Confidence: ", "") : "N/A";

                BatchResults.Add(new BatchResultModel
                {
                    ImageName = fileName,
                    ImagePath = savePath,
                    ImageUrl = "/uploads/" + fileName,
                    MarkedImageUrl = markedImagePath,
                    Tag = tag,
                    Confidence = confidence
                });
            }

            TempData["BatchResultsJson"] = JsonSerializer.Serialize(BatchResults);
            TempData.Keep("BatchResultsJson");
        }
    }

    public void OnGet(string selectedImage)
    {
        ActiveTab = "single";

        if (!string.IsNullOrEmpty(selectedImage))
        {
            var relativePath = selectedImage.TrimStart('/');
            var fullPath = Path.Combine("wwwroot", relativePath);

            ImagePath = "/" + relativePath;
            var modelPath = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "Models", "squeezenet1_1_Opset18.onnx");
            var analyzer = new ImageAnalyzer(modelPath);
            PredictionResult = analyzer.Predict(fullPath);

            HighlightedImagePath = analyzer.GenerateDamageOverlay(fullPath);
            MarkedImagePath = analyzer.MarkDamageAreas(fullPath);

            var qrHelper = new QrHelper();
            QrCodeImageUrl = qrHelper.GenerateQr(PredictedTag, ConfidenceValue, "", MarkedImagePath);
        }

        // 🔁 Restore BatchResults
        if (TempData["BatchResultsJson"] is string json)
        {
            BatchResults = JsonSerializer.Deserialize<List<BatchResultModel>>(json);
            TempData.Keep("BatchResultsJson");
        }
    }



    // Optional PDF Export for single analysis
    public IActionResult OnPostExportSingle(string tag, float confidence, string markedImagePath)
    {
        QuestPDF.Settings.License = LicenseType.Community; // Or LicenseType.Commercial based on your organization's revenue.

        var note = HumanNote; // ✅ Already bound from form

        var helper = new ReportHelper();
        var pdfBytes = helper.GenerateSingleTabPdf(tag, confidence, markedImagePath, note);
        return File(pdfBytes, "application/pdf", "SingleTabReport.pdf");
    }


    //public IActionResult OnPostExportSingle(string tag, float confidence, string markedImagePath)
    //{
    //    QuestPDF.Settings.License = LicenseType.Community; // Or LicenseType.Commercial based on your organization's revenue.

    //    var helper = new ReportHelper();

    //    // 🧠 Step 1: Generate PDF
    //    var pdfBytes = helper.GenerateSingleTabPdf(tag, confidence, markedImagePath, "");

    //    // 🧠 Step 2: Save PDF & generate QR
    //    QrCodeImageUrl = SavePdfAndGenerateQr(pdfBytes); // ✅ Here's the call

    //    // ✅ Optional: Persist values if needed for display
    //    PredictedTag = tag;
    //    ConfidenceValue = confidence;
    //    MarkedImagePath = markedImagePath;

    //    return Page(); // ⛔️ Avoid redirecting to keep QR visible
    //}


}