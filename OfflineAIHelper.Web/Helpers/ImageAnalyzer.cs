using Microsoft.ML;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using OfflineAIHelper.Web.Models;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp.Processing;
using SixLabors.ImageSharp.Drawing.Processing;
using System.IO;
using Tensorflow.Keras.Engine;

namespace OfflineAIHelper.Web.Helpers
{
    public class ImageAnalyzer
    {
        private readonly InferenceSession session;

        public ImageAnalyzer(string modelPath)
        {
            session = new InferenceSession(modelPath);

            //foreach (var input in session.InputMetadata)
            //{
            //    Console.WriteLine($"Input name: {input.Key}");
            //}
        }

        public string Predict(string imagePath)
        {
            using var image = Image.Load<Rgb24>(imagePath);
            image.Mutate(x => x.Resize(224, 224));

            var input = new DenseTensor<float>(new[] { 1, 3, 224, 224 });

            for (int y = 0; y < 224; y++)
            {
                for (int x = 0; x < 224; x++)
                {
                    var pixel = image[x, y];
                    input[0, 0, y, x] = pixel.R / 255f;
                    input[0, 1, y, x] = pixel.G / 255f;
                    input[0, 2, y, x] = pixel.B / 255f;
                }
            }

            var inputs = new List<NamedOnnxValue> {
            NamedOnnxValue.CreateFromTensor("x", input)

        };

            using var results = session.Run(inputs);
            //var output = results.First().AsEnumerable<float>().ToArray();
            //int maxIndex = Array.IndexOf(output, output.Max());
            //float confidence = output[maxIndex];

            var output = results.First().AsEnumerable<float>().ToArray();
            var probabilities = ApplySoftmax(output);

            int maxIndex = Array.IndexOf(probabilities, probabilities.Max());
            float confidence = probabilities[maxIndex];

            string[] labels = { "Minor", "Moderate", "Severe", "Unknown" };
            string tag = labels[maxIndex % labels.Length];

            return $"Tag: {tag} | Confidence: {confidence:P1}";
        }



        public ImageAnalysisResult AnalyzeImage(string imagePath)
        {
            using var image = Image.Load<Rgb24>(imagePath);
            image.Mutate(x => x.Resize(224, 224));

            var input = new DenseTensor<float>(new[] { 1, 3, 224, 224 });

            for (int y = 0; y < 224; y++)
            {
                for (int x = 0; x < 224; x++)
                {
                    var pixel = image[x, y];
                    input[0, 0, y, x] = pixel.R / 255f;
                    input[0, 1, y, x] = pixel.G / 255f;
                    input[0, 2, y, x] = pixel.B / 255f;
                }
            }

            var inputs = new List<NamedOnnxValue>
    {
        NamedOnnxValue.CreateFromTensor("x", input)
    };

            using var results = session.Run(inputs);
            var output = results.First().AsEnumerable<float>().ToArray();
            var probabilities = ApplySoftmax(output);

            int maxIndex = Array.IndexOf(probabilities, probabilities.Max());
            float confidence = probabilities[maxIndex];

            string[] labels = { "Minor", "Moderate", "Severe", "Unknown" };
            string tag = labels[maxIndex % labels.Length];

            return new ImageAnalysisResult
            {
                Tag = tag,
                Confidence = confidence,
                Probabilities = probabilities
            };
        }


        public string GenerateDamageOverlay(string imagePath)
        {
            using var original = Image.Load<Rgba32>(imagePath);
            original.Mutate(ctx => ctx.Resize(224, 224)); // Resize to match prediction dimensions

            var overlay = new Image<Rgba32>(original.Width, original.Height);

            var rand = new Random();
            for (int y = 0; y < original.Height; y++)
            {
                for (int x = 0; x < original.Width; x++)
                {
                    var pixel = original[x, y];
                    var intensity = (pixel.R + pixel.G + pixel.B) / 3;

                    // Heuristic: brighten low-texture zones as fake "damage" highlight
                    if (intensity < 80 && rand.NextDouble() < 0.2)
                    {
                        overlay[x, y] = new Rgba32(255, 0, 0, 100); // semi-transparent red
                    }
                    else
                    {
                        overlay[x, y] = new Rgba32(0, 0, 0, 0); // transparent
                    }
                }
            }

            original.Mutate(ctx => ctx.DrawImage(overlay, PixelColorBlendingMode.Overlay, PixelAlphaCompositionMode.SrcOver, 1f));

            var highlightedPath = Path.Combine("wwwroot/uploads", "highlighted_" + Path.GetFileName(imagePath));
            original.Save(highlightedPath);

            return "/uploads/" + Path.GetFileName(highlightedPath);
        }

        public string MarkDamageAreas(string imagePath)
        {
            using var original = Image.Load<Rgba32>(imagePath);

            var pen = Pens.Solid(Color.Red, 2);  // 🔴 Red border
            var rand = new Random();

            int gridSize = 20;

            for (int y = 10; y < original.Height - 10; y += gridSize)
            {
                for (int x = 10; x < original.Width - 10; x += gridSize)
                {
                    var pixel = original[x, y];
                    var brightness = (pixel.R + pixel.G + pixel.B) / 3;

                    // Heuristic: low brightness implies potential rubble/shadow/collapse
                    if (brightness < 70 && rand.NextDouble() < 0.4)
                    {
                        var box = new RectangleF(x - 8, y - 8, gridSize, gridSize);
                        original.Mutate(ctx => ctx.Draw(pen, box));
                    }
                }
            }

            // 🔄 Save marked image with new filename
            var markedPath = Path.Combine("wwwroot/uploads", "marked_" + Path.GetFileName(imagePath));
            original.Save(markedPath);

            return "/uploads/" + Path.GetFileName(markedPath);
        }


        private float[] ApplySoftmax(float[] scores)
        {
            var expScores = scores.Select(x => Math.Exp((double)x)).ToArray();
            double sumExp = expScores.Sum();
            return expScores.Select(x => (float)(x / sumExp)).ToArray();
        }

    }
}
