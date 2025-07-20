using Microsoft.AspNetCore.Mvc;
using Microsoft.ML.Data;

namespace OfflineAIHelper.Web.Models
{
    public class BatchResultModel
    {
        public string ImageUrl { get; set; }
        public string ImageName { get; set; }
        public string ImagePath { get; set; }
        public string Tag { get; set; }
        public string Confidence { get; set; }
        public List<float> Probabilities { get; set; } // Added property to fix CS0117  
        public string MarkedImageUrl { get; set; }
    }

    public class ImageInput
    {
        [ColumnName("Image")]
        public byte[] Image { get; set; }
    }

    public class ImageOutput
    {
        public float[] Score { get; set; } // Contains class confidences
    }

    public class ImageAnalysisResult
    {
        public string Tag { get; set; }
        public float Confidence { get; set; }
        public float[] Probabilities { get; set; }
    }
}
