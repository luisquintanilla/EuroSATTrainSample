using Microsoft.ML;
using Microsoft.ML.Vision;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Reflection.Emit;
using System.Timers;
using Tensorflow.Operations.Activation;

namespace EuroSATTrainSample
{
    class Program
    {
        private static string TRAIN_DATA_FILEPATH = @"C:\Users\luquinta.REDMOND\Desktop\traindata.tsv";

        private static MLContext mlContext = new MLContext(seed: 1);

        static void Main(string[] args)
        {
            var projectDirectory = Path.GetFullPath(Path.Combine(AppContext.BaseDirectory, "../../../"));
            var workspaceRelativePath = Path.Combine(projectDirectory, "workspace");

            // Load the data from the file
            var file =
                File.ReadAllLines(TRAIN_DATA_FILEPATH)
                .Select(line =>
                {
                    var columns = line.Split('\t');
                    return new ImageInput { Label = columns[0], ImagePath = columns[1] };
                })
                .GroupBy(image => image.Label)
                .Select(category =>
                {
                    return new Datasets { TrainSet = category.Take(1800), TestSet = category.Skip(1800).Take(200) };
                });

            // Create dataviews
            IDataView trainData =
                mlContext.Data.LoadFromEnumerable(file.SelectMany(x => x.TrainSet));
            IDataView testData = 
                mlContext.Data.LoadFromEnumerable(file.SelectMany(x => x.TestSet));

            // Shuffle the data
            IDataView shuffledTrainData =
                mlContext.Data.ShuffleRows(trainData);

            IDataView shuffledTestData =
                mlContext.Data.ShuffleRows(testData);

            // Create data preprocessing pipeline
            var dataProcessPipeline = mlContext.Transforms.Conversion.MapValueToKey("Label", "Label")
                          .Append(mlContext.Transforms.LoadRawImageBytes("ImageSource_featurized", null, "ImagePath"))
                          .Append(mlContext.Transforms.CopyColumns("Features", "ImageSource_featurized"));


            // Create trainer
            var trainer = mlContext.MulticlassClassification.Trainers.ImageClassification(new ImageClassificationTrainer.Options() { 
                LabelColumnName = "Label", 
                FeatureColumnName = "Features", 
                WorkspacePath=workspaceRelativePath, 
                MetricsCallback=(metrics) => Console.WriteLine(metrics), 
                ReuseTrainSetBottleneckCachedValues=true})
                .Append(mlContext.Transforms.Conversion.MapKeyToValue("PredictedLabel", "PredictedLabel"));

            // Merge data preprocessing and trainer pipelines
            var trainingPipeline = dataProcessPipeline.Append(trainer);
            
            var watch = Stopwatch.StartNew();

            Console.WriteLine("Training Model");
            // Create the model
            var model = trainingPipeline.Fit(shuffledTrainData);
            watch.Stop();
            Console.WriteLine($"Finished training in {watch.ElapsedMilliseconds}");

            watch.Start();
            Console.WriteLine("Evaluating Model");
            // Evaluate the model
            var evaluation = mlContext.MulticlassClassification.Evaluate(model.Transform(shuffledTestData));
            watch.Stop();
            Console.WriteLine($"Finished evaluation in {watch.ElapsedMilliseconds}\n");
            Console.WriteLine("Evaluation Metrics");
            Console.WriteLine($"Log Loss: {evaluation.LogLoss} | MacroAccuracy: {evaluation.MacroAccuracy}");

        }
    }

    class ImageInput
    {
        public string ImagePath { get; set; }
        public string Label { get; set; }
    }

    class Datasets
    {
        public IEnumerable<ImageInput> TrainSet { get; set; }
        public IEnumerable<ImageInput> TestSet { get; set; }
    }
}
