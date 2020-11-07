using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using Microsoft.ML.OnnxRuntime.Tensors;
using System.Text;
using CommandLine;
using CommandLine.Text;
using Microsoft.ML.OnnxRuntime;
using Serilog;
using Serilog.Core;
using NAudio.Wave;
using NAudio.Wave.SampleProviders;


namespace speech_recog_offline
{
    class Program
    {
        class Options
        {
            [Option('m', "model", Required = true,
              HelpText = "Model file to be used.")]
            public string ModelFile { get; set; }

            [Option('i', "input", Required = true,
              HelpText = "Input file to be processed.")]
            public string InputFile { get; set; }

            [Option('v', "verbose", Default = 1,
              HelpText = "Prints all messages to standard output.")]
            public int Verbose { get; set; }

            [Option("samplerate", Default = 16000)]
            public float Samplerate { get; set; }
            
            [Option('h', "help")]
            public bool Help { get; set; }
        }

        static void Main(string[] args)
        {
            // Read Arguments
            var parseResult = Parser.Default.ParseArguments<Options>(args);
            Options opts = null;
            switch (parseResult.Tag)
            {
                case ParserResultType.Parsed:
                    var parsed = parseResult as Parsed<Options>;
                    opts = parsed.Value;
                    break;
                case ParserResultType.NotParsed:
                    var notParsed = parseResult as NotParsed<Options>;
                    break;
            }

            // Logger
            int Verbose = opts.Verbose;
            var levelSwitch = new LoggingLevelSwitch();
            switch (Verbose)
            {
                case 0:
                    levelSwitch.MinimumLevel = Serilog.Events.LogEventLevel.Debug;
                    break;
                case 1:
                    levelSwitch.MinimumLevel = Serilog.Events.LogEventLevel.Information;
                    break;
                default:
                    levelSwitch.MinimumLevel = Serilog.Events.LogEventLevel.Warning;
                    break;
            }
            Log.Logger = new LoggerConfiguration()
                .MinimumLevel.ControlledBy(levelSwitch)
                .WriteTo.Console()
                .CreateLogger();
            Log.Warning($"Level of Logging: {levelSwitch.MinimumLevel}");

            // Read paths
            string modelFilePath = opts.ModelFile;
            string audioFilePath = opts.InputFile;
            string FileName = "";
            string Format = "";

            if (audioFilePath.Contains(":"))
            {
                FileName = audioFilePath.Substring(audioFilePath.IndexOf(':') + 1);
                Format = audioFilePath.Split(':')[0];
                Log.Information($"opt.Format = {Format}");
            }
            else
            {
                Log.Error("ERROR: Input file should follow format scp:<list_of_files> or wav:<filename.wav>");
                System.Environment.Exit(1);
            }

            // Print Options
            Log.Debug($"opt.ModelFile = {opts.ModelFile}");
            Log.Debug($"opt.InputFile = {opts.InputFile}");
            Log.Debug($"opt.Samplerate = {opts.Samplerate}");

            // Read Input
            float[] ThisInput = GetWav(FileName);
            Log.Debug($"Length of input = {ThisInput.Length}");

            // Run Inference
            using var session = new InferenceSession(modelFilePath);


        }

        static float[] GetWav(string ThisFile)
        {
            float[] ThisFloatBuffer;
            AudioFileReader ThisReader = new AudioFileReader(ThisFile);
            ISampleProvider ThisProvider = ThisReader.ToSampleProvider();
            ThisFloatBuffer = new float[ThisReader.Length / 2];
            ThisProvider.Read(ThisFloatBuffer, 0, ThisFloatBuffer.Length);
            return ThisFloatBuffer;
        }
    }
}
