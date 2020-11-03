using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using Microsoft.ML.OnnxRuntime.Tensors;
using System.Text;
using CommandLine;
using CommandLine.Text;


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

            //[Option('o', "output", Required = true,
            //  HelpText = "Output file to be processed.")]
            //public string OutputFile { get; set; }

            [Option('v', "verbose", Default = true,
              HelpText = "Prints all messages to standard output.")]
            public bool Verbose { get; set; }

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

            // Read paths
            string modelFilePath = opts.ModelFile;
            string audioFilePath = opts.InputFile;

            // Preprocess image
            var paddedHeight = (int)(Math.Ceiling(120 / 32f) * 32f);
            var paddedWidth = (int)(Math.Ceiling(120 / 32f) * 32f);

            Console.WriteLine($"opt.A = {opts.ModelFile}");
            Console.WriteLine($"opt.A = {opts.InputFile}");
            Console.WriteLine($"opt.A = {opts.Samplerate}");
        }
        static void RunOptions(Options opts)
        {
            //handle options
        }
        static void HandleParseError(IEnumerable<Error> errs)
        {
            //handle errors
        }
    }
}
