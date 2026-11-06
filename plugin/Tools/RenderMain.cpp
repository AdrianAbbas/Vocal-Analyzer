// VocalMatchRender: run audio through the plug-in's own AudioProcessor, offline.
// The Python-vs-C++ check (tests/parity) feeds it the same preset and input as the Python
// reference chain and compares the outputs.
//
//   VocalMatchRender --preset preset.json --in dry.wav --out out.wav
//                    [--match 100] [--tail 0] [--block 512] [--param id=value ...]
//   VocalMatchRender [--preset preset.json] --snapshot editor.png   (the editor as an image)
//
// The preset goes through the plug-in's "Load preset" code (loadPresetFile), then each
// --param sets one parameter in real units. Output: 32-bit float WAV, stereo.

#include "PluginProcessor.h"

#include <juce_audio_formats/juce_audio_formats.h>

#include <iostream>

namespace
{
int fail (const juce::String& message)
{
    std::cerr << "VocalMatchRender: " << message << std::endl;
    return 1;
}
} // namespace

int main (int argc, char* argv[])
{
    juce::ScopedJuceInitialiser_GUI juceInit;
    juce::File presetFile, inFile, outFile;
    double matchPct = -1.0, tailSeconds = 0.0;
    int block = 512;
    juce::StringArray extraParams;
    bool listParams = false;
    juce::File snapshotFile;

    for (int i = 1; i < argc; ++i)
    {
        const juce::String arg (argv[i]);
        auto next = [&]() -> juce::String { return i + 1 < argc ? juce::String (argv[++i]) : juce::String(); };
        if (arg == "--preset") presetFile = juce::File::getCurrentWorkingDirectory().getChildFile (next());
        else if (arg == "--in") inFile = juce::File::getCurrentWorkingDirectory().getChildFile (next());
        else if (arg == "--out") outFile = juce::File::getCurrentWorkingDirectory().getChildFile (next());
        else if (arg == "--match") matchPct = next().getDoubleValue();
        else if (arg == "--tail") tailSeconds = next().getDoubleValue();
        else if (arg == "--block") block = next().getIntValue();
        else if (arg == "--param") extraParams.add (next());
        else if (arg == "--list-params") listParams = true;
        else if (arg == "--snapshot") snapshotFile = juce::File::getCurrentWorkingDirectory().getChildFile (next());
        else return fail ("unknown argument " + arg);
    }

    VocalMatchProcessor processor;
    if (listParams)
    {
        for (auto* p : processor.getParameters())
            if (auto* withId = dynamic_cast<juce::AudioProcessorParameterWithID*> (p))
                std::cout << withId->getParameterID() << "\t" << p->getName (64) << "\t" << p->getText (p->getDefaultValue(), 32) << "\n";
        return 0;
    }
    if (inFile == juce::File() && snapshotFile == juce::File())
        return fail ("usage: --in dry.wav --out out.wav [--preset preset.json] [--match pct] [--tail s] [--block n] [--param id=value]");

    if (presetFile != juce::File())
        if (auto r = processor.loadPresetFile (presetFile); r.failed())
            return fail ("preset: " + r.getErrorMessage());
    if (matchPct >= 0.0)
        processor.setParameterValue (vm::params::matchAmount, (float) matchPct);
    for (const auto& kv : extraParams)
    {
        const auto id = kv.upToFirstOccurrenceOf ("=", false, false);
        if (processor.apvts.getParameter (id) == nullptr)
            return fail ("no parameter " + id);
        processor.setParameterValue (id, kv.fromFirstOccurrenceOf ("=", false, false).getFloatValue());
    }

    if (snapshotFile != juce::File())
    {
        // The editor as a PNG (after --preset, so the estimate markers show).
        std::unique_ptr<juce::AudioProcessorEditor> editor (processor.createEditor());
        const auto image = editor->createComponentSnapshot (editor->getLocalBounds(), true, 2.0f);
        snapshotFile.deleteFile();
        juce::FileOutputStream stream (snapshotFile);
        juce::PNGImageFormat png;
        if (! stream.openedOk() || ! png.writeImageToStream (image, stream))
            return fail ("cannot write " + snapshotFile.getFullPathName());
        std::cout << "wrote " << snapshotFile.getFullPathName() << " (" << editor->getWidth() << " x " << editor->getHeight() << ")\n";
        if (inFile == juce::File())
            return 0;
    }
    if (outFile == juce::File() || block < 1)
        return fail ("--out and a positive --block are needed to render");

    juce::AudioFormatManager formats;
    formats.registerBasicFormats();
    std::unique_ptr<juce::AudioFormatReader> reader (formats.createReaderFor (inFile));
    if (reader == nullptr)
        return fail ("cannot read " + inFile.getFullPathName());
    const double fs = reader->sampleRate;
    const int inLength = (int) reader->lengthInSamples;
    const int length = inLength + (int) std::lround (tailSeconds * fs);
    juce::AudioBuffer<float> audio (2, length);
    audio.clear();
    reader->read (&audio, 0, inLength, 0, true, true);
    if (reader->numChannels == 1)
        audio.copyFrom (1, 0, audio, 0, 0, inLength);

    processor.setPlayConfigDetails (2, 2, fs, block);
    processor.prepareToPlay (fs, block);
    juce::MidiBuffer midi;
    for (int start = 0; start < length; start += block)
    {
        const int n = std::min (block, length - start);
        float* channels[] = { audio.getWritePointer (0, start), audio.getWritePointer (1, start) };
        juce::AudioBuffer<float> view (channels, 2, n);
        processor.processBlock (view, midi);
    }
    processor.releaseResources();

    outFile.deleteFile();
    juce::WavAudioFormat wav;
    std::unique_ptr<juce::OutputStream> stream (outFile.createOutputStream());
    if (stream == nullptr)
        return fail ("cannot write " + outFile.getFullPathName());
    std::unique_ptr<juce::AudioFormatWriter> writer (wav.createWriterFor (stream, juce::AudioFormatWriterOptions()
                                                                                       .withSampleRate (fs)
                                                                                       .withNumChannels (2)
                                                                                       .withBitsPerSample (32)
                                                                                       .withSampleFormat (juce::AudioFormatWriterOptions::SampleFormat::floatingPoint)));
    if (writer == nullptr)
        return fail ("cannot create a WAV writer");
    writer->writeFromAudioSampleBuffer (audio, 0, length);
    std::cout << "wrote " << outFile.getFullPathName() << " (" << length << " frames, " << fs << " Hz, block " << block << ")\n";
    return 0;
}
