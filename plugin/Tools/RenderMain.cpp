// VocalMatchRender: run audio through the plug-in's own AudioProcessor, offline.
// The Python-vs-C++ check (tests/parity) feeds it the same preset and input as the Python
// reference chain and compares the outputs.
//
//   VocalMatchRender --preset preset.json --in dry.wav --out out.wav
//                    [--match 100] [--tail 0] [--block 512] [--param id=value ...]
//   VocalMatchRender [--preset preset.json] --snapshot editor.png   (the editor as an image)
//                    [--size 1040x660] [--page analyze|chain] [--reference song.wav] [--scan singers.json]
//                    [--regions a-b,c-d] [--singers A,B] [--collapse eq,delay]
//   VocalMatchRender ... --check-layout   (every component inside its parent, no overlaps; exit 1 if not)
//   ... --preview song.wav START END     (plays that span of the song through the output while rendering)
//   VocalMatchRender --analyze ref.wav dry.wav [--analyze-out p.json] [--analyze-refine N] [--analyze-passes N]
//                    (the plug-in's Analyze: runs the installed analyzer, loads its preset)
//                    [--analyze-singers A,B] [--analyze-regions a-b,c-d] [--analyze-per-singer]
//                    [--analyze-isolated-vocal]  (the reference is a vocal stem: no source separation;
//                    without it the song is separated, as the plug-in does by default)
//                    On failure: the plug-in's message, then "log: <the run's full log>".
//   VocalMatchRender --find-singers song.wav   (the Analyze panel's Find singers: runs the analyzer's scan)
//   ... --capture take.wav   (arms the plug-in's capture while rendering, writes the take)
//
// The preset goes through the plug-in's "Load preset" code (loadPresetFile), then each
// --param sets one parameter in real units. Output: 32-bit float WAV, stereo.

#include "PluginEditor.h"
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
    juce::File snapshotFile, analyzeRef, analyzeDry, analyzeOut, captureFile, referenceSong, scanFile, previewSong;
    juce::String sizeText, pageText, regionsText, singersText, collapseText;
    bool checkLayoutFlag = false, analyzePerSinger = false, analyzeIsolatedVocal = false;
    juce::String analyzeSingers, analyzeRegions;
    juce::File findSingersSong;
    double previewStart = 0.0, previewEnd = 0.0;
    int analyzeRefine = 60, analyzePasses = 2;

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
        else if (arg == "--analyze")
        {
            analyzeRef = juce::File::getCurrentWorkingDirectory().getChildFile (next());
            analyzeDry = juce::File::getCurrentWorkingDirectory().getChildFile (next());
        }
        else if (arg == "--analyze-out") analyzeOut = juce::File::getCurrentWorkingDirectory().getChildFile (next());
        else if (arg == "--analyze-refine") analyzeRefine = next().getIntValue();
        else if (arg == "--analyze-passes") analyzePasses = next().getIntValue();
        else if (arg == "--capture") captureFile = juce::File::getCurrentWorkingDirectory().getChildFile (next());
        else if (arg == "--snapshot") snapshotFile = juce::File::getCurrentWorkingDirectory().getChildFile (next());
        else if (arg == "--size") sizeText = next();
        else if (arg == "--page") pageText = next();
        else if (arg == "--reference") referenceSong = juce::File::getCurrentWorkingDirectory().getChildFile (next());
        else if (arg == "--scan") scanFile = juce::File::getCurrentWorkingDirectory().getChildFile (next());
        else if (arg == "--regions") regionsText = next();
        else if (arg == "--singers") singersText = next();
        else if (arg == "--collapse") collapseText = next();
        else if (arg == "--check-layout") checkLayoutFlag = true;
        else if (arg == "--analyze-singers") analyzeSingers = next();
        else if (arg == "--analyze-regions") analyzeRegions = next();
        else if (arg == "--analyze-per-singer") analyzePerSinger = true;
        else if (arg == "--analyze-isolated-vocal") analyzeIsolatedVocal = true;
        else if (arg == "--find-singers") findSingersSong = juce::File::getCurrentWorkingDirectory().getChildFile (next());
        else if (arg == "--preview")
        {
            previewSong = juce::File::getCurrentWorkingDirectory().getChildFile (next());
            previewStart = next().getDoubleValue();
            previewEnd = next().getDoubleValue();
        }
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
    if (findSingersSong != juce::File())
    {
        // Find singers, as the Analyze panel runs it (the same AnalyzerRun and scan loading).
        processor.referenceFile = findSingersSong;
        vm::ScanRequest scanRequest;
        scanRequest.reference = findSingersSong;
        scanRequest.outJson = processor.scanFileFor (findSingersSong);
        scanRequest.separate = "no";
        if (! processor.scanner.start (scanRequest))
            return fail ("find singers: " + processor.scanner.getError());
        while (! processor.scanner.waitUntilDone (200)) {}
        if (processor.scanner.getState() != vm::AnalyzerRun::State::finished)
            return fail ("find singers: " + processor.scanner.getError() + "\n" + processor.scanner.getLog());
        if (! processor.loadScanFile (processor.scanner.getPresetFile()))
            return fail ("find singers: cannot read " + processor.scanner.getPresetFile().getFullPathName());
        const auto& scan = processor.getScan();
        std::cout << "scan: " << scan.message << "\n";
        for (const auto& singer : scan.singers)
            std::cout << "singer " << singer.id << " " << singer.seconds << " s, preview " << singer.preview.getStart() << "-"
                      << singer.preview.getEnd() << "\n";
        if (analyzeRef == juce::File() && inFile == juce::File() && snapshotFile == juce::File())
            return 0;
    }
    if (analyzeRef != juce::File())
    {
        // The plug-in's Analyze button, without the button: the same AnalyzerRun, then the
        // same preset loading.
        processor.referenceFile = analyzeRef;
        processor.dryFile = analyzeDry;
        processor.useCapturedDry = false;
        processor.referenceIsVocal = analyzeIsolatedVocal;
        vm::AnalyzeRequest request;
        if (auto r = processor.makeAnalyzeRequest (request); r.failed())
            return fail ("analyze: " + r.getErrorMessage());
        if (analyzeOut != juce::File())
            request.outPreset = analyzeOut;
        request.refineEvals = analyzeRefine;
        request.passes = analyzePasses;
        for (const auto& r : juce::StringArray::fromTokens (analyzeRegions, ",", {}))
            if (r.contains ("-"))
                request.regions.add ({ r.upToFirstOccurrenceOf ("-", false, false).getDoubleValue(),
                                       r.fromFirstOccurrenceOf ("-", false, false).getDoubleValue() });
        if (analyzeSingers.isNotEmpty())
            request.singers = juce::StringArray::fromTokens (analyzeSingers, ",", {});
        request.perSinger = analyzePerSinger;
        std::cout << "command: " << vm::AnalyzerRun::shellCommandFor (request) << "\n";
        std::cout << "analyzer: " << vm::findAnalyzer().getFullPathName() << "\n";
        if (! processor.analyzer.start (request))
            return fail ("analyze: " + processor.analyzer.getError());
        juce::String lastStage;
        while (! processor.analyzer.waitUntilDone (200))
            if (const auto st = processor.analyzer.getStage(); st != lastStage)
            {
                std::cout << "  " << juce::String (processor.analyzer.getProgress(), 3) << " " << st << "\n";
                lastStage = st;
            }
        if (const auto st = processor.analyzer.getStage(); st != lastStage) // the last stage, if it came at the end
            std::cout << "  " << juce::String (processor.analyzer.getProgress(), 3) << " " << st << "\n";
        if (processor.analyzer.getState() != vm::AnalyzerRun::State::finished)
            return fail ("analyze: " + processor.analyzer.getError() + "\nlog: " + processor.analyzer.getLogFile().getFullPathName()
                         + "\n" + processor.analyzer.getLog());
        std::cout << "log: " << processor.analyzer.getLogFile().getFullPathName() << "\n";
        if (auto r = processor.loadPresetFile (processor.analyzer.getPresetFile()); r.failed())
            return fail ("analyze: " + r.getErrorMessage());
        std::cout << "analyzed: " << processor.analyzer.getPresetFile().getFullPathName() << "\n";
        if (inFile == juce::File() && snapshotFile == juce::File())
            return 0;
    }
    if (inFile == juce::File() && snapshotFile == juce::File() && ! checkLayoutFlag)
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

    if (snapshotFile != juce::File() || checkLayoutFlag)
    {
        // The editor's state as a host would restore it, then the editor itself.
        if (referenceSong != juce::File())
        {
            processor.referenceFile = referenceSong;
            const int gen = processor.getSongGeneration();
            processor.loadReferenceSong();
            for (int i = 0; i < 600 && processor.getSongGeneration() == gen; ++i)
            {
                juce::Thread::sleep (50);
                processor.pollSong();
            }
        }
        if (scanFile != juce::File() && ! processor.loadScanFile (scanFile))
            return fail ("not a singer scan of the reference: " + scanFile.getFullPathName());
        for (const auto& r : juce::StringArray::fromTokens (regionsText, ",", {}))
            if (r.contains ("-"))
                processor.regions.add ({ r.upToFirstOccurrenceOf ("-", false, false).getDoubleValue(),
                                         r.fromFirstOccurrenceOf ("-", false, false).getDoubleValue() });
        if (singersText.isNotEmpty())
            processor.chosenSingers = juce::StringArray::fromTokens (singersText, ",", {});
        if (collapseText.isNotEmpty())
            processor.collapsedCards = juce::StringArray::fromTokens (collapseText, ",", {});
        if (pageText.isNotEmpty())
            processor.editorPage = pageText == "chain" ? 1 : 0;
        if (sizeText.isNotEmpty())
        {
            processor.editorWidth = sizeText.upToFirstOccurrenceOf ("x", false, false).getIntValue();
            processor.editorHeight = sizeText.fromFirstOccurrenceOf ("x", false, false).getIntValue();
        }
        std::unique_ptr<juce::AudioProcessorEditor> editor (processor.createEditor());
        if (auto* e = dynamic_cast<VocalMatchEditor*> (editor.get()))
        {
            e->getPanel().showPage (processor.editorPage);
            e->resized();
        }
        if (checkLayoutFlag)
        {
            int failures = 0;
            for (int page = 0; page < 2; ++page)
            {
                auto* e = dynamic_cast<VocalMatchEditor*> (editor.get());
                e->getPanel().showPage (page);
                const auto problems = vm::ui::checkLayout (*editor);
                std::cout << "layout " << editor->getWidth() << " x " << editor->getHeight() << (page == 0 ? " analyze" : " chain")
                          << ": " << (problems.isEmpty() ? juce::String ("ok") : juce::String (problems.size()) + " problems") << "\n";
                for (const auto& pr : problems)
                    std::cout << "  " << pr << "\n";
                failures += problems.size();
            }
            if (failures > 0)
                return 1;
            if (snapshotFile == juce::File())
                return 0;
            dynamic_cast<VocalMatchEditor*> (editor.get())->getPanel().showPage (pageText == "chain" ? 1 : 0);
        }
        const auto image = editor->createComponentSnapshot (editor->getLocalBounds(), true, 2.0f);
        snapshotFile.deleteFile();
        juce::FileOutputStream stream (snapshotFile);
        juce::PNGImageFormat png;
        if (! stream.openedOk() || ! png.writeImageToStream (image, stream))
            return fail ("cannot write " + snapshotFile.getFullPathName());
        std::cout << "wrote " << snapshotFile.getFullPathName() << " (" << editor->getWidth() << " x " << editor->getHeight() << ")\n";
        // the editor's size is saved with the song
        juce::MemoryBlock state;
        processor.getStateInformation (state);
        VocalMatchProcessor restored;
        restored.setStateInformation (state.getData(), (int) state.getSize());
        std::cout << "saved editor size " << restored.editorWidth << " x " << restored.editorHeight << ", page " << restored.editorPage << "\n";
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
    if (previewSong != juce::File())
    {
        juce::String error;
        auto song = vm::PreviewPlayer::decode (previewSong, error);
        if (song == nullptr)
            return fail ("preview: " + error);
        processor.preview.hand (std::move (song));
        processor.preview.play (previewStart, previewEnd);
    }
    if (captureFile != juce::File())
        processor.setCaptureArmed (true); // no play head here: everything is captured
    juce::MidiBuffer midi;
    for (int start = 0; start < length; start += block)
    {
        const int n = std::min (block, length - start);
        float* channels[] = { audio.getWritePointer (0, start), audio.getWritePointer (1, start) };
        juce::AudioBuffer<float> view (channels, 2, n);
        processor.processBlock (view, midi);
    }
    processor.releaseResources();
    if (captureFile != juce::File())
    {
        processor.setCaptureArmed (false);
        if (! processor.capture.writeWav (captureFile))
            return fail ("cannot write the captured take to " + captureFile.getFullPathName());
        std::cout << "captured " << processor.capture.seconds() << " s to " << captureFile.getFullPathName() << "\n";
    }

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
