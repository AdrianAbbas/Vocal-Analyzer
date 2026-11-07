#pragma once

#include "Analyzer.h"
#include "Capture.h"
#include "DSP/Chain.h"
#include "Parameters.h"
#include "PresetFile.h"
#include "Preview.h"
#include "SingerScan.h"

#include <juce_audio_processors/juce_audio_processors.h>

class VocalMatchProcessor final : public juce::AudioProcessor
{
public:
    VocalMatchProcessor();
    ~VocalMatchProcessor() override;

    void prepareToPlay (double sampleRate, int samplesPerBlock) override;
    void releaseResources() override {}
    bool isBusesLayoutSupported (const BusesLayout& layouts) const override;
    void processBlock (juce::AudioBuffer<float>&, juce::MidiBuffer&) override;
    void processBlock (juce::AudioBuffer<double>&, juce::MidiBuffer&) override;
    bool supportsDoublePrecisionProcessing() const override { return true; }
    void reset() override { chain.reset(); }

    juce::AudioProcessorEditor* createEditor() override;
    bool hasEditor() const override { return true; }

    const juce::String getName() const override { return JucePlugin_Name; }
    bool acceptsMidi() const override { return false; }
    bool producesMidi() const override { return false; }
    bool isMidiEffect() const override { return false; }
    double getTailLengthSeconds() const override { return vm::Chain::tailSeconds (reader.read()); }

    int getNumPrograms() override { return 1; }
    int getCurrentProgram() override { return 0; }
    void setCurrentProgram (int) override {}
    const juce::String getProgramName (int) override { return "Default"; }
    void changeProgramName (int, const juce::String&) override {}

    void getStateInformation (juce::MemoryBlock& destData) override;
    void setStateInformation (const void* data, int sizeInBytes) override;

    // ---- preset.json (message thread) ------------------------------------------------
    // Sets every parameter the file holds (as automatable gestures, so the host records
    // them) and remembers the estimates for the editor.
    juce::Result loadPresetFile (const juce::File& file);
    juce::Result loadPresetText (const juce::String& json, const juce::String& displayName);
    // Sets every parameter back to the loaded preset's estimates (Match amount and A/B stay).
    bool resetToEstimates();

    juce::String getPresetName() const { return presetName; }
    juce::String getPresetStatus() const { return presetStatus; }
    std::optional<float> getEstimate (const juce::String& paramId) const;
    std::optional<float> getConfidence (const juce::String& effectKey) const;
    int getPresetGeneration() const { return presetGeneration; } // bumps on load / restore

    void setParameterValue (const juce::String& paramId, float realValue);

    // ---- Analyze (milestone 6, message thread) ----------------------------------------
    vm::Capture capture;              // the plug-in's input, recorded while the host plays
    vm::AnalyzerRun analyzer;         // the analyzer as a separate process
    juce::File referenceFile, dryFile; // the user's choices (saved with the song)
    bool useCapturedDry = true;
    // Builds the request (writing the captured take to a WAV when it is used).
    juce::Result makeAnalyzeRequest (vm::AnalyzeRequest& out);
    void setCaptureArmed (bool armed) { capture.setArmed (armed); }

    // ---- the reference song: waveform, preview, singers (message thread) ---------------
    struct Peaks
    {
        std::vector<float> lo, hi; // per bucket, mono mix
        double seconds = 0.0;
    };
    // Decodes referenceFile on a background thread (waveform + preview); pollSong() picks it up.
    void loadReferenceSong();
    void pollSong(); // editor timer: hands a finished decode to the preview, bumps the generation
    const Peaks& getPeaks() const { return peaks; }
    int getSongGeneration() const { return songGeneration; }
    juce::String getSongError() const { return songError; }
    bool isSongLoading() const { return songLoading; }
    vm::PreviewPlayer preview;

    vm::AnalyzerRun scanner; // `vocalmatch singers` for the reference
    const vm::SingerScan& getScan() const { return scan; }
    int getScanGeneration() const { return scanGeneration; }
    bool loadScanFile (const juce::File&); // false if it is not a scan of the current reference
    void clearScan();
    juce::File scanFileFor (const juce::File& song) const; // ~/Music/Vocal Match/Scans/<song>.singers.json

    juce::Array<juce::Range<double>> regions; // marked by hand; empty = the whole song
    juce::StringArray chosenSingers;          // ticked in the Analyze panel
    bool perSingerPresets = false;
    bool referenceIsVocal = false; // the reference is an isolated vocal: no source separation

    // ---- editor state (saved with the song) ----------------------------------------------
    int editorWidth = 1040, editorHeight = 660, editorPage = 0;
    juce::StringArray collapsedCards;

    // ---- meters (written by the audio thread, read by the editor) ----------------------
    std::atomic<float> inputPeak { 0.0f }, outputPeak { 0.0f };
    std::atomic<float> compressorReductionDb { 0.0f }, deesserReductionDb { 0.0f };

    juce::AudioProcessorValueTreeState apvts;

private:
    template <typename Sample>
    void process (juce::AudioBuffer<Sample>& buffer);
    template <typename Sample>
    void processChunk (Sample* left, Sample* right, int n, bool monoOut);
    bool hostIsPlaying();
    void rememberPreset (const vm::PresetValues&, const juce::String& json, const juce::String& name);
    juce::String describe (const vm::PresetValues&) const;

    vm::params::Reader reader;
    vm::Chain chain;
    std::atomic<float>* abParam = nullptr;

    // A/B: the dry input, kept per chunk, crossfaded with the processed signal over 20 ms.
    static constexpr int kChunk = 1024;
    std::array<std::array<double, kChunk>, 2> dryChunk {};
    double abGain = 0.0, abStep = 0.0;

    struct Decoded
    {
        std::unique_ptr<vm::Song> song;
        Peaks peaks;
        juce::String error;
        juce::File file;
    };
    std::mutex decodedLock;
    std::unique_ptr<Decoded> decoded; // a finished decode waiting for pollSong()
    Peaks peaks;
    juce::String songError;
    bool songLoading = false;
    int songGeneration = 0;
    vm::SingerScan scan;
    juce::String scanJson;
    int scanGeneration = 0;
    double currentRate = 44100.0;

    juce::String presetName, presetStatus, presetJson;
    vm::PresetValues estimates;
    int presetGeneration = 0;
    juce::ThreadPool songPool { 1 }; // last: its jobs write the members above

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR (VocalMatchProcessor)
};
