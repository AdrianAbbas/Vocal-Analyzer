#pragma once

#include "Analyzer.h"
#include "Capture.h"
#include "DSP/Chain.h"
#include "Parameters.h"
#include "PresetFile.h"

#include <juce_audio_processors/juce_audio_processors.h>

class VocalMatchProcessor final : public juce::AudioProcessor
{
public:
    VocalMatchProcessor();
    ~VocalMatchProcessor() override = default;

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

    juce::String presetName, presetStatus, presetJson;
    vm::PresetValues estimates;
    int presetGeneration = 0;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR (VocalMatchProcessor)
};
