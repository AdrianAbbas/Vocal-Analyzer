#pragma once

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
    double getTailLengthSeconds() const override { return chain.tailSeconds(); }

    int getNumPrograms() override { return 1; }
    int getCurrentProgram() override { return 0; }
    void setCurrentProgram (int) override {}
    const juce::String getProgramName (int) override { return "Default"; }
    void changeProgramName (int, const juce::String&) override {}

    void getStateInformation (juce::MemoryBlock& destData) override;
    void setStateInformation (const void* data, int sizeInBytes) override;

    // ---- preset.json (message thread) ------------------------------------------------
    // Sets every parameter the file holds (as automatable gestures, so the host records
    // them) and remembers the estimates for the editor. Returns a one-line status.
    juce::Result loadPresetFile (const juce::File& file);
    juce::Result loadPresetText (const juce::String& json, const juce::String& displayName);

    // The last loaded preset: its name, a status line, and estimated value / confidence.
    juce::String getPresetName() const { return presetName; }
    juce::String getPresetStatus() const { return presetStatus; }
    std::optional<float> getEstimate (const juce::String& paramId) const;
    std::optional<float> getConfidence (const juce::String& effectKey) const;
    // Increments whenever a preset is loaded or restored (the editor polls it).
    int getPresetGeneration() const { return presetGeneration; }

    void setParameterValue (const juce::String& paramId, float realValue);

    juce::AudioProcessorValueTreeState apvts;

private:
    template <typename Sample>
    void process (juce::AudioBuffer<Sample>& buffer);
    void rememberPreset (const vm::PresetValues&, const juce::String& json, const juce::String& name);
    juce::String describe (const vm::PresetValues&) const;

    vm::params::Reader reader;
    vm::Chain chain;

    juce::String presetName, presetStatus, presetJson;
    vm::PresetValues estimates;
    int presetGeneration = 0;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR (VocalMatchProcessor)
};
