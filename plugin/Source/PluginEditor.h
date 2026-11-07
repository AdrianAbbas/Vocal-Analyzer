#pragma once

#include "PluginProcessor.h"

#include <juce_audio_utils/juce_audio_utils.h>

namespace vm::ui
{
// A rotary knob bound to one parameter, with a caption, a value box and a marker at the
// value the analyzer estimated. Double-click returns it to the estimate (or the default).
class ParamKnob final : public juce::Component
{
public:
    ParamKnob (VocalMatchProcessor&, const juce::String& paramId, const juce::String& caption);
    void resized() override;
    void paintOverChildren (juce::Graphics&) override;
    void refresh();
    const juce::String paramId;

private:
    VocalMatchProcessor& processor;
    juce::Label caption;
    juce::Slider slider { juce::Slider::RotaryHorizontalVerticalDrag, juce::Slider::TextBoxBelow };
    juce::AudioProcessorValueTreeState::SliderAttachment attachment;
};

// A horizontal gain-reduction bar (0 .. 12 dB) with its value.
class ReductionMeter final : public juce::Component, public juce::SettableTooltipClient
{
public:
    explicit ReductionMeter (std::function<float()> source) : read (std::move (source)) {}
    void paint (juce::Graphics&) override;
    void update();

private:
    std::function<float()> read;
    float shown = 0.0f;
};

// One effect: a title, its bypass ("on") switch, the estimator's confidence, and knobs.
class EffectCard : public juce::Component
{
public:
    EffectCard (VocalMatchProcessor&, const juce::String& title, const juce::String& effectKey, const juce::String& onParamId);
    void addKnob (const juce::String& paramId, const juce::String& caption);
    void addMeter (std::function<float()> source);
    void paint (juce::Graphics&) override;
    void resized() override;
    void refresh(); // after a preset load
    void updateMeter();
    int span = 1;   // grid columns this card takes

protected:
    juce::Rectangle<int> contentArea() const;
    VocalMatchProcessor& processor;
    const juce::String effectKey;
    juce::OwnedArray<ParamKnob> knobs;

private:
    juce::String title;
    juce::ToggleButton onButton { "On" };
    juce::AudioProcessorValueTreeState::ButtonAttachment onAttachment;
    juce::Label confidence;
    std::unique_ptr<ReductionMeter> meter;
};

// The EQ: six bands, each with its own on switch and type, frequency, gain and Q.
class EqCard final : public EffectCard
{
public:
    explicit EqCard (VocalMatchProcessor&);
    void resized() override;

private:
    struct Band
    {
        std::unique_ptr<juce::ToggleButton> on;
        std::unique_ptr<juce::ComboBox> type;
        std::unique_ptr<juce::AudioProcessorValueTreeState::ButtonAttachment> onAttachment;
        std::unique_ptr<juce::AudioProcessorValueTreeState::ComboBoxAttachment> typeAttachment;
    };
    std::array<Band, kNumEqBands> bands;
};

// Input / output peak meters (vertical, with peak hold).
class LevelMeters final : public juce::Component
{
public:
    explicit LevelMeters (VocalMatchProcessor& p) : processor (p) {}
    void paint (juce::Graphics&) override;
    void update();

private:
    VocalMatchProcessor& processor;
    float in = -90.0f, out = -90.0f;
};

// Analyze (milestone 6): the reference song, the dry vocal (the captured take or a file),
// Analyze / Cancel with progress, and the Terminal command for hosts that cannot run it.
class AnalyzeStrip final : public juce::Component
{
public:
    explicit AnalyzeStrip (VocalMatchProcessor&);
    void paint (juce::Graphics&) override;
    void resized() override;
    void update(); // ~10 Hz from the panel's timer
    std::function<void (const juce::File&)> onPresetReady;

private:
    void chooseReference();
    void chooseDry();
    void startAnalysis();
    void refreshLabels();

    VocalMatchProcessor& processor;
    juce::TextButton referenceButton { "Reference song..." };
    juce::Label referenceLabel;
    juce::TextButton captureButton { "Capture" };
    juce::ComboBox drySource;
    juce::Label dryLabel;
    juce::TextButton analyzeButton { "Analyze" }, cancelButton { "Cancel" }, copyButton { "Copy command" };
    double progressValue = 0.0;
    juce::ProgressBar progressBar { progressValue };
    juce::Label status;
    std::unique_ptr<juce::FileChooser> chooser;
    juce::String lastCommand;
    vm::AnalyzerRun::State lastState = vm::AnalyzerRun::State::idle;
};

// Everything in the window at its natural size (kWidth wide); the editor scales it.
class MainPanel final : public juce::Component,
                        public juce::FileDragAndDropTarget,
                        private juce::Timer
{
public:
    explicit MainPanel (VocalMatchProcessor&);
    ~MainPanel() override;

    void paint (juce::Graphics&) override;
    void resized() override;
    int naturalHeight() const;

    bool isInterestedInFileDrag (const juce::StringArray& files) override;
    void filesDropped (const juce::StringArray& files, int, int) override;
    void fileDragExit (const juce::StringArray&) override;

private:
    void timerCallback() override;
    void choosePreset();
    void load (const juce::File&);
    void refresh();

    VocalMatchProcessor& processor;
    juce::TextButton loadButton { "Load preset..." }, resetButton { "Reset to estimates" }, abButton { "A/B: dry" };
    juce::AudioProcessorValueTreeState::ButtonAttachment abAttachment;
    juce::Label status;
    ParamKnob matchKnob;
    LevelMeters meters;
    AnalyzeStrip analyze;
    juce::OwnedArray<EffectCard> cards;
    std::unique_ptr<juce::FileChooser> chooser;
    juce::TooltipWindow tooltips { this, 600 };
    int seenGeneration = -1;
    int ticks = 0;
    bool dragOver = false;
};
} // namespace vm::ui

class VocalMatchEditor final : public juce::AudioProcessorEditor
{
public:
    explicit VocalMatchEditor (VocalMatchProcessor&);
    ~VocalMatchEditor() override;
    void resized() override;

private:
    juce::LookAndFeel_V4 lookAndFeel;
    vm::ui::MainPanel panel;
    int naturalWidth = 0, naturalHeight = 0;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR (VocalMatchEditor)
};
