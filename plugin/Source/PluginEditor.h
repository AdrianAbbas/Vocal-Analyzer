#pragma once

#include "PluginProcessor.h"

#include <juce_audio_utils/juce_audio_utils.h>

namespace vm::ui
{
// A rotary knob bound to one parameter, with a caption, a value box and a marker at the
// value the analyzer estimated (when a preset is loaded).
class ParamKnob final : public juce::Component
{
public:
    ParamKnob (VocalMatchProcessor&, const juce::String& paramId, const juce::String& caption);
    void resized() override;
    void paintOverChildren (juce::Graphics&) override;
    const juce::String paramId;

private:
    VocalMatchProcessor& processor;
    juce::Label caption;
    juce::Slider slider { juce::Slider::RotaryHorizontalVerticalDrag, juce::Slider::TextBoxBelow };
    juce::AudioProcessorValueTreeState::SliderAttachment attachment;
};

// One effect: a title, its bypass ("on") switch, the estimator's confidence, and knobs.
class EffectCard : public juce::Component
{
public:
    EffectCard (VocalMatchProcessor&, const juce::String& title, const juce::String& effectKey, const juce::String& onParamId);
    void addKnob (const juce::String& paramId, const juce::String& caption);
    void paint (juce::Graphics&) override;
    void resized() override;
    void refresh(); // after a preset load
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
} // namespace vm::ui

class VocalMatchEditor final : public juce::AudioProcessorEditor,
                               public juce::FileDragAndDropTarget,
                               private juce::Timer
{
public:
    explicit VocalMatchEditor (VocalMatchProcessor&);
    ~VocalMatchEditor() override;

    void paint (juce::Graphics&) override;
    void resized() override;

    bool isInterestedInFileDrag (const juce::StringArray& files) override;
    void filesDropped (const juce::StringArray& files, int, int) override;
    void fileDragExit (const juce::StringArray&) override;

private:
    void timerCallback() override;
    void choosePreset();
    void load (const juce::File&);
    void refresh();

    VocalMatchProcessor& processor;
    juce::LookAndFeel_V4 lookAndFeel;
    juce::TextButton loadButton { "Load preset..." };
    juce::Label status;
    vm::ui::ParamKnob matchKnob;
    juce::OwnedArray<vm::ui::EffectCard> cards;
    std::unique_ptr<juce::FileChooser> chooser;
    int seenGeneration = -1;
    bool dragOver = false;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR (VocalMatchEditor)
};
