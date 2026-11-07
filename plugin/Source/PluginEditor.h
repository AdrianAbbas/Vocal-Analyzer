#pragma once

#include "PluginProcessor.h"

#include <juce_audio_utils/juce_audio_utils.h>

// The editor (milestone 6, compact): a header that is always visible (tabs, Load preset,
// Reset, A/B, the status line, IN/OUT meters and Match amount) over one of two pages:
//   Analyze: reference and dry vocal, the song's waveform with its vocal activity and singers,
//            regions to analyse, singer choice and preview, Analyze;
//   Chain:   the effects as a grid of collapsible cards with small knobs and value readouts.
// Layout is in logical points at a design size of kDesignWidth x kDesignHeight; the editor
// scales it uniformly (fonts and knobs stay proportional) and gives the page the extra room
// when the window's shape differs. Default 1040 x 660, resizable 900 x 580 .. 2080 x 1320.

namespace vm::ui
{
constexpr int kDesignWidth = 1040, kDesignHeight = 660;
constexpr int kMinWidth = 900, kMinHeight = 580, kMaxWidth = 2080, kMaxHeight = 1320;

// A small rotary knob bound to one parameter: caption above, value readout below (click it to
// type a value), a dot at the analyzer's estimate. Double-click returns to the estimate.
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

// A slim gain-reduction bar (0 .. 12 dB).
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

// One effect: title bar (collapse arrow, title, confidence, On) over its knobs.
class EffectCard : public juce::Component
{
public:
    EffectCard (VocalMatchProcessor&, const juce::String& title, const juce::String& effectKey, const juce::String& onParamId);
    void addKnob (const juce::String& paramId, const juce::String& caption);
    void addMeter (std::function<float()> source);
    void paint (juce::Graphics&) override;
    void resized() override;
    void mouseUp (const juce::MouseEvent&) override;
    void refresh(); // after a preset load
    void updateMeter();

    bool isCollapsed() const { return collapsed; }
    void setCollapsed (bool);
    std::function<void()> onCollapseChange;
    // Natural size for a knob width k (logical points); collapsed cards are a title bar only.
    virtual int naturalWidth (int k) const;
    virtual int naturalHeight (int k) const;
    const juce::String effectKey;
    static constexpr int kTitleHeight = 22;

protected:
    juce::Rectangle<int> contentArea() const;
    VocalMatchProcessor& processor;
    juce::OwnedArray<ParamKnob> knobs;
    bool collapsed = false;

private:
    juce::String title;
    juce::ToggleButton onButton { "On" };
    juce::AudioProcessorValueTreeState::ButtonAttachment onAttachment;
    juce::String confidenceText;
    std::unique_ptr<ReductionMeter> meter;
};

// The EQ: six bands, each with its own on switch and type over freq, gain and Q.
class EqCard final : public EffectCard
{
public:
    explicit EqCard (VocalMatchProcessor&);
    void resized() override;
    int naturalWidth (int k) const override;
    int naturalHeight (int k) const override;

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

// Input / output peak meters (vertical, with a slow fall).
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

// The Chain page: the cards in reading order, flowed into rows that fill the width. The knob
// size is the largest that lets every card fit the page.
class ChainPage final : public juce::Component
{
public:
    explicit ChainPage (VocalMatchProcessor&);
    void resized() override;
    void refresh();
    void updateMeters();

private:
    VocalMatchProcessor& processor;
    juce::OwnedArray<EffectCard> cards;
};

// The song's waveform with the vocal-activity lane, the singer lane (a colour per singer,
// hatched where they overlap), marked regions and the preview position.
// Drag to mark a region; drag a region's edge to move it; right-click (or Alt-click) a region
// to remove it; click a region to select it.
class WaveformView final : public juce::Component, public juce::SettableTooltipClient
{
public:
    explicit WaveformView (VocalMatchProcessor&);
    void paint (juce::Graphics&) override;
    void resized() override;
    void mouseDown (const juce::MouseEvent&) override;
    void mouseDrag (const juce::MouseEvent&) override;
    void mouseUp (const juce::MouseEvent&) override;
    void mouseMove (const juce::MouseEvent&) override;
    void update(); // the preview position (timer)
    void songChanged(); // new peaks or scan: redraw the cached background
    std::function<void()> onRegionsChanged;
    int selected = -1;

private:
    double timeAt (float x) const;
    float xAt (double t) const;
    juce::Rectangle<float> waveArea() const;
    int regionAt (float x, int* edge) const;
    void renderBackground();

    VocalMatchProcessor& processor;
    juce::Image background; // waveform + lanes, re-rendered only when the song, scan or size changes
    float backgroundScale = 0.0f;
    int dragRegion = -1, dragEdge = 0; // edge: -1 start, 1 end, 0 new
    double dragAnchor = 0.0;
    float playX = -1.0f;
};

// The Analyze page.
class AnalyzePage final : public juce::Component, private juce::Timer
{
public:
    explicit AnalyzePage (VocalMatchProcessor&);
    void paint (juce::Graphics&) override;
    void resized() override;
    void update(); // ~10 Hz from the editor's timer
    std::function<void (const juce::File&)> onPresetReady;

private:
    void timerCallback() override; // waiting for a scan run in Terminal
    void chooseReference();
    void chooseDry();
    void referenceChanged();
    void startScan();
    void startAnalysis();
    void refreshLabels();
    void rebuildSingerRows();
    void refreshPlan();
    juce::String planText() const;

    VocalMatchProcessor& processor;
    juce::TextButton referenceButton { "Reference song..." };
    juce::Label referenceLabel;
    juce::TextButton captureButton { "Capture" };
    juce::ComboBox drySource;
    juce::Label dryLabel;
    WaveformView waveform;
    juce::TextButton playButton { "Play" }, stopButton { "Stop" }, clearButton { "Clear regions" }, scanButton { "Find singers" };
    juce::Label hint, scanStatus;
    struct SingerRow
    {
        std::unique_ptr<juce::ToggleButton> match;
        std::unique_ptr<juce::TextButton> preview;
        std::unique_ptr<juce::Label> info;
        juce::Colour colour;
    };
    std::vector<SingerRow> singerRows;
    juce::Label singerMessage, plan;
    juce::ToggleButton perSinger { "One preset per singer" };
    juce::TextButton analyzeButton { "Analyze" }, cancelButton { "Cancel" }, copyButton { "Copy command" }, showLogButton { "Show log" };
    juce::ToggleButton isolatedVocal { "Isolated vocal" };
    juce::File failedLog; // the log of the last run that failed (Show log)
    double progressValue = 0.0;
    juce::ProgressBar progressBar { progressValue };
    juce::Label status;
    std::unique_ptr<juce::FileChooser> chooser;
    juce::String lastCommand;
    vm::AnalyzerRun::State lastState = vm::AnalyzerRun::State::idle, lastScanState = vm::AnalyzerRun::State::idle;
    int seenScanGeneration = -1, seenSongGeneration = -1;
    juce::File waitingForScan;
};

// Everything in the window, laid out at logical size; the editor scales it.
class MainPanel final : public juce::Component,
                        public juce::FileDragAndDropTarget,
                        private juce::Timer
{
public:
    explicit MainPanel (VocalMatchProcessor&);
    ~MainPanel() override;

    void paint (juce::Graphics&) override;
    void resized() override;

    bool isInterestedInFileDrag (const juce::StringArray& files) override;
    void filesDropped (const juce::StringArray& files, int, int) override;
    void fileDragExit (const juce::StringArray&) override;
    void showPage (int page); // 0 Analyze, 1 Chain

private:
    void timerCallback() override;
    void choosePreset();
    void load (const juce::File&);
    void refresh();

    VocalMatchProcessor& processor;
    juce::TextButton analyzeTab { "Analyze" }, chainTab { "Chain" };
    juce::TextButton loadButton { "Load preset..." }, resetButton { "Reset to estimates" }, abButton { "A/B: dry" };
    juce::AudioProcessorValueTreeState::ButtonAttachment abAttachment;
    juce::Label status;
    ParamKnob matchKnob;
    LevelMeters meters;
    AnalyzePage analyzePage;
    ChainPage chainPage;
    std::unique_ptr<juce::FileChooser> chooser;
    juce::TooltipWindow tooltips { this, 600 };
    int seenGeneration = -1;
    int ticks = 0;
    bool dragOver = false;
};

// The look: the dark scheme, with rotary knobs that use their whole area (the stock drawing
// leaves a 10 pt margin, too much for small knobs).
class CompactLook final : public juce::LookAndFeel_V4
{
public:
    CompactLook();
    void drawRotarySlider (juce::Graphics&, int x, int y, int width, int height, float sliderPos, float startAngle,
                           float endAngle, juce::Slider&) override;
};

// Layout check (tests, VocalMatchRender --check-layout): every visible component inside its
// parent and not overlapping a visible sibling. Returns the problems found.
juce::StringArray checkLayout (juce::Component& root);
} // namespace vm::ui

class VocalMatchEditor final : public juce::AudioProcessorEditor
{
public:
    explicit VocalMatchEditor (VocalMatchProcessor&);
    ~VocalMatchEditor() override;
    void resized() override;
    vm::ui::MainPanel& getPanel() { return panel; }

private:
    vm::ui::CompactLook lookAndFeel;
    vm::ui::MainPanel panel;
    VocalMatchProcessor& processor;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR (VocalMatchEditor)
};
