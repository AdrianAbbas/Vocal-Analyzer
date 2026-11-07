#include "PluginEditor.h"

namespace
{
namespace colours
{
const juce::Colour background { 0xff15181d };
const juce::Colour card { 0xff1f242b };
const juce::Colour cardOff { 0xff1a1d22 };
const juce::Colour outline { 0xff2e353f };
const juce::Colour text { 0xffe4e8ee };
const juce::Colour dim { 0xff8a94a3 };
const juce::Colour accent { 0xff4fb3a9 };
const juce::Colour estimate { 0xfff2b84b };
} // namespace colours

constexpr int kHeaderHeight = 96;
constexpr int kCardHeight = 128;
constexpr int kEqHeight = 160;
constexpr int kWidth = 940;
constexpr int kGap = 8;
constexpr int kKnobWidth = 64;
constexpr int kAnalyzeHeight = 74;
} // namespace

namespace vm::ui
{
// -------------------------------------------------------------------------- ParamKnob
ParamKnob::ParamKnob (VocalMatchProcessor& p, const juce::String& id, const juce::String& text)
    : paramId (id), processor (p), attachment (p.apvts, id, slider)
{
    caption.setText (text, juce::dontSendNotification);
    caption.setJustificationType (juce::Justification::centred);
    caption.setFont (juce::FontOptions (12.0f));
    caption.setColour (juce::Label::textColourId, colours::dim);
    caption.setInterceptsMouseClicks (false, false);
    addAndMakeVisible (caption);
    slider.setTextBoxStyle (juce::Slider::TextBoxBelow, false, kKnobWidth, 16);
    slider.setColour (juce::Slider::rotarySliderFillColourId, colours::accent);
    slider.setColour (juce::Slider::textBoxOutlineColourId, juce::Colours::transparentBlack);
    slider.setColour (juce::Slider::textBoxTextColourId, colours::text);
    if (auto* param = p.apvts.getParameter (id))
        slider.setTooltip (param->getName (64));
    addAndMakeVisible (slider);
}

void ParamKnob::resized()
{
    auto r = getLocalBounds();
    caption.setBounds (r.removeFromTop (16));
    slider.setBounds (r);
}

void ParamKnob::refresh()
{
    if (auto* param = processor.apvts.getParameter (paramId))
    {
        const auto est = processor.getEstimate (paramId);
        const float value = est ? *est : param->convertFrom0to1 (param->getDefaultValue());
        slider.setDoubleClickReturnValue (true, (double) value);
        slider.setTooltip (param->getName (64) + (est ? " - estimate " + param->getText (param->convertTo0to1 (*est), 16)
                                                              + " (double-click to return to it)"
                                                        : juce::String()));
    }
    repaint();
}

void ParamKnob::paintOverChildren (juce::Graphics& g)
{
    const auto est = processor.getEstimate (paramId);
    auto* param = processor.apvts.getParameter (paramId);
    if (! est || param == nullptr)
        return;
    // A dot on the knob's arc at the estimated value.
    const auto rp = slider.getRotaryParameters();
    const float proportion = juce::jlimit (0.0f, 1.0f, param->convertTo0to1 (*est));
    const float angle = rp.startAngleRadians + proportion * (rp.endAngleRadians - rp.startAngleRadians);
    auto knob = slider.getBounds().toFloat().withTrimmedBottom (18.0f).reduced (2.0f);
    const float radius = juce::jmin (knob.getWidth(), knob.getHeight()) / 2.0f;
    const auto centre = knob.getCentre();
    const juce::Point<float> dot (centre.x + radius * std::sin (angle), centre.y - radius * std::cos (angle));
    g.setColour (colours::estimate);
    g.fillEllipse (juce::Rectangle<float> (5.0f, 5.0f).withCentre (dot));
}

// ------------------------------------------------------------------------- EffectCard
EffectCard::EffectCard (VocalMatchProcessor& p, const juce::String& t, const juce::String& key, const juce::String& onId)
    : processor (p), effectKey (key), title (t), onAttachment (p.apvts, onId, onButton)
{
    onButton.setTooltip ("Off = this effect is bypassed");
    onButton.onStateChange = [this] { repaint(); };
    addAndMakeVisible (onButton);
    confidence.setJustificationType (juce::Justification::centredRight);
    confidence.setFont (juce::FontOptions (12.0f));
    confidence.setColour (juce::Label::textColourId, colours::estimate);
    addAndMakeVisible (confidence);
}

void EffectCard::addKnob (const juce::String& paramId, const juce::String& caption)
{
    addAndMakeVisible (knobs.add (new ParamKnob (processor, paramId, caption)));
}

void EffectCard::paint (juce::Graphics& g)
{
    const auto r = getLocalBounds().toFloat().reduced (0.5f);
    g.setColour (onButton.getToggleState() ? colours::card : colours::cardOff);
    g.fillRoundedRectangle (r, 6.0f);
    g.setColour (colours::outline);
    g.drawRoundedRectangle (r, 6.0f, 1.0f);
    g.setColour (onButton.getToggleState() ? colours::text : colours::dim);
    g.setFont (juce::FontOptions (15.0f, juce::Font::bold));
    g.drawText (title, 12, 6, 200, 20, juce::Justification::centredLeft);
}

juce::Rectangle<int> EffectCard::contentArea() const { return getLocalBounds().reduced (10, 0).withTrimmedTop (30).withTrimmedBottom (6); }

void EffectCard::resized()
{
    onButton.setBounds (getWidth() - 64, 6, 56, 20);
    confidence.setBounds (getWidth() - 250, 6, 180, 20);
    auto area = contentArea();
    if (meter != nullptr)
        meter->setBounds (area.removeFromRight (24));
    for (auto* k : knobs)
        k->setBounds (area.removeFromLeft (kKnobWidth + 6).withTrimmedRight (6));
}

void EffectCard::refresh()
{
    const auto c = processor.getConfidence (effectKey);
    confidence.setText (c && *c > 0.0f ? "estimated, confidence " + juce::String (juce::roundToInt (*c * 100.0f)) + " %" : juce::String(),
                        juce::dontSendNotification);
    repaint();
    for (auto* k : knobs)
        k->refresh();
}

void EffectCard::addMeter (std::function<float()> source)
{
    meter = std::make_unique<ReductionMeter> (std::move (source));
    meter->setTooltip ("Gain reduction (0 - 12 dB)");
    addAndMakeVisible (*meter);
}

void EffectCard::updateMeter()
{
    if (meter != nullptr)
        meter->update();
}

// --------------------------------------------------------------------- ReductionMeter
void ReductionMeter::update()
{
    const float v = juce::jlimit (0.0f, 24.0f, read());
    // fast attack, slower fall, so short reductions stay visible
    shown = v > shown ? v : shown + 0.35f * (v - shown);
    repaint();
}

void ReductionMeter::paint (juce::Graphics& g)
{
    // A slim vertical bar (0 .. 12 dB, filling downwards like a gain-reduction meter), "GR" below.
    auto r = getLocalBounds().toFloat();
    auto label = r.removeFromBottom (14.0f);
    auto bar = r.reduced (7.0f, 4.0f);
    g.setColour (colours::outline);
    g.fillRoundedRectangle (bar, 2.0f);
    g.setColour (colours::estimate);
    g.fillRoundedRectangle (bar.withHeight (bar.getHeight() * juce::jmin (shown, 12.0f) / 12.0f), 2.0f);
    g.setColour (colours::dim);
    g.setFont (juce::FontOptions (10.0f));
    g.drawText ("GR", label, juce::Justification::centred);
}

// ----------------------------------------------------------------------------- EqCard
EqCard::EqCard (VocalMatchProcessor& p) : EffectCard (p, "EQ", "eq", params::eqOn)
{
    span = 2;
    for (int b = 0; b < kNumEqBands; ++b)
    {
        auto& band = bands[(size_t) b];
        band.on = std::make_unique<juce::ToggleButton> (juce::String (b + 1));
        band.type = std::make_unique<juce::ComboBox>();
        band.type->addItemList (params::eqTypeNames, 1);
        band.onAttachment = std::make_unique<juce::AudioProcessorValueTreeState::ButtonAttachment> (p.apvts, params::eqBandOn (b), *band.on);
        band.typeAttachment = std::make_unique<juce::AudioProcessorValueTreeState::ComboBoxAttachment> (p.apvts, params::eqBandType (b), *band.type);
        addAndMakeVisible (*band.on);
        addAndMakeVisible (*band.type);
        addKnob (params::eqBandFreq (b), "Freq");
        addKnob (params::eqBandGain (b), "Gain");
        addKnob (params::eqBandQ (b), "Q");
    }
}

void EqCard::resized()
{
    EffectCard::resized();
    auto area = contentArea();
    const int bandWidth = area.getWidth() / kNumEqBands;
    for (int b = 0; b < kNumEqBands; ++b)
    {
        auto col = area.removeFromLeft (bandWidth).reduced (3, 0);
        auto top = col.removeFromTop (24);
        bands[(size_t) b].on->setBounds (top.removeFromLeft (40));
        bands[(size_t) b].type->setBounds (top.reduced (0, 1));
        col.removeFromTop (4);
        const int w = col.getWidth() / 3; // freq, gain, Q side by side
        knobs[b * 3 + 0]->setBounds (col.removeFromLeft (w));
        knobs[b * 3 + 1]->setBounds (col.removeFromLeft (w));
        knobs[b * 3 + 2]->setBounds (col);
    }
}
} // namespace vm::ui


namespace vm::ui
{
// ------------------------------------------------------------------------ LevelMeters
void LevelMeters::update()
{
    auto db = [] (float lin) { return juce::Decibels::gainToDecibels (lin, -90.0f); };
    const float i = db (processor.inputPeak.exchange (0.0f)), o = db (processor.outputPeak.exchange (0.0f));
    in = i > in ? i : juce::jmax (i, in - 1.5f);
    out = o > out ? o : juce::jmax (o, out - 1.5f);
    repaint();
}

void LevelMeters::paint (juce::Graphics& g)
{
    auto r = getLocalBounds().toFloat();
    auto labels = r.removeFromBottom (14.0f);
    const float w = r.getWidth() / 2.0f;
    auto draw = [&] (juce::Rectangle<float> bar, float db, const juce::String& name, juce::Rectangle<float> label)
    {
        bar = bar.reduced (4.0f, 0.0f);
        g.setColour (colours::outline);
        g.fillRoundedRectangle (bar, 2.0f);
        const float frac = juce::jlimit (0.0f, 1.0f, (db + 60.0f) / 60.0f);
        g.setColour (db > -0.5f ? juce::Colours::orangered : colours::accent);
        g.fillRoundedRectangle (bar.withTrimmedTop (bar.getHeight() * (1.0f - frac)), 2.0f);
        g.setColour (colours::dim);
        g.setFont (juce::FontOptions (10.0f));
        g.drawText (name, label, juce::Justification::centred);
    };
    draw (r.withWidth (w), in, "IN", labels.withWidth (w));
    draw (r.withTrimmedLeft (w), out, "OUT", labels.withTrimmedLeft (w));
}

// ------------------------------------------------------------------------ AnalyzeStrip
AnalyzeStrip::AnalyzeStrip (VocalMatchProcessor& p) : processor (p)
{
    auto style = [] (juce::Label& l)
    {
        l.setFont (juce::FontOptions (12.0f));
        l.setColour (juce::Label::textColourId, colours::dim);
        l.setMinimumHorizontalScale (0.7f);
    };
    referenceButton.setTooltip ("The song (or vocal stem) whose vocal sound you want. With Demucs installed a full song works.");
    referenceButton.onClick = [this] { chooseReference(); };
    captureButton.setClickingTogglesState (true);
    captureButton.setColour (juce::TextButton::buttonOnColourId, juce::Colours::darkred);
    captureButton.setTooltip ("Record this track's input while the song plays: your dry vocal for Analyze (up to 5 minutes). "
                              "Click again to stop; Shift-click to start over.");
    captureButton.onClick = [this]
    {
        if (juce::ModifierKeys::currentModifiers.isShiftDown())
            processor.capture.clear();
        processor.setCaptureArmed (captureButton.getToggleState());
        if (captureButton.getToggleState())
        {
            processor.useCapturedDry = true;
            drySource.setSelectedId (1, juce::dontSendNotification);
        }
        refreshLabels();
    };
    drySource.addItem ("Dry vocal: captured take", 1);
    drySource.addItem ("Dry vocal: a file...", 2);
    drySource.setSelectedId (processor.useCapturedDry ? 1 : 2, juce::dontSendNotification);
    drySource.onChange = [this]
    {
        processor.useCapturedDry = drySource.getSelectedId() == 1;
        if (! processor.useCapturedDry)
            chooseDry();
        refreshLabels();
    };
    analyzeButton.setTooltip ("Run the Vocal Match analyzer on the reference and your dry vocal, then load its preset");
    analyzeButton.onClick = [this] { startAnalysis(); };
    cancelButton.onClick = [this] { processor.analyzer.cancel(); };
    copyButton.setTooltip ("Copy the analyzer command, to run it in Terminal (then Load preset)");
    copyButton.onClick = [this]
    {
        vm::AnalyzeRequest r;
        const auto res = processor.makeAnalyzeRequest (r);
        if (res.failed())
        {
            status.setText (res.getErrorMessage(), juce::dontSendNotification);
            return;
        }
        lastCommand = vm::AnalyzerRun::shellCommandFor (r);
        juce::SystemClipboard::copyTextToClipboard (lastCommand);
        status.setText ("Copied: " + lastCommand, juce::dontSendNotification);
    };
    style (referenceLabel);
    style (dryLabel);
    style (status);
    progressBar.setColour (juce::ProgressBar::foregroundColourId, colours::accent);
    for (juce::Component* c : std::initializer_list<juce::Component*> { &referenceButton, &referenceLabel, &captureButton, &drySource,
                                                                       &dryLabel, &analyzeButton, &cancelButton, &copyButton,
                                                                       &progressBar, &status })
        addAndMakeVisible (c);
    captureButton.setToggleState (processor.capture.isArmed(), juce::dontSendNotification);
    refreshLabels();
    update();
}

void AnalyzeStrip::paint (juce::Graphics& g)
{
    const auto r = getLocalBounds().toFloat().reduced (0.5f);
    g.setColour (colours::card);
    g.fillRoundedRectangle (r, 6.0f);
    g.setColour (colours::outline);
    g.drawRoundedRectangle (r, 6.0f, 1.0f);
    g.setColour (colours::text);
    g.setFont (juce::FontOptions (15.0f, juce::Font::bold));
    g.drawText ("Analyze", 12, 6, 80, 20, juce::Justification::centredLeft);
}

void AnalyzeStrip::resized()
{
    auto r = getLocalBounds().reduced (10, 6);
    auto row1 = r.removeFromTop (28);
    row1.removeFromLeft (80);
    referenceButton.setBounds (row1.removeFromLeft (130).reduced (0, 2));
    referenceLabel.setBounds (row1.removeFromLeft (170));
    captureButton.setBounds (row1.removeFromLeft (80).reduced (0, 2));
    row1.removeFromLeft (6);
    drySource.setBounds (row1.removeFromLeft (190).reduced (0, 2));
    dryLabel.setBounds (row1);
    r.removeFromTop (6);
    auto row2 = r.removeFromTop (26);
    row2.removeFromLeft (80);
    analyzeButton.setBounds (row2.removeFromLeft (90).reduced (0, 1));
    row2.removeFromLeft (6);
    cancelButton.setBounds (row2.removeFromLeft (70).reduced (0, 1));
    row2.removeFromLeft (8);
    copyButton.setBounds (row2.removeFromRight (120).reduced (0, 1));
    row2.removeFromRight (8);
    progressBar.setBounds (row2.removeFromLeft (220).reduced (0, 3));
    row2.removeFromLeft (8);
    status.setBounds (row2);
}

void AnalyzeStrip::refreshLabels()
{
    referenceLabel.setText (processor.referenceFile.existsAsFile() ? processor.referenceFile.getFileName() : "none chosen",
                            juce::dontSendNotification);
    referenceLabel.setTooltip (processor.referenceFile.getFullPathName());
    if (processor.useCapturedDry)
    {
        const double s = processor.capture.seconds();
        juce::String t = s > 0.0 ? juce::String ((int) s / 60) + ":" + juce::String ((int) s % 60).paddedLeft ('0', 2) + " captured"
                                 : "nothing captured yet";
        if (processor.capture.isArmed())
            t << " (recording while the song plays)";
        if (processor.capture.isFull())
            t << " (full)";
        dryLabel.setText (t, juce::dontSendNotification);
    }
    else
        dryLabel.setText (processor.dryFile.existsAsFile() ? processor.dryFile.getFileName() : "no file chosen", juce::dontSendNotification);
}

void AnalyzeStrip::chooseReference()
{
    chooser = std::make_unique<juce::FileChooser> ("Choose the reference song or vocal stem", processor.referenceFile,
                                                   "*.wav;*.flac;*.mp3;*.aif;*.aiff");
    chooser->launchAsync (juce::FileBrowserComponent::openMode | juce::FileBrowserComponent::canSelectFiles,
                          [safe = juce::Component::SafePointer<AnalyzeStrip> (this)] (const juce::FileChooser& fc)
                          {
                              if (safe != nullptr && fc.getResult() != juce::File())
                              {
                                  safe->processor.referenceFile = fc.getResult();
                                  safe->refreshLabels();
                              }
                          });
}

void AnalyzeStrip::chooseDry()
{
    chooser = std::make_unique<juce::FileChooser> ("Choose your dry vocal", processor.dryFile, "*.wav;*.flac;*.mp3;*.aif;*.aiff");
    chooser->launchAsync (juce::FileBrowserComponent::openMode | juce::FileBrowserComponent::canSelectFiles,
                          [safe = juce::Component::SafePointer<AnalyzeStrip> (this)] (const juce::FileChooser& fc)
                          {
                              if (safe != nullptr && fc.getResult() != juce::File())
                              {
                                  safe->processor.dryFile = fc.getResult();
                                  safe->refreshLabels();
                              }
                          });
}

void AnalyzeStrip::startAnalysis()
{
    vm::AnalyzeRequest r;
    const auto res = processor.makeAnalyzeRequest (r);
    if (res.failed())
    {
        status.setText (res.getErrorMessage(), juce::dontSendNotification);
        return;
    }
    if (processor.capture.isArmed()) // the take is written; stop recording
    {
        processor.setCaptureArmed (false);
        captureButton.setToggleState (false, juce::dontSendNotification);
    }
    lastCommand = vm::AnalyzerRun::shellCommandFor (r);
    if (! processor.analyzer.start (r))
        status.setText (processor.analyzer.getError(), juce::dontSendNotification);
    update();
}

void AnalyzeStrip::update()
{
    using State = vm::AnalyzerRun::State;
    const auto state = processor.analyzer.getState();
    progressValue = state == State::running ? (double) processor.analyzer.getProgress() : (state == State::finished ? 1.0 : 0.0);
    analyzeButton.setEnabled (state != State::running);
    cancelButton.setEnabled (state == State::running);
    if (state == State::running)
        status.setText (processor.analyzer.getStage(), juce::dontSendNotification);
    if (state != lastState)
    {
        if (state == State::finished)
        {
            status.setText ("Done: " + processor.analyzer.getPresetFile().getFileName(), juce::dontSendNotification);
            if (onPresetReady)
                onPresetReady (processor.analyzer.getPresetFile());
        }
        else if (state == State::failed)
        {
            auto msg = processor.analyzer.getError();
            if (lastCommand.isNotEmpty())
            {
                juce::SystemClipboard::copyTextToClipboard (lastCommand);
                msg << " (Command copied: paste it in Terminal.)";
            }
            status.setText (msg, juce::dontSendNotification);
        }
        else if (state == State::cancelled)
            status.setText ("Cancelled.", juce::dontSendNotification);
        lastState = state;
    }
    status.setTooltip (status.getText());
    if (processor.useCapturedDry)
        refreshLabels();
}

// --------------------------------------------------------------------------- MainPanel
MainPanel::MainPanel (VocalMatchProcessor& p)
    : processor (p), abAttachment (p.apvts, vm::params::abDry, abButton),
      matchKnob (p, vm::params::matchAmount, "Match amount"), meters (p), analyze (p)
{
    loadButton.setTooltip ("Open a preset.json written by the Vocal Match analyzer (or drop one on this window)");
    loadButton.onClick = [this] { choosePreset(); };
    resetButton.setTooltip ("Set every setting back to the loaded preset's estimates");
    resetButton.onClick = [this] { processor.resetToEstimates(); };
    abButton.setClickingTogglesState (true);
    abButton.setColour (juce::TextButton::buttonOnColourId, colours::estimate.darker (0.3f));
    abButton.setTooltip ("Hear your dry input instead of the processed signal (a 20 ms crossfade; the effects keep running)");
    status.setColour (juce::Label::textColourId, colours::dim);
    status.setFont (juce::FontOptions (13.0f));
    for (juce::Component* c : std::initializer_list<juce::Component*> { &loadButton, &resetButton, &abButton, &status, &matchKnob, &meters, &analyze })
        addAndMakeVisible (c);
    analyze.onPresetReady = [this] (const juce::File& f) { load (f); };

    namespace P = vm::params;
    // The two gain stages share the first row; then the EQ; then the chain in order.
    auto* input = cards.add (new EffectCard (p, "Input gain", "input_gain", P::inputOn));
    input->addKnob (P::inputGain, "Gain");
    auto* output = cards.add (new EffectCard (p, "Output gain", "output_gain", P::outputOn));
    output->addKnob (P::outputGain, "Gain");
    cards.add (new EqCard (p));
    auto* comp = cards.add (new EffectCard (p, "Compressor", "compressor", P::compOn));
    comp->addKnob (P::compThreshold, "Threshold");
    comp->addKnob (P::compRatio, "Ratio");
    comp->addKnob (P::compAttack, "Attack");
    comp->addKnob (P::compRelease, "Release");
    comp->addKnob (P::compKnee, "Knee");
    comp->addKnob (P::compMakeup, "Make-up");
    comp->addMeter ([&p] { return p.compressorReductionDb.load(); });
    auto* sat = cards.add (new EffectCard (p, "Saturation", "saturation", P::satOn));
    sat->addKnob (P::satDrive, "Drive");
    sat->addKnob (P::satMix, "Mix");
    auto* deess = cards.add (new EffectCard (p, "De-esser", "deesser", P::deessOn));
    deess->addKnob (P::deessFreq, "Freq");
    deess->addKnob (P::deessThreshold, "Threshold");
    deess->addKnob (P::deessReduction, "Reduction");
    deess->addMeter ([&p] { return p.deesserReductionDb.load(); });
    auto* delay = cards.add (new EffectCard (p, "Delay", "delay", P::delayOn));
    delay->addKnob (P::delayTime, "Time");
    delay->addKnob (P::delayFeedback, "Feedback");
    delay->addKnob (P::delayMix, "Mix");
    delay->addKnob (P::delayHighcut, "High-cut");
    auto* reverb = cards.add (new EffectCard (p, "Reverb", "reverb", P::reverbOn));
    reverb->addKnob (P::reverbDecay, "Decay");
    reverb->addKnob (P::reverbPredelay, "Predelay");
    reverb->addKnob (P::reverbMix, "Mix");
    reverb->addKnob (P::reverbDamping, "Damping");
    auto* doubler = cards.add (new EffectCard (p, "Doubler / width", "doubler", P::doublerOn));
    doubler->addKnob (P::doublerMix, "Mix");
    doubler->addKnob (P::doublerWidth, "Width");
    for (auto* c : cards)
        addAndMakeVisible (c);

    setSize (kWidth, naturalHeight());
    refresh();
    startTimerHz (30);
}

MainPanel::~MainPanel() { stopTimer(); }

int MainPanel::naturalHeight() const
{
    int y = kHeaderHeight + kAnalyzeHeight + kGap, col = 0, rowHeight = 0;
    for (auto* c : cards)
    {
        const int h = c->span == 2 ? kEqHeight : kCardHeight;
        if (col + c->span > 2)
        {
            y += rowHeight + kGap;
            col = 0;
            rowHeight = 0;
        }
        rowHeight = juce::jmax (rowHeight, h);
        col += c->span;
    }
    return y + rowHeight + kGap;
}

void MainPanel::paint (juce::Graphics& g)
{
    g.fillAll (colours::background);
    g.setColour (colours::text);
    g.setFont (juce::FontOptions (22.0f, juce::Font::bold));
    g.drawText ("VOCAL MATCH", 16, 10, 220, 26, juce::Justification::centredLeft);
    g.setColour (colours::dim);
    g.setFont (juce::FontOptions (12.0f));
    g.drawText ("Windward Audio", 16, 36, 220, 16, juce::Justification::centredLeft);
    if (dragOver)
    {
        g.setColour (colours::accent);
        g.drawRect (getLocalBounds(), 3);
    }
}

void MainPanel::resized()
{
    auto header = juce::Rectangle<int> (0, 0, kWidth, kHeaderHeight).reduced (kGap);
    matchKnob.setBounds (header.removeFromRight (100));
    meters.setBounds (header.removeFromRight (56).reduced (0, 6));
    header.removeFromLeft (230);
    header.removeFromTop (10);
    auto buttonRow = header.removeFromTop (28);
    loadButton.setBounds (buttonRow.removeFromLeft (130));
    buttonRow.removeFromLeft (8);
    resetButton.setBounds (buttonRow.removeFromLeft (140));
    buttonRow.removeFromLeft (8);
    abButton.setBounds (buttonRow.removeFromLeft (90));
    status.setBounds (header.withTrimmedTop (2));
    analyze.setBounds (kGap, kHeaderHeight, kWidth - 2 * kGap, kAnalyzeHeight);

    // Two columns; a card with span 2 takes a full row.
    const int colWidth = (kWidth - 3 * kGap) / 2;
    int y = kHeaderHeight + kAnalyzeHeight + kGap, col = 0, rowHeight = 0;
    for (auto* c : cards)
    {
        const int h = c->span == 2 ? kEqHeight : kCardHeight;
        if (col + c->span > 2)
        {
            y += rowHeight + kGap;
            col = 0;
            rowHeight = 0;
        }
        const int x = kGap + col * (colWidth + kGap);
        const int w = c->span == 2 ? kWidth - 2 * kGap : colWidth;
        c->setBounds (x, y, w, h);
        rowHeight = juce::jmax (rowHeight, h);
        col += c->span;
    }
}

void MainPanel::timerCallback()
{
    meters.update();
    for (auto* c : cards)
        c->updateMeter();
    if (++ticks % 3 == 0)
    {
        analyze.update();
        if (processor.getPresetGeneration() != seenGeneration)
            refresh();
    }
}

void MainPanel::refresh()
{
    seenGeneration = processor.getPresetGeneration();
    const auto s = processor.getPresetStatus();
    status.setText (s.isNotEmpty() ? s : "No preset loaded: Analyze below, or Load preset from the analyzer.",
                    juce::dontSendNotification);
    status.setTooltip (status.getText());
    resetButton.setEnabled (processor.getPresetName().isNotEmpty());
    for (auto* c : cards)
        c->refresh();
    matchKnob.refresh();
}

void MainPanel::choosePreset()
{
    chooser = std::make_unique<juce::FileChooser> ("Load a Vocal Match preset", vm::vocalMatchFolder ("Presets"), "*.json");
    chooser->launchAsync (juce::FileBrowserComponent::openMode | juce::FileBrowserComponent::canSelectFiles,
                          [safe = juce::Component::SafePointer<MainPanel> (this)] (const juce::FileChooser& fc)
                          {
                              if (safe != nullptr && fc.getResult() != juce::File())
                                  safe->load (fc.getResult());
                          });
}

void MainPanel::load (const juce::File& file)
{
    processor.loadPresetFile (file);
    refresh();
}

bool MainPanel::isInterestedInFileDrag (const juce::StringArray& files)
{
    const bool json = files.size() == 1 && files[0].endsWithIgnoreCase (".json");
    if (json != dragOver)
    {
        dragOver = json;
        repaint();
    }
    return json;
}

void MainPanel::fileDragExit (const juce::StringArray&)
{
    dragOver = false;
    repaint();
}

void MainPanel::filesDropped (const juce::StringArray& files, int, int)
{
    dragOver = false;
    repaint();
    if (! files.isEmpty())
        load (juce::File (files[0]));
}
} // namespace vm::ui

// --------------------------------------------------------------------------- the editor
VocalMatchEditor::VocalMatchEditor (VocalMatchProcessor& p) : AudioProcessorEditor (&p), panel (p)
{
    lookAndFeel.setColourScheme (juce::LookAndFeel_V4::getMidnightColourScheme());
    lookAndFeel.setColour (juce::ResizableWindow::backgroundColourId, colours::background);
    lookAndFeel.setColour (juce::ToggleButton::tickColourId, colours::accent);
    setLookAndFeel (&lookAndFeel);
    addAndMakeVisible (panel);
    naturalWidth = panel.getWidth();
    naturalHeight = panel.getHeight();
    // Resizable, keeping the proportions: 60 % - 150 % of the natural size.
    setResizable (true, true);
    setResizeLimits (naturalWidth * 6 / 10, naturalHeight * 6 / 10, naturalWidth * 3 / 2, naturalHeight * 3 / 2);
    if (auto* c = getConstrainer())
        c->setFixedAspectRatio ((double) naturalWidth / (double) naturalHeight);
    setSize (naturalWidth, naturalHeight);
}

VocalMatchEditor::~VocalMatchEditor() { setLookAndFeel (nullptr); }

void VocalMatchEditor::resized()
{
    const float scale = (float) getWidth() / (float) juce::jmax (1, naturalWidth);
    panel.setTransform (juce::AffineTransform::scale (scale));
    panel.setBounds (0, 0, naturalWidth, naturalHeight);
}
