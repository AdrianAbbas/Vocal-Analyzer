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

constexpr int kHeaderHeight = 64;
constexpr int kCardHeight = 128;
constexpr int kEqHeight = 196;
constexpr int kWidth = 940;
constexpr int kGap = 8;
constexpr int kKnobWidth = 64;
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
    for (auto* k : knobs)
        k->setBounds (area.removeFromLeft (kKnobWidth + 6).withTrimmedRight (6));
}

void EffectCard::refresh()
{
    const auto c = processor.getConfidence (effectKey);
    confidence.setText (c ? "estimated, confidence " + juce::String (juce::roundToInt (*c * 100.0f)) + " %" : juce::String(),
                        juce::dontSendNotification);
    repaint();
    for (auto* k : knobs)
        k->repaint();
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
        auto upper = col.removeFromTop (col.getHeight() / 2 + 8);
        knobs[b * 3 + 0]->setBounds (upper.removeFromLeft (upper.getWidth() / 2));
        knobs[b * 3 + 1]->setBounds (upper);
        knobs[b * 3 + 2]->setBounds (col.withSizeKeepingCentre (upper.getWidth(), col.getHeight()));
    }
}
} // namespace vm::ui

// --------------------------------------------------------------------------- the editor
VocalMatchEditor::VocalMatchEditor (VocalMatchProcessor& p)
    : AudioProcessorEditor (&p), processor (p), matchKnob (p, vm::params::matchAmount, "Match amount")
{
    lookAndFeel.setColourScheme (juce::LookAndFeel_V4::getMidnightColourScheme());
    lookAndFeel.setColour (juce::ResizableWindow::backgroundColourId, colours::background);
    lookAndFeel.setColour (juce::ToggleButton::tickColourId, colours::accent);
    setLookAndFeel (&lookAndFeel);

    loadButton.setTooltip ("Open a preset.json written by the Vocal Match analyzer (or drop one on this window)");
    loadButton.onClick = [this] { choosePreset(); };
    addAndMakeVisible (loadButton);
    status.setColour (juce::Label::textColourId, colours::dim);
    status.setFont (juce::FontOptions (13.0f));
    addAndMakeVisible (status);
    addAndMakeVisible (matchKnob);

    namespace P = vm::params;
    using vm::ui::EffectCard;
    auto* input = cards.add (new EffectCard (p, "Input gain", "input_gain", P::inputOn));
    input->addKnob (P::inputGain, "Gain");
    cards.add (new vm::ui::EqCard (p));
    auto* comp = cards.add (new EffectCard (p, "Compressor", "compressor", P::compOn));
    comp->addKnob (P::compThreshold, "Threshold");
    comp->addKnob (P::compRatio, "Ratio");
    comp->addKnob (P::compAttack, "Attack");
    comp->addKnob (P::compRelease, "Release");
    comp->addKnob (P::compKnee, "Knee");
    comp->addKnob (P::compMakeup, "Make-up");
    auto* output = cards.add (new EffectCard (p, "Output gain", "output_gain", P::outputOn));
    output->addKnob (P::outputGain, "Gain");
    for (auto* c : cards)
        addAndMakeVisible (c);

    setSize (kWidth, 400); // resized() sets the height the cards need
    resized();
    refresh();
    startTimerHz (5);
}

VocalMatchEditor::~VocalMatchEditor() { setLookAndFeel (nullptr); }

void VocalMatchEditor::paint (juce::Graphics& g)
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

void VocalMatchEditor::resized()
{
    auto header = juce::Rectangle<int> (0, 0, kWidth, kHeaderHeight).reduced (kGap);
    matchKnob.setBounds (header.removeFromRight (90).withTrimmedTop (-4));
    header.removeFromLeft (230);
    auto buttonRow = header.removeFromTop (28);
    loadButton.setBounds (buttonRow.removeFromLeft (140));
    status.setBounds (header.withTrimmedTop (2));

    // Two columns; a card with span 2 takes a full row.
    const int colWidth = (kWidth - 3 * kGap) / 2;
    int y = kHeaderHeight, col = 0, rowHeight = 0;
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
    const int height = y + rowHeight + kGap;
    if (getHeight() != height)
        setSize (kWidth, height);
}

void VocalMatchEditor::timerCallback()
{
    if (processor.getPresetGeneration() != seenGeneration)
        refresh();
}

void VocalMatchEditor::refresh()
{
    seenGeneration = processor.getPresetGeneration();
    const auto s = processor.getPresetStatus();
    status.setText (s.isNotEmpty() ? s : "No preset loaded. Run the analyzer, then load its preset.json here.",
                    juce::dontSendNotification);
    status.setTooltip (status.getText());
    for (auto* c : cards)
        c->refresh();
    matchKnob.repaint();
}

void VocalMatchEditor::choosePreset()
{
    chooser = std::make_unique<juce::FileChooser> ("Load a Vocal Match preset", juce::File(), "*.json");
    chooser->launchAsync (juce::FileBrowserComponent::openMode | juce::FileBrowserComponent::canSelectFiles,
                          [safe = juce::Component::SafePointer<VocalMatchEditor> (this)] (const juce::FileChooser& fc)
                          {
                              if (safe == nullptr)
                                  return;
                              const auto file = fc.getResult();
                              if (file != juce::File())
                                  safe->load (file);
                          });
}

void VocalMatchEditor::load (const juce::File& file)
{
    processor.loadPresetFile (file);
    refresh();
}

bool VocalMatchEditor::isInterestedInFileDrag (const juce::StringArray& files)
{
    const bool json = files.size() == 1 && files[0].endsWithIgnoreCase (".json");
    if (json != dragOver)
    {
        dragOver = json;
        repaint();
    }
    return json;
}

void VocalMatchEditor::fileDragExit (const juce::StringArray&)
{
    dragOver = false;
    repaint();
}

void VocalMatchEditor::filesDropped (const juce::StringArray& files, int, int)
{
    dragOver = false;
    repaint();
    if (! files.isEmpty())
        load (juce::File (files[0]));
}
