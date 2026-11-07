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
const juce::Colour wave { 0xff56606e };
const juce::Colour region { 0xff4fb3a9 };
} // namespace colours

// Singer A, B, C, D ... (colour-blind friendly: blue, orange, purple, green)
juce::Colour singerColour (int i)
{
    static const juce::Colour c[] = { juce::Colour (0xff5aa9e6), juce::Colour (0xfff28f3b), juce::Colour (0xffb084f5),
                                      juce::Colour (0xff7bd389), juce::Colour (0xffe56b6f), juce::Colour (0xffd4c35a) };
    return c[(size_t) juce::jlimit (0, 5, i)];
}

constexpr int kHeaderHeight = 66;
constexpr int kGap = 8;
constexpr int kCaption = 12, kReadout = 14;

int knobHeight (int k) { return kCaption + juce::roundToInt (k * 0.62f) + kReadout; }

juce::String timeText (double s)
{
    const int t = (int) std::floor (s + 1e-6);
    return juce::String (t / 60) + ":" + juce::String (t % 60).paddedLeft ('0', 2);
}

void styleLabel (juce::Label& l, float size = 12.0f, juce::Colour c = colours::dim)
{
    l.setFont (juce::FontOptions (size));
    l.setColour (juce::Label::textColourId, c);
    l.setMinimumHorizontalScale (0.75f);
    l.setBorderSize ({ 0, 2, 0, 2 });
}
} // namespace

namespace vm::ui
{
// -------------------------------------------------------------------------- ParamKnob
ParamKnob::ParamKnob (VocalMatchProcessor& p, const juce::String& id, const juce::String& text)
    : paramId (id), processor (p), attachment (p.apvts, id, slider)
{
    caption.setText (text, juce::dontSendNotification);
    caption.setJustificationType (juce::Justification::centred);
    styleLabel (caption, 10.5f);
    caption.setInterceptsMouseClicks (false, false);
    addAndMakeVisible (caption);
    slider.setColour (juce::Slider::rotarySliderFillColourId, colours::accent);
    slider.setColour (juce::Slider::textBoxOutlineColourId, juce::Colours::transparentBlack);
    slider.setColour (juce::Slider::textBoxTextColourId, colours::text);
    slider.setColour (juce::Slider::textBoxBackgroundColourId, juce::Colours::transparentBlack);
    if (auto* param = p.apvts.getParameter (id))
        slider.setTooltip (param->getName (64));
    addAndMakeVisible (slider);
}

void ParamKnob::resized()
{
    auto r = getLocalBounds();
    caption.setBounds (r.removeFromTop (kCaption));
    slider.setTextBoxStyle (juce::Slider::TextBoxBelow, false, getWidth(), kReadout);
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
    auto knob = slider.getBounds().toFloat().withTrimmedBottom ((float) kReadout);
    const float side = juce::jmin (knob.getWidth(), knob.getHeight());
    knob = knob.withSizeKeepingCentre (side, side).reduced (3.0f);
    const float radius = knob.getWidth() / 2.0f;
    const auto centre = knob.getCentre();
    const juce::Point<float> dot (centre.x + radius * std::sin (angle), centre.y - radius * std::cos (angle));
    g.setColour (colours::estimate);
    g.fillEllipse (juce::Rectangle<float> (4.0f, 4.0f).withCentre (dot));
}

// ------------------------------------------------------------------------- EffectCard
EffectCard::EffectCard (VocalMatchProcessor& p, const juce::String& t, const juce::String& key, const juce::String& onId)
    : effectKey (key), processor (p), title (t), onAttachment (p.apvts, onId, onButton)
{
    onButton.setTooltip ("Off = this effect is bypassed");
    onButton.onStateChange = [this] { repaint(); };
    addAndMakeVisible (onButton);
    setRepaintsOnMouseActivity (false);
    collapsed = p.collapsedCards.contains (key);
}

void EffectCard::addKnob (const juce::String& paramId, const juce::String& caption)
{
    addChildComponent (knobs.add (new ParamKnob (processor, paramId, caption)));
}

int EffectCard::naturalWidth (int k) const
{
    // the title bar: arrow, title, confidence, On
    const int titleW = 80 + juce::GlyphArrangement::getStringWidthInt (juce::Font (juce::FontOptions (13.0f, juce::Font::bold)), title)
                       + (confidenceText.isNotEmpty() ? 36 : 0);
    if (collapsed)
        return titleW;
    return juce::jmax (titleW, 12 + knobs.size() * (k + 4) + (meter != nullptr ? 18 : 0));
}

int EffectCard::naturalHeight (int k) const { return collapsed ? kTitleHeight : kTitleHeight + knobHeight (k) + 6; }

void EffectCard::setCollapsed (bool c)
{
    if (c == collapsed)
        return;
    collapsed = c;
    processor.collapsedCards.removeString (effectKey);
    if (c)
        processor.collapsedCards.add (effectKey);
    if (onCollapseChange)
        onCollapseChange();
}

void EffectCard::mouseUp (const juce::MouseEvent& e)
{
    if (e.mouseWasClicked() && e.y < kTitleHeight && e.x < getWidth() - 50)
        setCollapsed (! collapsed);
}

void EffectCard::paint (juce::Graphics& g)
{
    const bool on = onButton.getToggleState();
    const auto r = getLocalBounds().toFloat().reduced (0.5f);
    g.setColour (on ? colours::card : colours::cardOff);
    g.fillRoundedRectangle (r, 5.0f);
    g.setColour (colours::outline);
    g.drawRoundedRectangle (r, 5.0f, 1.0f);
    // collapse arrow
    juce::Path arrow;
    const float ax = 10.0f, ay = kTitleHeight * 0.5f;
    if (collapsed)
        arrow.addTriangle (ax - 2.5f, ay - 4.0f, ax - 2.5f, ay + 4.0f, ax + 3.5f, ay);
    else
        arrow.addTriangle (ax - 4.0f, ay - 2.5f, ax + 4.0f, ay - 2.5f, ax, ay + 3.5f);
    g.setColour (colours::dim);
    g.fillPath (arrow);
    g.setColour (on ? colours::text : colours::dim);
    g.setFont (juce::FontOptions (13.0f, juce::Font::bold));
    const int textRight = getWidth() - 50;
    g.drawText (title, 20, 0, textRight - 20, kTitleHeight, juce::Justification::centredLeft);
    if (confidenceText.isNotEmpty())
    {
        g.setColour (colours::estimate);
        g.setFont (juce::FontOptions (10.5f));
        const int tw = juce::GlyphArrangement::getStringWidthInt (juce::Font (juce::FontOptions (13.0f, juce::Font::bold)), title);
        g.drawText (confidenceText, 26 + tw, 0, juce::jmax (0, textRight - 30 - tw), kTitleHeight, juce::Justification::centredLeft);
    }
}

juce::Rectangle<int> EffectCard::contentArea() const { return getLocalBounds().reduced (6, 0).withTrimmedTop (kTitleHeight).withTrimmedBottom (6); }

void EffectCard::resized()
{
    onButton.setBounds (getWidth() - 48, 2, 44, kTitleHeight - 4);
    auto area = contentArea();
    for (auto* k : knobs)
        k->setVisible (! collapsed);
    if (meter != nullptr)
        meter->setVisible (! collapsed);
    if (collapsed)
        return;
    if (meter != nullptr)
        meter->setBounds (area.removeFromRight (16));
    if (knobs.isEmpty())
        return;
    // knobs evenly spread over the card, each as wide as fits (they never overlap)
    const int n = knobs.size();
    const int k = juce::jmin (area.getWidth() / n - 4, juce::roundToInt ((area.getHeight() - kCaption - kReadout) / 0.62f));
    const int slot = area.getWidth() / n;
    for (int i = 0; i < n; ++i)
        knobs[i]->setBounds (juce::Rectangle<int> (area.getX() + i * slot, area.getY(), slot, area.getHeight())
                                 .withSizeKeepingCentre (juce::jmax (30, k), area.getHeight()));
}

void EffectCard::refresh()
{
    const auto c = processor.getConfidence (effectKey);
    confidenceText = c && *c > 0.0f ? juce::String (juce::roundToInt (*c * 100.0f)) + " %" : juce::String();
    repaint();
    for (auto* k : knobs)
        k->refresh();
}

void EffectCard::addMeter (std::function<float()> source)
{
    meter = std::make_unique<ReductionMeter> (std::move (source));
    meter->setTooltip ("Gain reduction (0 - 12 dB)");
    addChildComponent (*meter);
}

void EffectCard::updateMeter()
{
    if (meter != nullptr && meter->isVisible())
        meter->update();
}

// --------------------------------------------------------------------- ReductionMeter
void ReductionMeter::update()
{
    const float v = juce::jlimit (0.0f, 24.0f, read());
    // fast attack, slower fall, so short reductions stay visible
    const float next = v > shown ? v : shown + 0.35f * (v - shown);
    if (std::abs (next - shown) > 0.02f)
    {
        shown = next;
        repaint();
    }
}

void ReductionMeter::paint (juce::Graphics& g)
{
    auto r = getLocalBounds().toFloat();
    auto label = r.removeFromBottom (12.0f);
    auto bar = r.reduced (4.0f, 4.0f);
    g.setColour (colours::outline);
    g.fillRoundedRectangle (bar, 2.0f);
    g.setColour (colours::estimate);
    g.fillRoundedRectangle (bar.withHeight (bar.getHeight() * juce::jmin (shown, 12.0f) / 12.0f), 2.0f);
    g.setColour (colours::dim);
    g.setFont (juce::FontOptions (9.0f));
    g.drawText ("GR", label, juce::Justification::centred);
}

// ----------------------------------------------------------------------------- EqCard
EqCard::EqCard (VocalMatchProcessor& p) : EffectCard (p, "EQ", "eq", params::eqOn)
{
    for (int b = 0; b < kNumEqBands; ++b)
    {
        auto& band = bands[(size_t) b];
        band.on = std::make_unique<juce::ToggleButton> (juce::String (b + 1));
        band.type = std::make_unique<juce::ComboBox>();
        band.type->addItemList (params::eqTypeNames, 1);
        band.onAttachment = std::make_unique<juce::AudioProcessorValueTreeState::ButtonAttachment> (p.apvts, params::eqBandOn (b), *band.on);
        band.typeAttachment = std::make_unique<juce::AudioProcessorValueTreeState::ComboBoxAttachment> (p.apvts, params::eqBandType (b), *band.type);
        addChildComponent (*band.on);
        addChildComponent (*band.type);
        addKnob (params::eqBandFreq (b), "Freq");
        addKnob (params::eqBandGain (b), "Gain");
        addKnob (params::eqBandQ (b), "Q");
    }
}

int EqCard::naturalWidth (int k) const { return collapsed ? 150 : 12 + kNumEqBands * (3 * k + 12); }
int EqCard::naturalHeight (int k) const { return collapsed ? kTitleHeight : kTitleHeight + 22 + knobHeight (k) + 6; }

void EqCard::resized()
{
    EffectCard::resized();
    for (auto& b : bands)
    {
        b.on->setVisible (! collapsed);
        b.type->setVisible (! collapsed);
    }
    if (collapsed)
        return;
    auto area = contentArea();
    const int bandWidth = area.getWidth() / kNumEqBands;
    for (int b = 0; b < kNumEqBands; ++b)
    {
        auto col = area.removeFromLeft (bandWidth).reduced (3, 0);
        auto top = col.removeFromTop (20);
        bands[(size_t) b].on->setBounds (top.removeFromLeft (36));
        bands[(size_t) b].type->setBounds (top.reduced (0, 1));
        col.removeFromTop (2);
        const int w = col.getWidth() / 3; // freq, gain, Q side by side
        for (int i = 0; i < 3; ++i)
        {
            auto slot = col.removeFromLeft (i < 2 ? w : col.getWidth());
            const int kw = juce::jmin (slot.getWidth() - 2, juce::roundToInt ((slot.getHeight() - kCaption - kReadout) / 0.62f));
            knobs[b * 3 + i]->setBounds (slot.withSizeKeepingCentre (juce::jmax (28, kw), slot.getHeight()));
        }
    }
}

// ------------------------------------------------------------------------ LevelMeters
void LevelMeters::update()
{
    auto db = [] (float lin) { return juce::Decibels::gainToDecibels (lin, -90.0f); };
    const float i = db (processor.inputPeak.exchange (0.0f)), o = db (processor.outputPeak.exchange (0.0f));
    const float ni = i > in ? i : juce::jmax (i, in - 1.5f), no = o > out ? o : juce::jmax (o, out - 1.5f);
    if (std::abs (ni - in) > 0.05f || std::abs (no - out) > 0.05f)
    {
        in = ni;
        out = no;
        repaint();
    }
}

void LevelMeters::paint (juce::Graphics& g)
{
    auto r = getLocalBounds().toFloat();
    auto labels = r.removeFromBottom (12.0f);
    const float w = r.getWidth() / 2.0f;
    auto draw = [&] (juce::Rectangle<float> bar, float db, const juce::String& name, juce::Rectangle<float> label)
    {
        bar = bar.reduced (3.0f, 0.0f);
        g.setColour (colours::outline);
        g.fillRoundedRectangle (bar, 2.0f);
        const float frac = juce::jlimit (0.0f, 1.0f, (db + 60.0f) / 60.0f);
        g.setColour (db > -0.5f ? juce::Colours::orangered : colours::accent);
        g.fillRoundedRectangle (bar.withTrimmedTop (bar.getHeight() * (1.0f - frac)), 2.0f);
        g.setColour (colours::dim);
        g.setFont (juce::FontOptions (9.0f));
        g.drawText (name, label, juce::Justification::centred);
    };
    draw (r.withWidth (w), in, "IN", labels.withWidth (w));
    draw (r.withTrimmedLeft (w), out, "OUT", labels.withTrimmedLeft (w));
}

// -------------------------------------------------------------------------- ChainPage
ChainPage::ChainPage (VocalMatchProcessor& p) : processor (p)
{
    namespace P = vm::params;
    cards.add (new EqCard (p));
    auto* input = cards.add (new EffectCard (p, "Input", "input_gain", P::inputOn));
    input->addKnob (P::inputGain, "Gain");
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
    auto* output = cards.add (new EffectCard (p, "Output", "output_gain", P::outputOn));
    output->addKnob (P::outputGain, "Gain");
    for (auto* c : cards)
    {
        c->onCollapseChange = [this] { resized(); repaint(); };
        addAndMakeVisible (c);
    }
}

void ChainPage::resized()
{
    const int width = getWidth();
    // Flow the cards into rows for knob width k; the height they need.
    struct Row
    {
        int first = 0, count = 0, height = 0, natural = 0;
    };
    auto flow = [&] (int k, std::vector<Row>& rows)
    {
        rows.clear();
        Row row;
        for (int i = 0; i < cards.size(); ++i)
        {
            const int w = juce::jmin (width, cards[i]->naturalWidth (k));
            if (row.count > 0 && row.natural + kGap + w > width)
            {
                rows.push_back (row);
                row = { i, 0, 0, 0 };
            }
            row.natural += (row.count > 0 ? kGap : 0) + w;
            row.height = juce::jmax (row.height, cards[i]->naturalHeight (k));
            ++row.count;
        }
        if (row.count > 0)
            rows.push_back (row);
        int h = 0;
        for (const auto& r : rows)
            h += r.height + kGap;
        return h - kGap;
    };
    std::vector<Row> rows;
    int k = 96;
    // the largest knobs that fit (a card wider than the page is narrowed; its knobs shrink)
    while (k > 40 && flow (k, rows) > getHeight())
        k -= 2;
    flow (k, rows);
    int y = 0;
    for (const auto& r : rows)
    {
        // stretch the row to the full width, in proportion to each card's natural width
        const int extra = width - r.natural;
        int x = 0, used = 0;
        for (int j = 0; j < r.count; ++j)
        {
            auto* c = cards[r.first + j];
            const int w0 = juce::jmin (width, c->naturalWidth (k));
            const int add = j == r.count - 1 ? extra - used : extra * w0 / juce::jmax (1, r.natural - kGap * (r.count - 1));
            used += add;
            const int h = c->isCollapsed() ? EffectCard::kTitleHeight : r.height;
            c->setBounds (x, y, w0 + add, h);
            x += w0 + add + kGap;
        }
        y += r.height + kGap;
    }
}

void ChainPage::refresh()
{
    for (auto* c : cards)
        c->refresh();
    resized(); // confidence labels change the title widths
}

void ChainPage::updateMeters()
{
    if (! isShowing())
        return;
    for (auto* c : cards)
        c->updateMeter();
}

// ------------------------------------------------------------------------ WaveformView
WaveformView::WaveformView (VocalMatchProcessor& p) : processor (p)
{
    setOpaque (true);
    setTooltip ("Drag to mark a section to analyse (several allowed). Drag a region's edge to change it; "
                "right-click or Alt-click a region to remove it; double-click a region to hear it.");
}

juce::Rectangle<float> WaveformView::waveArea() const { return getLocalBounds().toFloat().reduced (1.0f).withTrimmedTop (14.0f).withTrimmedBottom (30.0f); }

double WaveformView::timeAt (float x) const
{
    const auto a = waveArea();
    const double len = juce::jmax (processor.getPeaks().seconds, processor.getScan().durationS);
    return juce::jlimit (0.0, len, (double) (x - a.getX()) / juce::jmax (1.0f, a.getWidth()) * len);
}

float WaveformView::xAt (double t) const
{
    const auto a = waveArea();
    const double len = juce::jmax (processor.getPeaks().seconds, processor.getScan().durationS);
    return len > 0.0 ? a.getX() + (float) (t / len) * a.getWidth() : a.getX();
}

void WaveformView::resized() { background = {}; }

void WaveformView::songChanged()
{
    background = {};
    repaint();
}

void WaveformView::renderBackground()
{
    const int w = juce::jmax (1, juce::roundToInt (getWidth() * backgroundScale)), h = juce::jmax (1, juce::roundToInt (getHeight() * backgroundScale));
    background = juce::Image (juce::Image::RGB, w, h, true);
    juce::Graphics g (background);
    g.addTransform (juce::AffineTransform::scale (backgroundScale));
    g.fillAll (colours::cardOff);
    const auto& peaks = processor.getPeaks();
    const auto& scan = processor.getScan();
    const auto a = waveArea();
    const double len = juce::jmax (peaks.seconds, scan.durationS);
    g.setColour (colours::outline);
    g.drawRect (getLocalBounds(), 1);
    // time ruler
    g.setFont (juce::FontOptions (9.5f));
    if (len > 0.0)
    {
        const double steps[] = { 1, 2, 5, 10, 15, 30, 60, 120 };
        double step = 120;
        for (double s : steps)
            if (a.getWidth() / (len / s) >= 60.0f)
            {
                step = s;
                break;
            }
        for (double t = 0.0; t < len; t += step)
        {
            const float x = xAt (t);
            g.setColour (colours::outline);
            g.drawVerticalLine ((int) x, 12.0f, a.getBottom());
            g.setColour (colours::dim);
            g.drawText (timeText (t), (int) x + 2, 1, 40, 12, juce::Justification::centredLeft);
        }
    }
    // waveform
    if (! peaks.lo.empty())
    {
        const float mid = a.getCentreY(), half = a.getHeight() * 0.48f;
        float peak = 1e-6f;
        for (size_t i = 0; i < peaks.lo.size(); ++i)
            peak = juce::jmax (peak, -peaks.lo[i], peaks.hi[i]);
        const float norm = 1.0f / peak;
        g.setColour (colours::wave);
        const int x0 = (int) a.getX(), x1 = (int) a.getRight();
        for (int x = x0; x < x1; ++x)
        {
            const size_t b0 = (size_t) ((x - x0) * (double) peaks.lo.size() / (x1 - x0));
            const size_t b1 = juce::jmax (b0 + 1, (size_t) ((x + 1 - x0) * (double) peaks.lo.size() / (x1 - x0)));
            float lo = 0.0f, hi = 0.0f;
            for (size_t b = b0; b < juce::jmin (b1, peaks.lo.size()); ++b)
            {
                lo = juce::jmin (lo, peaks.lo[b]);
                hi = juce::jmax (hi, peaks.hi[b]);
            }
            g.drawVerticalLine (x, mid - hi * norm * half, mid - lo * norm * half + 1.0f);
        }
    }
    else
    {
        g.setColour (colours::dim);
        g.setFont (juce::FontOptions (12.0f));
        g.drawText (processor.isSongLoading() ? "Loading the song..."
                    : processor.getSongError().isNotEmpty() ? processor.getSongError()
                    : "Choose the reference song to see its waveform.",
                    a.toNearestInt(), juce::Justification::centred);
    }
    // vocal-activity lane
    auto lanes = getLocalBounds().toFloat().reduced (1.0f).removeFromBottom (30.0f);
    auto vocalLane = lanes.removeFromTop (12.0f).reduced (0.0f, 1.0f);
    auto singerLane = lanes.reduced (0.0f, 2.0f);
    g.setColour (colours::background);
    g.fillRect (vocalLane);
    g.fillRect (singerLane);
    if (scan.isValid() && len > 0.0)
    {
        const int x0 = (int) a.getX(), x1 = (int) a.getRight();
        for (int x = x0; x < x1; ++x)
        {
            const size_t f0 = (size_t) (timeAt ((float) x) / scan.frameS), f1 = juce::jmax (f0 + 1, (size_t) (timeAt ((float) x + 1.0f) / scan.frameS));
            float v = 0.0f;
            for (size_t f = f0; f < juce::jmin (f1, scan.activity.size()); ++f)
                v = juce::jmax (v, scan.activity[f]);
            if (v > 0.02f)
            {
                g.setColour (colours::accent.withAlpha (0.25f + 0.75f * v));
                g.drawVerticalLine (x, vocalLane.getY(), vocalLane.getBottom());
            }
        }
        for (const auto& seg : scan.segments)
        {
            const auto r = juce::Rectangle<float>::leftTopRightBottom (xAt (seg.start), singerLane.getY(), xAt (seg.end), singerLane.getBottom());
            if (seg.singer >= 0)
            {
                g.setColour (singerColour (seg.singer));
                g.fillRect (r);
            }
            else if (seg.singer == -2) // overlap: hatched
            {
                g.setColour (colours::dim.withAlpha (0.5f));
                g.fillRect (r);
                g.setColour (colours::background);
                juce::Graphics::ScopedSaveState save (g);
                g.reduceClipRegion (r.toNearestInt());
                for (float x = r.getX() - r.getHeight(); x < r.getRight(); x += 5.0f)
                    g.drawLine (x, r.getBottom(), x + r.getHeight(), r.getY(), 1.0f);
            }
        }
    }
    g.setColour (colours::dim);
    g.setFont (juce::FontOptions (9.0f));
    g.drawText (scan.isValid() ? "vocal" : "vocal (Find singers)", vocalLane.withWidth (120.0f).translated (3.0f, 0.0f), juce::Justification::centredLeft);
    if (scan.isValid() && ! scan.segments.empty())
        g.drawText ("singers", singerLane.withWidth (60.0f).translated (3.0f, 0.0f), juce::Justification::centredLeft);
}

void WaveformView::paint (juce::Graphics& g)
{
    const float scale = juce::jmax (1.0f, g.getInternalContext().getPhysicalPixelScaleFactor());
    if (! background.isValid() || std::abs (scale - backgroundScale) > 0.01f)
    {
        backgroundScale = scale;
        renderBackground();
    }
    g.drawImage (background, getLocalBounds().toFloat());
    // regions
    const auto a = waveArea();
    for (int i = 0; i < processor.regions.size(); ++i)
    {
        const auto& r = processor.regions.getReference (i);
        const auto box = juce::Rectangle<float>::leftTopRightBottom (xAt (r.getStart()), a.getY(), xAt (r.getEnd()), a.getBottom());
        g.setColour (colours::region.withAlpha (i == selected ? 0.32f : 0.18f));
        g.fillRect (box);
        g.setColour (colours::region);
        g.drawVerticalLine ((int) box.getX(), box.getY(), box.getBottom());
        g.drawVerticalLine ((int) box.getRight(), box.getY(), box.getBottom());
        g.setFont (juce::FontOptions (9.5f));
        if (box.getWidth() > 30.0f)
            g.drawText (juce::String (i + 1), box.withHeight (12.0f).reduced (3.0f, 0.0f), juce::Justification::centredLeft);
    }
    if (playX >= 0.0f)
    {
        g.setColour (colours::text);
        g.drawVerticalLine ((int) playX, 12.0f, (float) getHeight() - 1.0f);
    }
}

void WaveformView::update()
{
    float x = -1.0f;
    if (processor.preview.isPlaying())
        x = xAt (processor.preview.positionSeconds());
    if (std::abs (x - playX) >= 0.5f)
    {
        const auto old = playX;
        playX = x;
        if (old >= 0.0f)
            repaint ((int) old - 2, 0, 5, getHeight());
        if (x >= 0.0f)
            repaint ((int) x - 2, 0, 5, getHeight());
    }
}

int WaveformView::regionAt (float x, int* edge) const
{
    for (int i = processor.regions.size(); --i >= 0;)
    {
        const auto& r = processor.regions.getReference (i);
        const float xa = xAt (r.getStart()), xb = xAt (r.getEnd());
        if (std::abs (x - xa) <= 4.0f)
        {
            *edge = -1;
            return i;
        }
        if (std::abs (x - xb) <= 4.0f)
        {
            *edge = 1;
            return i;
        }
        if (x > xa && x < xb)
        {
            *edge = 0;
            return i;
        }
    }
    *edge = 0;
    return -1;
}

void WaveformView::mouseMove (const juce::MouseEvent& e)
{
    int edge = 0;
    const int i = regionAt (e.position.x, &edge);
    setMouseCursor (i >= 0 && edge != 0 ? juce::MouseCursor::LeftRightResizeCursor : juce::MouseCursor::IBeamCursor);
}

void WaveformView::mouseDown (const juce::MouseEvent& e)
{
    const double len = juce::jmax (processor.getPeaks().seconds, processor.getScan().durationS);
    if (len <= 0.0)
        return;
    int edge = 0;
    const int i = regionAt (e.position.x, &edge);
    if (i >= 0 && (e.mods.isPopupMenu() || e.mods.isAltDown()))
    {
        processor.regions.remove (i);
        selected = -1;
        if (onRegionsChanged)
            onRegionsChanged();
        repaint();
        dragRegion = -2;
        return;
    }
    if (i >= 0 && e.getNumberOfClicks() >= 2)
    {
        const auto& r = processor.regions.getReference (i);
        processor.preview.play (r.getStart(), r.getEnd());
        dragRegion = -2;
        return;
    }
    if (i >= 0 && edge != 0)
    {
        dragRegion = i;
        dragEdge = edge;
        selected = i;
    }
    else
    {
        selected = i;
        dragRegion = -1; // a new region, once the mouse moves
        dragEdge = 0;
        dragAnchor = timeAt (e.position.x);
    }
    repaint();
}

void WaveformView::mouseDrag (const juce::MouseEvent& e)
{
    if (dragRegion == -2)
        return;
    const double t = timeAt (e.position.x);
    if (dragRegion == -1)
    {
        if (e.getDistanceFromDragStart() < 3)
            return;
        processor.regions.add ({ juce::jmin (dragAnchor, t), juce::jmax (dragAnchor, t) });
        dragRegion = processor.regions.size() - 1;
        selected = dragRegion;
        dragEdge = t >= dragAnchor ? 1 : -1;
    }
    auto r = processor.regions[dragRegion];
    if (dragEdge < 0)
        r = { juce::jmin (t, r.getEnd() - 0.1), r.getEnd() };
    else
        r = { r.getStart(), juce::jmax (t, r.getStart() + 0.1) };
    processor.regions.set (dragRegion, r);
    repaint();
}

void WaveformView::mouseUp (const juce::MouseEvent&)
{
    if (dragRegion >= 0)
    {
        // sort and merge overlapping regions
        auto& rs = processor.regions;
        std::sort (rs.begin(), rs.end(), [] (auto a, auto b) { return a.getStart() < b.getStart(); });
        juce::Array<juce::Range<double>> merged;
        for (const auto& r : rs)
        {
            if (! merged.isEmpty() && r.getStart() <= merged.getLast().getEnd())
                merged.getReference (merged.size() - 1).setEnd (juce::jmax (merged.getLast().getEnd(), r.getEnd()));
            else if (r.getLength() >= 0.25)
                merged.add (r);
        }
        rs = merged;
        selected = juce::jmin (selected, rs.size() - 1);
        if (onRegionsChanged)
            onRegionsChanged();
    }
    dragRegion = -1;
    repaint();
}

// ------------------------------------------------------------------------ AnalyzePage
AnalyzePage::AnalyzePage (VocalMatchProcessor& p) : processor (p), waveform (p)
{
    referenceButton.setTooltip ("The song (or a vocal stem) whose vocal sound you want. With source separation installed, a full song works.");
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
    waveform.onRegionsChanged = [this] { refreshPlan(); };
    playButton.setTooltip ("Hear the selected region (or the first region, or the song) through this track's output. "
                           "The track must be playing or monitored.");
    playButton.onClick = [this]
    {
        const auto& rs = processor.regions;
        const double len = processor.getPeaks().seconds;
        if (waveform.selected >= 0 && waveform.selected < rs.size())
            processor.preview.play (rs[waveform.selected].getStart(), rs[waveform.selected].getEnd());
        else if (! rs.isEmpty())
            processor.preview.play (rs[0].getStart(), rs[0].getEnd());
        else
            processor.preview.play (0.0, len);
    };
    stopButton.onClick = [this] { processor.preview.stop(); };
    clearButton.onClick = [this]
    {
        processor.regions.clear();
        waveform.selected = -1;
        waveform.repaint();
        refreshPlan();
    };
    scanButton.setTooltip ("Find the singers in the song (songs with more than one singer): each gets a colour on the waveform");
    scanButton.onClick = [this] { startScan(); };
    hint.setText ("Drag on the waveform to mark sections to analyse.", juce::dontSendNotification);
    perSinger.setTooltip ("With several singers ticked: analyse each one on their own and save one preset per singer");
    perSinger.setToggleState (processor.perSingerPresets, juce::dontSendNotification);
    perSinger.onClick = [this]
    {
        processor.perSingerPresets = perSinger.getToggleState();
        refreshPlan();
    };
    analyzeButton.setTooltip ("Run the analyzer on the reference and your dry vocal, then load its preset");
    analyzeButton.onClick = [this] { startAnalysis(); };
    cancelButton.onClick = [this]
    {
        processor.analyzer.cancel();
        processor.scanner.cancel();
    };
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
    for (auto* l : { &referenceLabel, &dryLabel, &hint, &scanStatus, &status })
        styleLabel (*l);
    styleLabel (singerMessage, 12.0f, colours::text);
    singerMessage.setJustificationType (juce::Justification::topLeft);
    singerMessage.setMinimumHorizontalScale (1.0f); // wrap instead of squeezing
    status.setMinimumHorizontalScale (1.0f);
    styleLabel (plan, 12.0f, colours::estimate);
    status.setJustificationType (juce::Justification::topLeft);
    progressBar.setColour (juce::ProgressBar::foregroundColourId, colours::accent);
    for (juce::Component* c : std::initializer_list<juce::Component*> { &referenceButton, &referenceLabel, &captureButton, &drySource, &dryLabel,
                                                                       &waveform, &playButton, &stopButton, &clearButton, &scanButton, &hint,
                                                                       &scanStatus, &singerMessage, &plan, &perSinger, &analyzeButton,
                                                                       &cancelButton, &copyButton, &progressBar, &status })
        addAndMakeVisible (c);
    captureButton.setToggleState (processor.capture.isArmed(), juce::dontSendNotification);
    if (processor.getPeaks().lo.empty() && processor.referenceFile.existsAsFile() && ! processor.isSongLoading())
        processor.loadReferenceSong();
    refreshLabels();
    update();
}

void AnalyzePage::paint (juce::Graphics& g)
{
    // colour chips of the singer rows
    for (const auto& row : singerRows)
    {
        const auto b = row.match->getBounds();
        g.setColour (row.colour);
        g.fillRoundedRectangle (juce::Rectangle<float> ((float) b.getX() - 14.0f, (float) b.getCentreY() - 5.0f, 10.0f, 10.0f), 2.0f);
    }
}

void AnalyzePage::resized()
{
    auto r = getLocalBounds();
    auto row1 = r.removeFromTop (24);
    referenceButton.setBounds (row1.removeFromLeft (130));
    row1.removeFromLeft (6);
    referenceLabel.setBounds (row1.removeFromLeft (juce::jmax (120, row1.getWidth() / 3)));
    row1.removeFromLeft (6);
    captureButton.setBounds (row1.removeFromLeft (70));
    row1.removeFromLeft (6);
    drySource.setBounds (row1.removeFromLeft (180));
    row1.removeFromLeft (6);
    dryLabel.setBounds (row1);

    auto bottom = r.removeFromBottom (24 + 6 + 110 + 6 + 26 + 4 + 34);
    r.removeFromTop (6);
    r.removeFromBottom (6);
    waveform.setBounds (r);

    auto transport = bottom.removeFromTop (24);
    playButton.setBounds (transport.removeFromLeft (60));
    transport.removeFromLeft (4);
    stopButton.setBounds (transport.removeFromLeft (60));
    transport.removeFromLeft (4);
    clearButton.setBounds (transport.removeFromLeft (100));
    transport.removeFromLeft (8);
    scanButton.setBounds (transport.removeFromRight (110));
    transport.removeFromRight (6);
    scanStatus.setBounds (transport.removeFromRight (juce::jmin (300, transport.getWidth() / 2)));
    hint.setBounds (transport);
    bottom.removeFromTop (6);

    auto singers = bottom.removeFromTop (110);
    auto left = singers.removeFromLeft (singers.getWidth() / 2);
    for (auto& row : singerRows)
    {
        auto line = left.removeFromTop (22);
        line.removeFromLeft (16); // colour chip
        row.match->setBounds (line.removeFromLeft (100));
        row.preview->setBounds (line.removeFromLeft (66).reduced (0, 1));
        line.removeFromLeft (6);
        row.info->setBounds (line);
        left.removeFromTop (2);
    }
    singers.removeFromLeft (10);
    plan.setBounds (singers.removeFromBottom (20));
    singerMessage.setBounds (singers);
    bottom.removeFromTop (6);

    auto run = bottom.removeFromTop (26);
    perSinger.setBounds (run.removeFromLeft (180));
    run.removeFromLeft (6);
    analyzeButton.setBounds (run.removeFromLeft (90));
    run.removeFromLeft (6);
    cancelButton.setBounds (run.removeFromLeft (70));
    run.removeFromLeft (8);
    copyButton.setBounds (run.removeFromRight (120));
    run.removeFromRight (8);
    progressBar.setBounds (run.reduced (0, 3));
    bottom.removeFromTop (4);
    status.setBounds (bottom);
}

void AnalyzePage::refreshLabels()
{
    referenceLabel.setText (processor.referenceFile.existsAsFile() ? processor.referenceFile.getFileName() : "no song chosen",
                            juce::dontSendNotification);
    referenceLabel.setTooltip (processor.referenceFile.getFullPathName());
    if (processor.useCapturedDry)
    {
        const double s = processor.capture.seconds();
        juce::String t = s > 0.0 ? timeText (s) + " captured" : "nothing captured yet";
        if (processor.capture.isArmed())
            t << " (recording while the song plays)";
        if (processor.capture.isFull())
            t << " (full)";
        dryLabel.setText (t, juce::dontSendNotification);
    }
    else
        dryLabel.setText (processor.dryFile.existsAsFile() ? processor.dryFile.getFileName() : "no file chosen", juce::dontSendNotification);
}

void AnalyzePage::rebuildSingerRows()
{
    for (auto& row : singerRows)
    {
        removeChildComponent (row.match.get());
        removeChildComponent (row.preview.get());
        removeChildComponent (row.info.get());
    }
    singerRows.clear();
    const auto& scan = processor.getScan();
    for (int i = 0; i < juce::jmin (4, scan.singers.size()); ++i)
    {
        const auto singer = scan.singers[i];
        SingerRow row;
        row.colour = singerColour (i);
        row.match = std::make_unique<juce::ToggleButton> ("Singer " + singer.id);
        row.match->setTooltip ("Analyse this singer (tick several to analyse them together, or for one preset each)");
        row.match->setToggleState (processor.chosenSingers.contains (singer.id), juce::dontSendNotification);
        row.match->onClick = [this, id = singer.id, b = row.match.get()]
        {
            processor.chosenSingers.removeString (id);
            if (b->getToggleState())
                processor.chosenSingers.add (id);
            refreshPlan();
        };
        row.preview = std::make_unique<juce::TextButton> ("Preview");
        row.preview->setTooltip ("Hear a few seconds of this singer");
        row.preview->onClick = [this, range = singer.preview] { processor.preview.play (range.getStart(), range.getEnd()); };
        row.info = std::make_unique<juce::Label>();
        styleLabel (*row.info, 11.5f);
        juce::String info = juce::String (singer.seconds, 0) + " s";
        if (singer.medianF0 > 0.0)
            info << ", around " << juce::String (juce::roundToInt (singer.medianF0)) << " Hz";
        row.info->setText (info, juce::dontSendNotification);
        addAndMakeVisible (*row.match);
        addAndMakeVisible (*row.preview);
        addAndMakeVisible (*row.info);
        singerRows.push_back (std::move (row));
    }
    juce::String msg;
    if (scan.isValid())
    {
        msg = scan.message;
        if (scan.singers.size() > 4)
            msg << " (the first 4 are listed)";
        msg << " Confidence " << juce::roundToInt (scan.confidence * 100.0) << " %.";
        if (! scan.separated)
            msg << " (The reference was used as a vocal stem.)";
    }
    else
        msg = "One singer? Just Analyze. Duets: Find singers, then tick who to match, or mark the sections by hand.";
    singerMessage.setText (msg, juce::dontSendNotification);
    singerMessage.setColour (juce::Label::textColourId, scan.isValid() && scan.unsure ? colours::estimate : colours::text);
    resized();
    repaint();
    refreshPlan();
}

juce::String AnalyzePage::planText() const
{
    double regionS = 0.0;
    for (const auto& r : processor.regions)
        regionS += r.getLength();
    const auto& scan = processor.getScan();
    juce::StringArray chosen;
    double singerS = 0.0;
    for (const auto& s : scan.singers)
        if (processor.chosenSingers.contains (s.id))
        {
            chosen.add (s.id);
            singerS += s.seconds;
        }
    const int nr = processor.regions.size();
    const juce::String regionsText = juce::String (nr) + (nr == 1 ? " marked region" : " marked regions") + " (" + juce::String (regionS, 0) + " s)";
    juce::String t = "Will analyse: ";
    if (! chosen.isEmpty())
    {
        t << (chosen.size() > 1 ? "Singers " : "Singer ") << chosen.joinIntoString (" + ");
        if (nr > 0)
            t << " within " << regionsText;
        else
            t << " (" << juce::String (singerS, 0) << " s)";
        if (scan.overlapS > 0.0)
            t << ", overlaps excluded";
        if (chosen.size() > 1 && processor.perSingerPresets)
            t << "; one preset per singer";
    }
    else if (nr > 0)
        t << regionsText;
    else
        t << "the whole vocal";
    return t;
}

void AnalyzePage::refreshPlan()
{
    plan.setText (planText(), juce::dontSendNotification);
    plan.setTooltip (plan.getText());
    perSinger.setEnabled (processor.chosenSingers.size() > 1);
}

void AnalyzePage::chooseReference()
{
    chooser = std::make_unique<juce::FileChooser> ("Choose the reference song or vocal stem", processor.referenceFile,
                                                   "*.wav;*.flac;*.mp3;*.aif;*.aiff");
    chooser->launchAsync (juce::FileBrowserComponent::openMode | juce::FileBrowserComponent::canSelectFiles,
                          [safe = juce::Component::SafePointer<AnalyzePage> (this)] (const juce::FileChooser& fc)
                          {
                              if (safe != nullptr && fc.getResult() != juce::File())
                              {
                                  safe->processor.referenceFile = fc.getResult();
                                  safe->referenceChanged();
                              }
                          });
}

void AnalyzePage::referenceChanged()
{
    processor.preview.stop();
    processor.regions.clear();
    processor.clearScan();
    waveform.selected = -1;
    processor.loadReferenceSong();
    // a scan of this song from before (e.g. run in Terminal)
    const auto cached = processor.scanFileFor (processor.referenceFile);
    if (cached.existsAsFile() && cached.getLastModificationTime() >= processor.referenceFile.getLastModificationTime())
        processor.loadScanFile (cached);
    refreshLabels();
    update();
}

void AnalyzePage::chooseDry()
{
    chooser = std::make_unique<juce::FileChooser> ("Choose your dry vocal", processor.dryFile, "*.wav;*.flac;*.mp3;*.aif;*.aiff");
    chooser->launchAsync (juce::FileBrowserComponent::openMode | juce::FileBrowserComponent::canSelectFiles,
                          [safe = juce::Component::SafePointer<AnalyzePage> (this)] (const juce::FileChooser& fc)
                          {
                              if (safe != nullptr && fc.getResult() != juce::File())
                              {
                                  safe->processor.dryFile = fc.getResult();
                                  safe->refreshLabels();
                              }
                          });
}

void AnalyzePage::startScan()
{
    if (! processor.referenceFile.existsAsFile())
    {
        scanStatus.setText ("Choose the reference song first.", juce::dontSendNotification);
        return;
    }
    vm::ScanRequest r;
    r.reference = processor.referenceFile;
    r.outJson = processor.scanFileFor (r.reference);
    lastCommand = vm::AnalyzerRun::shellCommandFor (r);
    if (! processor.scanner.start (r))
        scanStatus.setText (processor.scanner.getError(), juce::dontSendNotification);
    waitingForScan = r.outJson;
    update();
}

void AnalyzePage::timerCallback()
{
    // The scan was copied to Terminal: pick its result up when it appears.
    if (waitingForScan.existsAsFile() && waitingForScan.getLastModificationTime() > juce::Time::getCurrentTime() - juce::RelativeTime::hours (1))
        if (processor.loadScanFile (waitingForScan))
        {
            stopTimer();
            waitingForScan = juce::File();
            scanStatus.setText ("Singers found.", juce::dontSendNotification);
        }
}

void AnalyzePage::startAnalysis()
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

void AnalyzePage::update()
{
    using State = vm::AnalyzerRun::State;
    processor.pollSong();
    if (processor.getSongGeneration() != seenSongGeneration || processor.getScanGeneration() != seenScanGeneration)
    {
        if (processor.getScanGeneration() != seenScanGeneration)
        {
            seenScanGeneration = processor.getScanGeneration();
            rebuildSingerRows();
        }
        seenSongGeneration = processor.getSongGeneration();
        waveform.songChanged();
    }
    waveform.update();

    // the singers scan
    const auto scanState = processor.scanner.getState();
    if (scanState == State::running)
        scanStatus.setText ("Finding singers: " + processor.scanner.getStage(), juce::dontSendNotification);
    if (scanState != lastScanState)
    {
        if (scanState == State::finished)
        {
            scanStatus.setText (processor.loadScanFile (processor.scanner.getPresetFile()) ? "Singers found." : "The singer scan could not be read.",
                                juce::dontSendNotification);
            waitingForScan = juce::File();
        }
        else if (scanState == State::failed)
        {
            auto msg = processor.scanner.getError();
            if (lastCommand.startsWith ("vocalmatch singers"))
            {
                juce::SystemClipboard::copyTextToClipboard (lastCommand);
                msg << " (Command copied: paste it in Terminal; the result appears here.)";
                startTimer (1000);
            }
            scanStatus.setText (msg, juce::dontSendNotification);
        }
        else if (scanState == State::cancelled)
            scanStatus.setText ("Cancelled.", juce::dontSendNotification);
        lastScanState = scanState;
    }
    scanStatus.setTooltip (scanStatus.getText());
    scanButton.setEnabled (scanState != State::running);

    // the analysis
    const auto state = processor.analyzer.getState();
    progressValue = state == State::running ? (double) processor.analyzer.getProgress() : (state == State::finished ? 1.0 : 0.0);
    analyzeButton.setEnabled (state != State::running);
    cancelButton.setEnabled (state == State::running || scanState == State::running);
    if (state == State::running)
        status.setText (processor.analyzer.getStage(), juce::dontSendNotification);
    if (state != lastState)
    {
        if (state == State::finished)
        {
            const auto preset = processor.analyzer.getPresetFile();
            juce::String msg = "Done: " + preset.getFileName();
            const auto perSingerFiles = preset.getParentDirectory().findChildFiles (juce::File::findFiles, false,
                                                                                    preset.getFileNameWithoutExtension() + " - Singer *.json");
            if (! perSingerFiles.isEmpty())
            {
                juce::StringArray names;
                for (const auto& f : perSingerFiles)
                    names.add (f.getFileName());
                names.sort (true);
                msg << ". One preset per singer saved: " << names.joinIntoString (", ") << " (Load preset to switch).";
            }
            status.setText (msg, juce::dontSendNotification);
            if (onPresetReady)
                onPresetReady (preset);
        }
        else if (state == State::failed)
        {
            auto msg = processor.analyzer.getError();
            if (lastCommand.startsWith ("vocalmatch analyze"))
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
    playButton.setEnabled (! processor.getPeaks().lo.empty());
    for (auto& row : singerRows)
        row.preview->setEnabled (! processor.getPeaks().lo.empty());
    if (processor.useCapturedDry)
        refreshLabels();
}

// --------------------------------------------------------------------------- MainPanel
MainPanel::MainPanel (VocalMatchProcessor& p)
    : processor (p), abAttachment (p.apvts, vm::params::abDry, abButton),
      matchKnob (p, vm::params::matchAmount, "Match"), meters (p), analyzePage (p), chainPage (p)
{
    analyzeTab.setRadioGroupId (1);
    chainTab.setRadioGroupId (1);
    for (auto* t : { &analyzeTab, &chainTab })
    {
        t->setClickingTogglesState (true);
        t->setColour (juce::TextButton::buttonOnColourId, colours::accent.darker (0.4f));
        t->setConnectedEdges (t == &analyzeTab ? juce::Button::ConnectedOnRight : juce::Button::ConnectedOnLeft);
    }
    analyzeTab.onClick = [this] { showPage (0); };
    chainTab.onClick = [this] { showPage (1); };
    loadButton.setTooltip ("Open a preset.json written by the analyzer (or drop one on this window)");
    loadButton.onClick = [this] { choosePreset(); };
    resetButton.setTooltip ("Set every setting back to the loaded preset's estimates");
    resetButton.onClick = [this] { processor.resetToEstimates(); };
    abButton.setClickingTogglesState (true);
    abButton.setColour (juce::TextButton::buttonOnColourId, colours::estimate.darker (0.3f));
    abButton.setTooltip ("Hear your dry input instead of the processed signal (a 20 ms crossfade; the effects keep running)");
    styleLabel (status, 11.5f);
    for (juce::Component* c : std::initializer_list<juce::Component*> { &analyzeTab, &chainTab, &loadButton, &resetButton, &abButton,
                                                                       &status, &matchKnob, &meters })
        addAndMakeVisible (c);
    addChildComponent (analyzePage);
    addChildComponent (chainPage);
    analyzePage.onPresetReady = [this] (const juce::File& f) { load (f); };
    showPage (p.editorPage);
    setSize (kDesignWidth, kDesignHeight);
    refresh();
    startTimerHz (30);
}

MainPanel::~MainPanel() { stopTimer(); }

void MainPanel::showPage (int page)
{
    page = juce::jlimit (0, 1, page);
    processor.editorPage = page;
    (page == 0 ? analyzeTab : chainTab).setToggleState (true, juce::dontSendNotification);
    analyzePage.setVisible (page == 0);
    chainPage.setVisible (page == 1);
}

void MainPanel::paint (juce::Graphics& g)
{
    g.fillAll (colours::background);
    g.setColour (colours::text);
    g.setFont (juce::FontOptions (16.0f, juce::Font::bold));
    g.drawText ("VOCAL MATCH", 12, 4, 160, 22, juce::Justification::centredLeft);
    g.setColour (colours::outline);
    g.drawHorizontalLine (kHeaderHeight, 0.0f, (float) getWidth());
    if (dragOver)
    {
        g.setColour (colours::accent);
        g.drawRect (getLocalBounds(), 3);
    }
}

void MainPanel::resized()
{
    auto header = getLocalBounds().removeFromTop (kHeaderHeight).reduced (kGap, 4);
    matchKnob.setBounds (header.removeFromRight (64));
    header.removeFromRight (4);
    meters.setBounds (header.removeFromRight (36).reduced (0, 2));
    header.removeFromRight (10);
    auto left = header.removeFromLeft (164);
    left.removeFromTop (26);
    analyzeTab.setBounds (left.removeFromLeft (72).withHeight (22));
    chainTab.setBounds (left.removeFromLeft (72).withHeight (22));
    header.removeFromLeft (8);
    auto buttons = header.removeFromTop (24);
    loadButton.setBounds (buttons.removeFromLeft (110));
    buttons.removeFromLeft (6);
    resetButton.setBounds (buttons.removeFromLeft (130));
    buttons.removeFromLeft (6);
    abButton.setBounds (buttons.removeFromLeft (80));
    header.removeFromTop (2);
    status.setBounds (header);
    const auto body = getLocalBounds().withTrimmedTop (kHeaderHeight + kGap).reduced (kGap, 0).withTrimmedBottom (kGap);
    analyzePage.setBounds (body);
    chainPage.setBounds (body);
}

void MainPanel::timerCallback()
{
    meters.update();
    chainPage.updateMeters();
    if (++ticks % 3 == 0)
    {
        analyzePage.update();
        if (processor.getPresetGeneration() != seenGeneration)
            refresh();
    }
}

void MainPanel::refresh()
{
    seenGeneration = processor.getPresetGeneration();
    const auto s = processor.getPresetStatus();
    status.setText (s.isNotEmpty() ? s : "No preset loaded: Analyze, or Load preset from the analyzer.", juce::dontSendNotification);
    status.setTooltip (status.getText());
    resetButton.setEnabled (processor.getPresetName().isNotEmpty());
    chainPage.refresh();
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

// ------------------------------------------------------------------------- CompactLook
CompactLook::CompactLook()
{
    setColourScheme (getMidnightColourScheme());
    setColour (juce::ResizableWindow::backgroundColourId, colours::background);
    setColour (juce::ToggleButton::tickColourId, colours::accent);
    setColour (juce::TooltipWindow::backgroundColourId, colours::card);
}

void CompactLook::drawRotarySlider (juce::Graphics& g, int x, int y, int width, int height, float pos, float start, float end,
                                    juce::Slider& slider)
{
    const auto fill = slider.findColour (juce::Slider::rotarySliderFillColourId);
    const float side = (float) juce::jmin (width, height);
    const auto bounds = juce::Rectangle<float> ((float) x, (float) y, (float) width, (float) height).withSizeKeepingCentre (side, side).reduced (3.0f);
    const float radius = bounds.getWidth() / 2.0f;
    const float line = juce::jmax (2.0f, radius * 0.16f);
    const float arcRadius = radius - line * 0.5f;
    const float angle = start + pos * (end - start);
    const auto c = bounds.getCentre();
    juce::Path back;
    back.addCentredArc (c.x, c.y, arcRadius, arcRadius, 0.0f, start, end, true);
    g.setColour (colours::outline);
    g.strokePath (back, juce::PathStrokeType (line, juce::PathStrokeType::curved, juce::PathStrokeType::rounded));
    if (slider.isEnabled())
    {
        juce::Path value;
        value.addCentredArc (c.x, c.y, arcRadius, arcRadius, 0.0f, start, angle, true);
        g.setColour (fill);
        g.strokePath (value, juce::PathStrokeType (line, juce::PathStrokeType::curved, juce::PathStrokeType::rounded));
    }
    // the knob body and its pointer
    const float body = arcRadius - line * 1.2f;
    g.setColour (colours::card.brighter (0.15f));
    g.fillEllipse (juce::Rectangle<float> (2.0f * body, 2.0f * body).withCentre (c));
    g.setColour (colours::text);
    const juce::Point<float> tip (c.x + body * 0.85f * std::sin (angle), c.y - body * 0.85f * std::cos (angle));
    g.drawLine ({ c.getPointOnCircumference (body * 0.25f, angle), tip }, juce::jmax (1.5f, line * 0.6f));
}

// ------------------------------------------------------------------------ layout check
juce::StringArray checkLayout (juce::Component& root)
{
    juce::StringArray problems;
    std::function<void (juce::Component&, const juce::String&)> visit = [&] (juce::Component& c, const juce::String& path)
    {
        // Sliders, combo boxes and buttons draw their own internals: not checked inside.
        if (dynamic_cast<juce::Slider*> (&c) != nullptr || dynamic_cast<juce::ComboBox*> (&c) != nullptr
            || dynamic_cast<juce::Button*> (&c) != nullptr || dynamic_cast<juce::Label*> (&c) != nullptr)
            return;
        juce::Array<juce::Component*> kids;
        for (auto* k : c.getChildren())
            if (k->isVisible() && dynamic_cast<juce::TooltipWindow*> (k) == nullptr)
                kids.add (k);
        auto name = [] (juce::Component* k)
        {
            if (auto* b = dynamic_cast<juce::Button*> (k))
                return "'" + b->getButtonText() + "'";
            if (auto* l = dynamic_cast<juce::Label*> (k))
                return "label '" + l->getText().substring (0, 24) + "'";
            return juce::String (typeid (*k).name());
        };
        for (int i = 0; i < kids.size(); ++i)
        {
            const auto b = kids[i]->getBounds();
            if (b.isEmpty())
                problems.add (path + "/" + name (kids[i]) + " has no size");
            else if (! c.getLocalBounds().contains (b))
                problems.add (path + "/" + name (kids[i]) + " " + b.toString() + " outside " + c.getLocalBounds().toString());
            for (int j = i + 1; j < kids.size(); ++j)
                if (b.intersects (kids[j]->getBounds()))
                    problems.add (path + ": " + name (kids[i]) + " " + b.toString() + " overlaps " + name (kids[j]) + " " + kids[j]->getBounds().toString());
            visit (*kids[i], path + "/" + name (kids[i]));
        }
    };
    // The editor holds the scaled panel (and the resize corner): the panel must cover the
    // editor exactly; inside it, everything is checked.
    if (auto* editor = dynamic_cast<VocalMatchEditor*> (&root))
    {
        auto& panel = editor->getPanel();
        const auto covered = panel.getBoundsInParent();
        if (std::abs (covered.getWidth() - root.getWidth()) > 1 || std::abs (covered.getHeight() - root.getHeight()) > 1)
            problems.add ("editor: the panel covers " + covered.toString() + " of " + root.getLocalBounds().toString());
        visit (panel, "panel");
    }
    else
        visit (root, "root");
    return problems;
}
} // namespace vm::ui

// --------------------------------------------------------------------------- the editor
VocalMatchEditor::VocalMatchEditor (VocalMatchProcessor& p) : AudioProcessorEditor (&p), panel (p), processor (p)
{
    using namespace vm::ui;
    setLookAndFeel (&lookAndFeel);
    addAndMakeVisible (panel);
    // Free aspect ratio: the host's own resize corner (and ours) can take any shape; the
    // layout scales by the smaller ratio and the pages use the extra room.
    // (read the saved size first: setting the limits resizes the editor, and resized() saves)
    const int w = juce::jlimit (kMinWidth, kMaxWidth, p.editorWidth), h = juce::jlimit (kMinHeight, kMaxHeight, p.editorHeight);
    setResizable (true, true);
    setResizeLimits (kMinWidth, kMinHeight, kMaxWidth, kMaxHeight);
    setSize (w, h);
}

VocalMatchEditor::~VocalMatchEditor() { setLookAndFeel (nullptr); }

void VocalMatchEditor::resized()
{
    using namespace vm::ui;
    // uniform scale, at most a little below 1 at the minimum size (fonts stay readable)
    const float scale = juce::jmin ((float) getWidth() / (float) kDesignWidth, (float) getHeight() / (float) kDesignHeight);
    const int w = juce::roundToInt ((float) getWidth() / scale), h = juce::roundToInt ((float) getHeight() / scale);
    panel.setTransform (juce::AffineTransform::scale (scale));
    panel.setBounds (0, 0, w, h);
    processor.editorWidth = getWidth();
    processor.editorHeight = getHeight();
}
