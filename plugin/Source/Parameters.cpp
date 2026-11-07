#include "Parameters.h"

namespace vm::params
{
namespace
{
using APF = juce::AudioParameterFloat;
using APB = juce::AudioParameterBool;
using APC = juce::AudioParameterChoice;

juce::ParameterID pid (const juce::String& id, int version = kShellVersion) { return { id, version }; }

// Continuous range with no snapping (presets set exact values) and an optional skew so
// that the centre of the knob sits at `centre`.
juce::NormalisableRange<float> range (float lo, float hi, float centre = 0.0f)
{
    juce::NormalisableRange<float> r (lo, hi);
    if (centre > lo && centre < hi)
        r.setSkewForCentre (centre);
    return r;
}

auto attrs (const juce::String& unit, int decimals)
{
    return juce::AudioParameterFloatAttributes()
        .withLabel (unit)
        .withStringFromValueFunction ([decimals] (float v, int) { return juce::String (v, decimals); });
}

auto group (const juce::String& id, const juce::String& name)
{
    return std::make_unique<juce::AudioProcessorParameterGroup> (id, name, "|");
}
} // namespace

juce::AudioProcessorValueTreeState::ParameterLayout createLayout()
{
    juce::AudioProcessorValueTreeState::ParameterLayout layout;

    layout.add (std::make_unique<APF> (pid (matchAmount), "Match amount", range (0.0f, 100.0f), 100.0f, attrs ("%", 0)));
    layout.add (std::make_unique<APB> (pid (abDry, kMilestone6Version), "A/B: dry", false));

    auto input = group ("input", "Input gain");
    input->addChild (std::make_unique<APB> (pid (inputOn), "Input gain on", true));
    input->addChild (std::make_unique<APF> (pid (inputGain), "Input gain", range (-36.0f, 36.0f), 0.0f, attrs ("dB", 1)));
    layout.add (std::move (input));

    auto eq = group ("eq", "EQ");
    eq->addChild (std::make_unique<APB> (pid (eqOn), "EQ on", true));
    // The analyzer's fixed layout: low shelf, four peaks, high shelf.
    const int defaultTypes[kNumEqBands] = { 0, 1, 1, 1, 1, 2 };
    const float defaultFreqs[kNumEqBands] = { 100.0f, 250.0f, 800.0f, 2500.0f, 5000.0f, 10000.0f };
    const float defaultQs[kNumEqBands] = { 0.7071f, 1.0f, 1.0f, 1.0f, 1.0f, 0.7071f };
    for (int b = 0; b < kNumEqBands; ++b)
    {
        const auto n = "EQ " + juce::String (b + 1) + " ";
        eq->addChild (std::make_unique<APB> (pid (eqBandOn (b)), n + "on", false));
        eq->addChild (std::make_unique<APC> (pid (eqBandType (b)), n + "type", eqTypeNames, defaultTypes[b]));
        eq->addChild (std::make_unique<APF> (pid (eqBandFreq (b)), n + "frequency", range (20.0f, 20000.0f, 1000.0f),
                                             defaultFreqs[b], attrs ("Hz", 0)));
        eq->addChild (std::make_unique<APF> (pid (eqBandGain (b)), n + "gain", range (-12.0f, 12.0f), 0.0f, attrs ("dB", 1)));
        eq->addChild (std::make_unique<APF> (pid (eqBandQ (b)), n + "Q", range (0.1f, 18.0f, 1.0f), defaultQs[b], attrs ("", 2)));
    }
    layout.add (std::move (eq));

    auto comp = group ("compressor", "Compressor");
    comp->addChild (std::make_unique<APB> (pid (compOn), "Compressor on", true));
    comp->addChild (std::make_unique<APF> (pid (compThreshold), "Threshold", range (-60.0f, 0.0f), 0.0f, attrs ("dB", 1)));
    comp->addChild (std::make_unique<APF> (pid (compRatio), "Ratio", range (1.0f, 20.0f, 4.0f), 1.0f, attrs (":1", 2)));
    comp->addChild (std::make_unique<APF> (pid (compAttack), "Attack", range (0.1f, 200.0f, 10.0f), 10.0f, attrs ("ms", 1)));
    comp->addChild (std::make_unique<APF> (pid (compRelease), "Release", range (5.0f, 2000.0f, 150.0f), 100.0f, attrs ("ms", 0)));
    comp->addChild (std::make_unique<APF> (pid (compKnee), "Knee", range (0.0f, 24.0f), 6.0f, attrs ("dB", 1)));
    comp->addChild (std::make_unique<APF> (pid (compMakeup), "Make-up", range (-24.0f, 36.0f), 0.0f, attrs ("dB", 1)));
    layout.add (std::move (comp));

    const int v2 = kMilestone3Version;
    auto sat = group ("saturation", "Saturation");
    sat->addChild (std::make_unique<APB> (pid (satOn, v2), "Saturation on", true));
    sat->addChild (std::make_unique<APF> (pid (satDrive, v2), "Drive", range (0.0f, 36.0f), 0.0f, attrs ("dB", 1)));
    sat->addChild (std::make_unique<APF> (pid (satMix, v2), "Saturation mix", range (0.0f, 100.0f), 0.0f, attrs ("%", 0)));
    layout.add (std::move (sat));

    auto deess = group ("deesser", "De-esser");
    deess->addChild (std::make_unique<APB> (pid (deessOn, v2), "De-esser on", true));
    deess->addChild (std::make_unique<APF> (pid (deessFreq, v2), "De-esser frequency", range (2000.0f, 16000.0f, 6500.0f), 6500.0f, attrs ("Hz", 0)));
    deess->addChild (std::make_unique<APF> (pid (deessThreshold, v2), "De-esser threshold", range (-60.0f, 0.0f), 0.0f, attrs ("dB", 1)));
    deess->addChild (std::make_unique<APF> (pid (deessReduction, v2), "De-esser reduction", range (0.0f, 24.0f), 0.0f, attrs ("dB", 1)));
    layout.add (std::move (deess));

    auto delay = group ("delay", "Delay");
    delay->addChild (std::make_unique<APB> (pid (delayOn, v2), "Delay on", true));
    delay->addChild (std::make_unique<APF> (pid (delayTime, v2), "Delay time", range (1.0f, 2000.0f, 300.0f), 250.0f, attrs ("ms", 1)));
    delay->addChild (std::make_unique<APF> (pid (delayFeedback, v2), "Delay feedback", range (0.0f, 95.0f), 0.0f, attrs ("%", 0)));
    delay->addChild (std::make_unique<APF> (pid (delayMix, v2), "Delay mix", range (0.0f, 100.0f), 0.0f, attrs ("%", 0)));
    delay->addChild (std::make_unique<APF> (pid (delayHighcut, v2), "Delay high-cut", range (500.0f, 20000.0f, 5000.0f), 8000.0f, attrs ("Hz", 0)));
    layout.add (std::move (delay));

    auto reverb = group ("reverb", "Reverb");
    reverb->addChild (std::make_unique<APB> (pid (reverbOn, v2), "Reverb on", true));
    reverb->addChild (std::make_unique<APF> (pid (reverbDecay, v2), "Reverb decay", range (0.1f, 10.0f, 1.5f), 1.5f, attrs ("s", 2)));
    reverb->addChild (std::make_unique<APF> (pid (reverbPredelay, v2), "Reverb predelay", range (0.0f, 250.0f, 40.0f), 0.0f, attrs ("ms", 0)));
    reverb->addChild (std::make_unique<APF> (pid (reverbMix, v2), "Reverb mix", range (0.0f, 100.0f), 0.0f, attrs ("%", 0)));
    reverb->addChild (std::make_unique<APF> (pid (reverbDamping, v2), "Reverb damping", range (500.0f, 20000.0f, 5000.0f), 8000.0f, attrs ("Hz", 0)));
    layout.add (std::move (reverb));

    auto doubler = group ("doubler", "Doubler / width");
    doubler->addChild (std::make_unique<APB> (pid (doublerOn, v2), "Doubler on", true));
    doubler->addChild (std::make_unique<APF> (pid (doublerMix, v2), "Doubler mix", range (0.0f, 100.0f), 0.0f, attrs ("%", 0)));
    doubler->addChild (std::make_unique<APF> (pid (doublerWidth, v2), "Width", range (0.0f, 200.0f), 100.0f, attrs ("%", 0)));
    layout.add (std::move (doubler));

    auto output = group ("output", "Output gain");
    output->addChild (std::make_unique<APB> (pid (outputOn), "Output gain on", true));
    output->addChild (std::make_unique<APF> (pid (outputGain), "Output gain", range (-36.0f, 36.0f), 0.0f, attrs ("dB", 1)));
    layout.add (std::move (output));

    return layout;
}

std::atomic<float>* Reader::get (juce::AudioProcessorValueTreeState& s, const juce::String& id)
{
    auto* p = s.getRawParameterValue (id);
    jassert (p != nullptr);
    return p;
}

Reader::Reader (juce::AudioProcessorValueTreeState& s)
{
    match = get (s, matchAmount);
    inOn = get (s, inputOn);
    inGain = get (s, inputGain);
    outOn = get (s, outputOn);
    outGain = get (s, outputGain);
    eqEnabled = get (s, eqOn);
    for (int b = 0; b < kNumEqBands; ++b)
        bands[(size_t) b] = { get (s, eqBandOn (b)), get (s, eqBandType (b)), get (s, eqBandFreq (b)),
                              get (s, eqBandGain (b)), get (s, eqBandQ (b)) };
    cOn = get (s, compOn);
    cThr = get (s, compThreshold);
    cRatio = get (s, compRatio);
    cAtt = get (s, compAttack);
    cRel = get (s, compRelease);
    cKnee = get (s, compKnee);
    cMakeup = get (s, compMakeup);
    sOn = get (s, satOn);
    sDrive = get (s, satDrive);
    sMix = get (s, satMix);
    dsOn = get (s, deessOn);
    dsFreq = get (s, deessFreq);
    dsThr = get (s, deessThreshold);
    dsRed = get (s, deessReduction);
    dlOn = get (s, delayOn);
    dlTime = get (s, delayTime);
    dlFb = get (s, delayFeedback);
    dlMix = get (s, delayMix);
    dlHc = get (s, delayHighcut);
    rvOn = get (s, reverbOn);
    rvDecay = get (s, reverbDecay);
    rvPre = get (s, reverbPredelay);
    rvMix = get (s, reverbMix);
    rvDamp = get (s, reverbDamping);
    dbOn = get (s, doublerOn);
    dbMix = get (s, doublerMix);
    dbWidth = get (s, doublerWidth);
}

ChainParams Reader::readUnscaled() const noexcept
{
    auto on = [] (const std::atomic<float>* a) { return a->load (std::memory_order_relaxed) >= 0.5f; };
    auto val = [] (const std::atomic<float>* a) { return (double) a->load (std::memory_order_relaxed); };
    ChainParams p;
    p.inputGain = { on (inOn), val (inGain) };
    p.outputGain = { on (outOn), val (outGain) };
    p.eq.enabled = on (eqEnabled);
    for (int b = 0; b < kNumEqBands; ++b)
    {
        const auto& src = bands[(size_t) b];
        auto& dst = p.eq.bands[(size_t) b];
        dst.enabled = on (src.on);
        dst.type = (BiquadType) juce::jlimit (0, 2, (int) std::lround (val (src.type)));
        dst.freqHz = val (src.freq);
        dst.gainDb = val (src.gain);
        dst.q = val (src.q);
    }
    p.compressor = { on (cOn), val (cThr), val (cRatio), val (cAtt), val (cRel), val (cKnee), val (cMakeup) };
    p.saturation = { on (sOn), val (sDrive), val (sMix) };
    p.deesser = { on (dsOn), val (dsFreq), val (dsThr), val (dsRed) };
    p.delay = { on (dlOn), val (dlTime), val (dlFb), val (dlMix), val (dlHc) };
    p.reverb = { on (rvOn), val (rvDecay), val (rvPre), val (rvMix), val (rvDamp) };
    p.doubler = { on (dbOn), val (dbMix), val (dbWidth) };
    return p;
}

ChainParams Reader::read() const noexcept
{
    return applyMatchAmount (readUnscaled(), (double) match->load (std::memory_order_relaxed) / 100.0);
}
} // namespace vm::params
