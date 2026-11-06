#pragma once
// Every setting of the chain as an automatable AudioProcessorValueTreeState parameter.
// IDs are stable (saved in songs); new effects add parameters with a higher version hint.

#include "DSP/Chain.h"

#include <juce_audio_processors/juce_audio_processors.h>

namespace vm::params
{
// Parameter version hints (juce::ParameterID): 1 = the plug-in shell (gain, EQ, compressor).
constexpr int kShellVersion = 1;

inline const juce::String matchAmount { "match_amount" };

inline const juce::String inputOn { "input_on" }, inputGain { "input_gain_db" };
inline const juce::String outputOn { "output_on" }, outputGain { "output_gain_db" };

inline const juce::String eqOn { "eq_on" };
inline juce::String eqBandOn (int band) { return "eq" + juce::String (band + 1) + "_on"; }
inline juce::String eqBandType (int band) { return "eq" + juce::String (band + 1) + "_type"; }
inline juce::String eqBandFreq (int band) { return "eq" + juce::String (band + 1) + "_freq_hz"; }
inline juce::String eqBandGain (int band) { return "eq" + juce::String (band + 1) + "_gain_db"; }
inline juce::String eqBandQ (int band) { return "eq" + juce::String (band + 1) + "_q"; }

inline const juce::String compOn { "comp_on" }, compThreshold { "comp_threshold_db" }, compRatio { "comp_ratio" },
    compAttack { "comp_attack_ms" }, compRelease { "comp_release_ms" }, compKnee { "comp_knee_db" },
    compMakeup { "comp_makeup_db" };

// EQ band types, in the order of BiquadType's first three values.
inline const juce::StringArray eqTypeNames { "Low shelf", "Peak", "High shelf" };
inline const juce::StringArray eqTypeKeys { "low_shelf", "peak", "high_shelf" }; // preset.json spelling

juce::AudioProcessorValueTreeState::ParameterLayout createLayout();

// Reads the parameters (lock-free atomics) into the chain's settings, match amount applied.
class Reader
{
public:
    explicit Reader (juce::AudioProcessorValueTreeState& state);
    ChainParams read() const noexcept;
    ChainParams readUnscaled() const noexcept;

private:
    struct Band
    {
        std::atomic<float>*on, *type, *freq, *gain, *q;
    };
    std::atomic<float>* get (juce::AudioProcessorValueTreeState&, const juce::String& id);
    std::atomic<float>* match;
    std::atomic<float>*inOn, *inGain, *outOn, *outGain;
    std::atomic<float>* eqEnabled;
    std::array<Band, kNumEqBands> bands;
    std::atomic<float>*cOn, *cThr, *cRatio, *cAtt, *cRel, *cKnee, *cMakeup;
};
} // namespace vm::params
