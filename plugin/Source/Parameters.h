#pragma once
// Every setting of the chain as an automatable AudioProcessorValueTreeState parameter.
// IDs are stable (saved in songs); new effects add parameters with a higher version hint.

#include "DSP/Chain.h"

#include <juce_audio_processors/juce_audio_processors.h>

namespace vm::params
{
// Parameter version hints (juce::ParameterID): 1 = the plug-in shell (gain, EQ, compressor),
// 2 = saturation, de-esser, delay, reverb, doubler/width (milestone 3).
constexpr int kShellVersion = 1;
constexpr int kMilestone3Version = 2;

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

inline const juce::String satOn { "sat_on" }, satDrive { "sat_drive_db" }, satMix { "sat_mix_pct" };
inline const juce::String deessOn { "deess_on" }, deessFreq { "deess_freq_hz" }, deessThreshold { "deess_threshold_db" },
    deessReduction { "deess_reduction_db" };
inline const juce::String delayOn { "delay_on" }, delayTime { "delay_time_ms" }, delayFeedback { "delay_feedback_pct" },
    delayMix { "delay_mix_pct" }, delayHighcut { "delay_highcut_hz" };
inline const juce::String reverbOn { "reverb_on" }, reverbDecay { "reverb_decay_s" }, reverbPredelay { "reverb_predelay_ms" },
    reverbMix { "reverb_mix_pct" }, reverbDamping { "reverb_damping_hz" };
inline const juce::String doublerOn { "doubler_on" }, doublerMix { "doubler_mix_pct" }, doublerWidth { "doubler_width_pct" };

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
    std::atomic<float>*sOn, *sDrive, *sMix;
    std::atomic<float>*dsOn, *dsFreq, *dsThr, *dsRed;
    std::atomic<float>*dlOn, *dlTime, *dlFb, *dlMix, *dlHc;
    std::atomic<float>*rvOn, *rvDecay, *rvPre, *rvMix, *rvDamp;
    std::atomic<float>*dbOn, *dbMix, *dbWidth;
};
} // namespace vm::params
