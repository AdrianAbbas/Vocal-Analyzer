#include "PresetFile.h"

#include "Parameters.h"

namespace vm
{
namespace
{
constexpr int kSchemaVersion = 1;

struct Writer
{
    PresetValues& out;
    const juce::var& effect;

    bool has (const char* key) const { return effect.hasProperty (key); }

    void number (const char* key, const juce::String& paramId) const
    {
        if (has (key))
            out.values.emplace_back (paramId, (float) (double) effect[key]);
    }

    void flag (const char* key, const juce::String& paramId) const
    {
        if (has (key))
            out.values.emplace_back (paramId, (bool) effect[key] ? 1.0f : 0.0f);
    }
};
} // namespace

juce::Result parsePreset (const juce::String& jsonText, PresetValues& out)
{
    out = {};
    juce::var root;
    const auto parsed = juce::JSON::parse (jsonText, root);
    if (parsed.failed())
        return juce::Result::fail ("not valid JSON: " + parsed.getErrorMessage());
    if (! root.isObject())
        return juce::Result::fail ("not a Vocal Match preset (expected a JSON object)");
    const auto version = root["schema_version"];
    if (! (version.isInt() || version.isInt64() || version.isDouble()) || (int) version != kSchemaVersion)
        return juce::Result::fail ("unsupported schema_version " + version.toString() + " (this plug-in reads version "
                                   + juce::String (kSchemaVersion) + ")");
    const auto effects = root["effects"];
    if (! effects.isObject())
        return juce::Result::fail ("the preset has no \"effects\" object");

    out.referenceName = root["reference"]["filename"].toString();
    if (const auto sel = root["reference"]["selection"]; sel.isObject())
    {
        // "Singer A, 22.7 s analysed (overlap excluded 3.6 s, singer confidence 99 %)"
        juce::String info;
        if (const auto* singers = sel["singers"].getArray(); singers != nullptr && ! singers->isEmpty())
        {
            juce::StringArray ids;
            for (const auto& v : *singers)
                ids.add (v.toString());
            info << (ids.size() > 1 ? "Singers " : "Singer ") << ids.joinIntoString (" + ") << ", ";
        }
        else if (const auto* regions = sel["regions"].getArray(); regions != nullptr && ! regions->isEmpty())
            info << regions->size() << (regions->size() > 1 ? " regions, " : " region, ");
        info << juce::String ((double) sel["analysed_s"], 1) << " s analysed";
        juce::StringArray extra;
        if (sel.hasProperty ("overlap_excluded_s"))
            extra.add ("overlap excluded " + juce::String ((double) sel["overlap_excluded_s"], 1) + " s");
        if (sel.hasProperty ("singer_confidence"))
            extra.add ("singer confidence " + juce::String (juce::roundToInt ((double) sel["singer_confidence"] * 100.0)) + " %");
        if (! extra.isEmpty())
            info << " (" << extra.joinIntoString (", ") << ")";
        out.selectionInfo = info;
    }
    if (const auto pc = root["pitch_correction"]; pc.isObject())
    {
        // Milestone 7: shown, not applied (v1 has no pitch correction).
        juce::String info;
        if ((bool) pc["detected"])
        {
            info << "pitch-corrected " << juce::roundToInt ((double) pc["amount_pct"]) << " %";
            if (pc["retune"].isString())
                info << ", " << pc["retune"].toString() << " retune";
        }
        else
            info << "no pitch correction found";
        if (pc["key"].isString())
            info << ", key " << pc["key"].toString();
        out.pitchInfo = info;
    }

    if (auto* obj = effects.getDynamicObject())
    {
        for (const auto& prop : obj->getProperties())
        {
            const auto key = prop.name.toString();
            const auto& e = prop.value;
            if (! e.isObject())
                continue;
            if (e.hasProperty ("confidence"))
                out.confidence[key] = (float) (double) e["confidence"];
            const Writer w { out, e };

            if (key == "input_gain" || key == "output_gain")
            {
                const bool in = key == "input_gain";
                w.flag ("enabled", in ? params::inputOn : params::outputOn);
                w.number ("gain_db", in ? params::inputGain : params::outputGain);
            }
            else if (key == "eq")
            {
                w.flag ("enabled", params::eqOn);
                const auto* bands = e["bands"].getArray();
                for (int b = 0; b < kNumEqBands; ++b)
                {
                    if (bands == nullptr || b >= bands->size())
                    {
                        out.values.emplace_back (params::eqBandOn (b), 0.0f); // fewer bands in the file
                        continue;
                    }
                    const auto& band = bands->getReference (b);
                    const Writer bw { out, band };
                    bw.flag ("enabled", params::eqBandOn (b));
                    const int type = params::eqTypeKeys.indexOf (band["type"].toString());
                    if (type >= 0)
                        out.values.emplace_back (params::eqBandType (b), (float) type);
                    bw.number ("freq_hz", params::eqBandFreq (b));
                    bw.number ("gain_db", params::eqBandGain (b));
                    bw.number ("q", params::eqBandQ (b));
                }
            }
            else if (key == "compressor")
            {
                w.flag ("enabled", params::compOn);
                w.number ("threshold_db", params::compThreshold);
                w.number ("ratio", params::compRatio);
                w.number ("attack_ms", params::compAttack);
                w.number ("release_ms", params::compRelease);
                w.number ("knee_db", params::compKnee);
                w.number ("makeup_db", params::compMakeup);
            }
            else if (key == "saturation")
            {
                w.flag ("enabled", params::satOn);
                w.number ("drive_db", params::satDrive);
                w.number ("mix_pct", params::satMix);
            }
            else if (key == "deesser")
            {
                w.flag ("enabled", params::deessOn);
                w.number ("frequency_hz", params::deessFreq);
                w.number ("threshold_db", params::deessThreshold);
                w.number ("reduction_db", params::deessReduction);
            }
            else if (key == "delay")
            {
                w.flag ("enabled", params::delayOn);
                w.number ("time_ms", params::delayTime);
                w.number ("feedback_pct", params::delayFeedback);
                w.number ("mix_pct", params::delayMix);
                w.number ("highcut_hz", params::delayHighcut); // sync_note is informational
            }
            else if (key == "reverb")
            {
                w.flag ("enabled", params::reverbOn);
                w.number ("decay_s", params::reverbDecay);
                w.number ("predelay_ms", params::reverbPredelay);
                w.number ("mix_pct", params::reverbMix);
                w.number ("damping_hz", params::reverbDamping);
            }
            else if (key == "doubler")
            {
                w.flag ("enabled", params::doublerOn);
                w.number ("mix_pct", params::doublerMix);
                w.number ("width_pct", params::doublerWidth);
            }
            else if ((bool) e["enabled"])
            {
                out.notInThisVersion.add (key);
            }
        }
    }
    return juce::Result::ok();
}
} // namespace vm
