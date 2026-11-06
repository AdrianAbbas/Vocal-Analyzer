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
            else if ((bool) e["enabled"])
            {
                out.notInThisVersion.add (key);
            }
        }
    }
    return juce::Result::ok();
}
} // namespace vm
