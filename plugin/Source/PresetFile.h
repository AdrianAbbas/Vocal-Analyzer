#pragma once
// Reads the analyzer's preset.json (shared/preset.schema.json) into parameter values.

#include <juce_core/juce_core.h>

#include <map>
#include <vector>

namespace vm
{
struct PresetValues
{
    juce::String referenceName;               // reference.filename
    std::vector<std::pair<juce::String, float>> values; // parameter ID -> value in real units (bools 0/1, choices index)
    std::map<juce::String, float> confidence; // effect key ("eq", "compressor", ...) -> 0..1
    juce::StringArray notInThisVersion;       // enabled effects in the file this plug-in does not have yet
};

// Parses preset JSON text. Fails on invalid JSON or an unsupported schema_version; unknown
// effects and keys are ignored (enabled ones are listed in notInThisVersion).
juce::Result parsePreset (const juce::String& jsonText, PresetValues& out);
} // namespace vm
