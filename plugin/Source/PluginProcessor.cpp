#include "PluginProcessor.h"

#include "PluginEditor.h"

VocalMatchProcessor::VocalMatchProcessor()
    : AudioProcessor (BusesProperties()
                          .withInput ("Input", juce::AudioChannelSet::stereo(), true)
                          .withOutput ("Output", juce::AudioChannelSet::stereo(), true)),
      apvts (*this, nullptr, "VocalMatch", vm::params::createLayout()),
      reader (apvts)
{
    chain.prepare (44100.0, 512);
}

bool VocalMatchProcessor::isBusesLayoutSupported (const BusesLayout& layouts) const
{
    // mono -> mono, mono -> stereo and stereo -> stereo. The chain is stereo inside:
    // a mono input feeds both channels; a mono output is the mid (L+R)/2.
    const auto in = layouts.getMainInputChannelSet();
    const auto out = layouts.getMainOutputChannelSet();
    const bool inOk = in == juce::AudioChannelSet::mono() || in == juce::AudioChannelSet::stereo();
    const bool outOk = out == juce::AudioChannelSet::mono() || out == juce::AudioChannelSet::stereo();
    return inOk && outOk && in.size() <= out.size();
}

void VocalMatchProcessor::prepareToPlay (double sampleRate, int samplesPerBlock)
{
    chain.prepare (sampleRate, samplesPerBlock);
    chain.setParams (reader.read());
    chain.reset();
    setLatencySamples (0); // nothing in the chain looks ahead
}

template <typename Sample>
void VocalMatchProcessor::process (juce::AudioBuffer<Sample>& buffer)
{
    juce::ScopedNoDenormals noDenormals;
    const int numIn = getTotalNumInputChannels();
    const int numOut = getTotalNumOutputChannels();
    const int n = buffer.getNumSamples();
    for (int ch = numIn; ch < numOut; ++ch)
        buffer.clear (ch, 0, n);
    if (numIn == 0 || n == 0)
        return;

    chain.setParams (reader.read()); // recomputes coefficients only when something changed

    if (numOut >= 2)
    {
        if (numIn == 1)
            buffer.copyFrom (1, 0, buffer, 0, 0, n);
        chain.processStereo (buffer.getWritePointer (0), buffer.getWritePointer (1), n);
    }
    else
    {
        chain.processMono (buffer.getWritePointer (0), n);
    }
}

void VocalMatchProcessor::processBlock (juce::AudioBuffer<float>& buffer, juce::MidiBuffer&) { process (buffer); }
void VocalMatchProcessor::processBlock (juce::AudioBuffer<double>& buffer, juce::MidiBuffer&) { process (buffer); }

juce::AudioProcessorEditor* VocalMatchProcessor::createEditor() { return new VocalMatchEditor (*this); }

// ---------------------------------------------------------------------------- presets
void VocalMatchProcessor::setParameterValue (const juce::String& paramId, float realValue)
{
    if (auto* p = apvts.getParameter (paramId))
    {
        const float normalised = p->convertTo0to1 (realValue);
        p->beginChangeGesture();
        p->setValueNotifyingHost (normalised);
        p->endChangeGesture();
    }
}

juce::Result VocalMatchProcessor::loadPresetFile (const juce::File& file)
{
    if (! file.existsAsFile())
        return juce::Result::fail ("file not found: " + file.getFullPathName());
    return loadPresetText (file.loadFileAsString(), file.getFileName());
}

juce::Result VocalMatchProcessor::loadPresetText (const juce::String& json, const juce::String& displayName)
{
    vm::PresetValues values;
    const auto result = vm::parsePreset (json, values);
    if (result.failed())
    {
        presetStatus = displayName + ": " + result.getErrorMessage();
        ++presetGeneration;
        return result;
    }
    for (const auto& [id, value] : values.values)
        setParameterValue (id, value);
    rememberPreset (values, json, displayName);
    return juce::Result::ok();
}

void VocalMatchProcessor::rememberPreset (const vm::PresetValues& values, const juce::String& json, const juce::String& name)
{
    estimates = values;
    presetJson = json;
    presetName = name;
    presetStatus = describe (values);
    ++presetGeneration;
}

juce::String VocalMatchProcessor::describe (const vm::PresetValues& v) const
{
    juce::String s = presetName;
    if (v.referenceName.isNotEmpty())
        s << " (reference: " << v.referenceName << ")";
    if (! v.notInThisVersion.isEmpty())
        s << " - not in this version yet: " << v.notInThisVersion.joinIntoString (", ");
    return s;
}

std::optional<float> VocalMatchProcessor::getEstimate (const juce::String& paramId) const
{
    for (const auto& [id, value] : estimates.values)
        if (id == paramId)
            return value;
    return std::nullopt;
}

std::optional<float> VocalMatchProcessor::getConfidence (const juce::String& effectKey) const
{
    if (auto it = estimates.confidence.find (effectKey); it != estimates.confidence.end())
        return it->second;
    return std::nullopt;
}

// ------------------------------------------------------------------------------ state
void VocalMatchProcessor::getStateInformation (juce::MemoryBlock& destData)
{
    auto state = apvts.copyState();
    state.setProperty ("presetName", presetName, nullptr);
    state.setProperty ("presetJson", presetJson, nullptr);
    if (auto xml = state.createXml())
        copyXmlToBinary (*xml, destData);
}

void VocalMatchProcessor::setStateInformation (const void* data, int sizeInBytes)
{
    auto xml = getXmlFromBinary (data, sizeInBytes);
    if (xml == nullptr || ! xml->hasTagName (apvts.state.getType()))
        return;
    auto state = juce::ValueTree::fromXml (*xml);
    const auto name = state.getProperty ("presetName").toString();
    const auto json = state.getProperty ("presetJson").toString();
    state.removeProperty ("presetName", nullptr);
    state.removeProperty ("presetJson", nullptr);
    apvts.replaceState (state);
    // The estimates come back for the editor; the parameter values are the saved ones.
    vm::PresetValues values;
    if (json.isNotEmpty() && vm::parsePreset (json, values).wasOk())
        rememberPreset (values, json, name);
}

juce::AudioProcessor* JUCE_CALLTYPE createPluginFilter() { return new VocalMatchProcessor(); }
