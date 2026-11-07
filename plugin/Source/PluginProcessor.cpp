#include "PluginProcessor.h"

#include "PluginEditor.h"

VocalMatchProcessor::VocalMatchProcessor()
    : AudioProcessor (BusesProperties()
                          .withInput ("Input", juce::AudioChannelSet::stereo(), true)
                          .withOutput ("Output", juce::AudioChannelSet::stereo(), true)),
      apvts (*this, nullptr, "VocalMatch", vm::params::createLayout()),
      reader (apvts)
{
    abParam = apvts.getRawParameterValue (vm::params::abDry);
    chain.prepare (44100.0, 512);
    capture.prepare (44100.0);
}

VocalMatchProcessor::~VocalMatchProcessor() { songPool.removeAllJobs (true, 20000); }

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
    chain.prepare (sampleRate, juce::jmin (samplesPerBlock, kChunk));
    chain.setParams (reader.read());
    chain.reset();
    capture.prepare (sampleRate);
    currentRate = sampleRate;
    abGain = abParam->load() >= 0.5f ? 1.0 : 0.0; // no fade at start
    abStep = 1.0 / (0.02 * sampleRate);           // 20 ms A/B crossfade
    setLatencySamples (0);                        // nothing in the chain looks ahead
}

bool VocalMatchProcessor::hostIsPlaying()
{
    if (auto* ph = getPlayHead())
        if (const auto pos = ph->getPosition())
            return pos->getIsPlaying();
    return true; // no transport information: capture what arrives
}

template <typename Sample>
void VocalMatchProcessor::processChunk (Sample* left, Sample* right, int n, bool monoOut)
{
    const double target = abParam->load (std::memory_order_relaxed) >= 0.5f ? 1.0 : 0.0;
    const bool mixDry = target > 0.0 || abGain > 0.0;
    if (mixDry)
        for (int i = 0; i < n; ++i)
        {
            dryChunk[0][(size_t) i] = (double) left[i];
            dryChunk[1][(size_t) i] = right != nullptr ? (double) right[i] : (double) left[i];
        }
    // The chain always runs, so tails continue while you listen to the dry signal.
    if (monoOut)
        chain.processMono (left, n);
    else
        chain.processStereo (left, right, n);
    if (! mixDry)
        return;
    for (int i = 0; i < n; ++i)
    {
        abGain = target > abGain ? juce::jmin (target, abGain + abStep) : juce::jmax (target, abGain - abStep);
        left[i] = (Sample) ((1.0 - abGain) * (double) left[i] + abGain * dryChunk[0][(size_t) i]);
        if (right != nullptr)
            right[i] = (Sample) ((1.0 - abGain) * (double) right[i] + abGain * dryChunk[1][(size_t) i]);
    }
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

    if (capture.isArmed() && hostIsPlaying())
        capture.push (buffer.getReadPointer (0), numIn >= 2 ? buffer.getReadPointer (1) : (const Sample*) nullptr, n);
    inputPeak = juce::jmax (inputPeak.load(), (float) buffer.getMagnitude (0, n));

    if (numOut >= 2 && numIn == 1)
        buffer.copyFrom (1, 0, buffer, 0, 0, n);
    for (int start = 0; start < n; start += kChunk)
    {
        const int m = juce::jmin (kChunk, n - start);
        if (numOut >= 2)
            processChunk (buffer.getWritePointer (0, start), buffer.getWritePointer (1, start), m, false);
        else
            processChunk (buffer.getWritePointer (0, start), (Sample*) nullptr, m, true);
    }
    // The reference song's preview (Analyze panel), mixed in after the chain.
    preview.process (buffer.getWritePointer (0), numOut >= 2 ? buffer.getWritePointer (1) : (Sample*) nullptr, n, currentRate);
    // The editor takes the maximum since its last look and resets it (meters).
    outputPeak = juce::jmax (outputPeak.load(), (float) buffer.getMagnitude (0, n));
    compressorReductionDb = (float) chain.compressorReductionDb();
    deesserReductionDb = (float) chain.deesserReductionDb();
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
    if (v.selectionInfo.isNotEmpty())
        s << " - " << v.selectionInfo;
    if (v.pitchInfo.isNotEmpty())
        s << " - reference vocal: " << v.pitchInfo << " (tune your take to match; not applied)";
    if (! v.notInThisVersion.isEmpty())
        s << " - not in this version yet: " << v.notInThisVersion.joinIntoString (", ");
    return s;
}

bool VocalMatchProcessor::resetToEstimates()
{
    if (estimates.values.empty())
        return false;
    for (const auto& [id, value] : estimates.values)
        setParameterValue (id, value);
    return true;
}

// ------------------------------------------------------------------- reference song
void VocalMatchProcessor::loadReferenceSong()
{
    songError = {};
    if (! referenceFile.existsAsFile())
    {
        peaks = {};
        preview.hand (nullptr);
        ++songGeneration;
        return;
    }
    songLoading = true;
    const auto file = referenceFile;
    songPool.addJob ([this, file]
                     {
                         auto d = std::make_unique<Decoded>();
                         d->file = file;
                         d->song = vm::PreviewPlayer::decode (file, d->error);
                         if (d->song != nullptr)
                         {
                             const auto& a = d->song->audio;
                             const int n = a.getNumSamples(), buckets = 2000;
                             d->peaks.seconds = d->song->seconds();
                             d->peaks.lo.assign (buckets, 0.0f);
                             d->peaks.hi.assign (buckets, 0.0f);
                             for (int b = 0; b < buckets; ++b)
                             {
                                 const int s0 = (int) ((juce::int64) n * b / buckets), s1 = juce::jmax (s0 + 1, (int) ((juce::int64) n * (b + 1) / buckets));
                                 float lo = 0.0f, hi = 0.0f;
                                 for (int i = s0; i < juce::jmin (s1, n); ++i)
                                 {
                                     const float v = 0.5f * (a.getSample (0, i) + a.getSample (1, i));
                                     lo = juce::jmin (lo, v);
                                     hi = juce::jmax (hi, v);
                                 }
                                 d->peaks.lo[(size_t) b] = lo;
                                 d->peaks.hi[(size_t) b] = hi;
                             }
                         }
                         std::lock_guard<std::mutex> g (decodedLock);
                         decoded = std::move (d);
                     });
}

void VocalMatchProcessor::pollSong()
{
    preview.collect();
    std::unique_ptr<Decoded> d;
    {
        std::lock_guard<std::mutex> g (decodedLock);
        d = std::move (decoded);
    }
    if (d == nullptr)
        return;
    if (d->file != referenceFile) // the user chose another song meanwhile
        return;
    songLoading = false;
    songError = d->error;
    peaks = std::move (d->peaks);
    preview.hand (std::move (d->song));
    ++songGeneration;
}

juce::File VocalMatchProcessor::scanFileFor (const juce::File& song) const
{
    return vm::vocalMatchFolder ("Scans").getChildFile (song.getFileName() + ".singers.json");
}

bool VocalMatchProcessor::loadScanFile (const juce::File& file)
{
    vm::SingerScan s;
    const auto json = file.loadFileAsString();
    if (vm::SingerScan::parse (json, s).failed())
        return false;
    if (juce::File (s.reference) != referenceFile && juce::File (s.reference).getFileName() != referenceFile.getFileName())
        return false;
    scan = s;
    scanJson = json;
    juce::StringArray kept;
    for (const auto& id : chosenSingers)
        for (const auto& singer : scan.singers)
            if (singer.id == id)
                kept.add (id);
    chosenSingers = kept;
    ++scanGeneration;
    return true;
}

void VocalMatchProcessor::clearScan()
{
    scan = {};
    scanJson = {};
    chosenSingers.clear();
    ++scanGeneration;
}

juce::Result VocalMatchProcessor::makeAnalyzeRequest (vm::AnalyzeRequest& out)
{
    if (! referenceFile.existsAsFile())
        return juce::Result::fail ("Choose the reference song first.");
    out.reference = referenceFile;
    out.regions = regions;
    for (const auto& id : chosenSingers)
        for (const auto& singer : scan.singers)
            if (singer.id == id)
                out.singers.add (id);
    out.perSinger = perSingerPresets && ! out.singers.isEmpty();
    if (useCapturedDry)
    {
        if (capture.seconds() < 5.0)
            return juce::Result::fail ("Capture at least 5 s of your vocal first: arm Capture and play the track.");
        const auto file = vm::vocalMatchFolder ("Captures").getChildFile (
            "capture " + juce::Time::getCurrentTime().formatted ("%Y-%m-%d %H.%M.%S") + ".wav");
        if (! capture.writeWav (file))
            return juce::Result::fail ("Could not write the captured take to " + file.getFullPathName());
        dryFile = file;
    }
    else if (! dryFile.existsAsFile())
        return juce::Result::fail ("Choose your dry vocal file first (or use Capture).");
    out.dry = dryFile;
    out.outPreset = vm::vocalMatchFolder ("Presets").getChildFile (
        referenceFile.getFileNameWithoutExtension() + " " + juce::Time::getCurrentTime().formatted ("%Y-%m-%d %H.%M") + ".json");
    return juce::Result::ok();
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
    state.setProperty ("referenceFile", referenceFile.getFullPathName(), nullptr);
    state.setProperty ("dryFile", dryFile.getFullPathName(), nullptr);
    state.setProperty ("useCapturedDry", useCapturedDry, nullptr);
    state.setProperty ("regions", vm::formatRegions (regions), nullptr);
    state.setProperty ("chosenSingers", chosenSingers.joinIntoString (","), nullptr);
    state.setProperty ("perSinger", perSingerPresets, nullptr);
    state.setProperty ("singerScan", scanJson, nullptr);
    state.setProperty ("editorWidth", editorWidth, nullptr);
    state.setProperty ("editorHeight", editorHeight, nullptr);
    state.setProperty ("editorPage", editorPage, nullptr);
    state.setProperty ("collapsed", collapsedCards.joinIntoString (","), nullptr);
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
    for (const auto* key : { "referenceFile", "dryFile", "useCapturedDry" })
        if (state.hasProperty (key))
        {
            const auto v = state.getProperty (key);
            if (juce::String (key) == "useCapturedDry")
                useCapturedDry = (bool) v;
            else
                (juce::String (key) == "referenceFile" ? referenceFile : dryFile) =
                    juce::File::isAbsolutePath (v.toString()) ? juce::File (v.toString()) : juce::File();
            state.removeProperty (key, nullptr);
        }
    auto take = [&state] (const char* key) -> juce::var
    {
        const auto v = state.getProperty (key);
        state.removeProperty (key, nullptr);
        return v;
    };
    regions.clear();
    for (const auto& r : juce::StringArray::fromTokens (take ("regions").toString(), ",", {}))
        if (r.contains ("-"))
        {
            const double a = r.upToFirstOccurrenceOf ("-", false, false).getDoubleValue();
            const double b = r.fromFirstOccurrenceOf ("-", false, false).getDoubleValue();
            if (b > a)
                regions.add ({ a, b });
        }
    chosenSingers = juce::StringArray::fromTokens (take ("chosenSingers").toString(), ",", {});
    chosenSingers.removeEmptyStrings();
    perSingerPresets = (bool) take ("perSinger");
    const auto scanText = take ("singerScan").toString();
    scan = {};
    scanJson = {};
    if (scanText.isNotEmpty() && vm::SingerScan::parse (scanText, scan).wasOk())
        scanJson = scanText;
    ++scanGeneration;
    if (const auto w = take ("editorWidth"); ! w.isVoid())
        editorWidth = (int) w;
    if (const auto h = take ("editorHeight"); ! h.isVoid())
        editorHeight = (int) h;
    if (const auto pg = take ("editorPage"); ! pg.isVoid())
        editorPage = (int) pg;
    collapsedCards = juce::StringArray::fromTokens (take ("collapsed").toString(), ",", {});
    collapsedCards.removeEmptyStrings();
    apvts.replaceState (state);
    // The estimates come back for the editor; the parameter values are the saved ones.
    vm::PresetValues values;
    if (json.isNotEmpty() && vm::parsePreset (json, values).wasOk())
        rememberPreset (values, json, name);
}

juce::AudioProcessor* JUCE_CALLTYPE createPluginFilter() { return new VocalMatchProcessor(); }
