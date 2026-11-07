#pragma once
// Capture of the plug-in's input (milestone 6): in a DAW the dry vocal is the track the
// plug-in sits on, so "Capture" records what arrives (before the chain) while the host plays,
// and Analyze uses that take. The buffer is allocated on the message thread when capture is
// armed; the audio thread only writes samples and publishes the count (no allocation, no lock).

#include <juce_audio_formats/juce_audio_formats.h>

#include <atomic>
#include <functional>
#include <vector>

namespace vm
{
class Capture
{
public:
    static constexpr double kMaxSeconds = 300.0;

    // prepareToPlay (no audio running): a new sample rate drops the take and its buffer.
    void prepare (double fs)
    {
        if (std::equal_to<double>() (fs, sampleRate))
            return;
        armed = false;
        buffer.clear();
        buffer.shrink_to_fit();
        count = 0;
        sampleRate = fs;
    }

    // Message thread. Arms or disarms. The first arming allocates kMaxSeconds (mono); the
    // audio thread has never seen this buffer while it was unarmed, so that is safe.
    void setArmed (bool shouldCapture)
    {
        if (shouldCapture && buffer.empty() && sampleRate > 0.0)
            buffer.assign ((size_t) (kMaxSeconds * sampleRate), 0.0f);
        armed = shouldCapture && ! buffer.empty();
    }

    // Message thread: forget the take (keeps the allocation).
    void clear()
    {
        const bool was = armed.exchange (false);
        count = 0;
        armed = was;
    }

    bool isArmed() const noexcept { return armed.load(); }
    double seconds() const noexcept { return sampleRate > 0.0 ? (double) count.load() / sampleRate : 0.0; }
    bool isFull() const noexcept { return ! buffer.empty() && count.load() >= buffer.size(); }

    // Audio thread: append the mono mix of the input. *right* may be null (mono input).
    template <typename Sample>
    void push (const Sample* left, const Sample* right, int n) noexcept
    {
        if (! armed.load (std::memory_order_acquire))
            return;
        size_t c = count.load (std::memory_order_relaxed);
        const size_t room = buffer.size() - std::min (c, buffer.size());
        const size_t m = std::min ((size_t) n, room);
        for (size_t i = 0; i < m; ++i)
            buffer[c + i] = right != nullptr ? (float) (0.5 * ((double) left[i] + (double) right[i])) : (float) left[i];
        count.store (c + m, std::memory_order_release);
    }

    // Message thread: the take so far as a 32-bit float WAV.
    bool writeWav (const juce::File& file) const
    {
        const size_t n = count.load (std::memory_order_acquire);
        if (n == 0 || sampleRate <= 0.0)
            return false;
        file.deleteFile();
        std::unique_ptr<juce::OutputStream> stream (file.createOutputStream());
        if (stream == nullptr)
            return false;
        juce::WavAudioFormat wav;
        std::unique_ptr<juce::AudioFormatWriter> writer (wav.createWriterFor (stream, juce::AudioFormatWriterOptions()
                                                                                           .withSampleRate (sampleRate)
                                                                                           .withNumChannels (1)
                                                                                           .withBitsPerSample (32)
                                                                                           .withSampleFormat (juce::AudioFormatWriterOptions::SampleFormat::floatingPoint)));
        if (writer == nullptr)
            return false;
        const float* channels[] = { buffer.data() };
        return writer->writeFromFloatArrays (channels, 1, (int) n);
    }

private:
    std::vector<float> buffer;
    double sampleRate = 0.0;
    std::atomic<size_t> count { 0 };
    std::atomic<bool> armed { false };
};
} // namespace vm
