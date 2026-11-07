#pragma once
// Preview of the reference song (Analyze panel): plays a span of the song through the
// plug-in's output, mixed in after the chain, so you hear which singer is which or what a
// marked region holds. The track must be playing or monitored for the host to run the
// plug-in (a stopped, unarmed track may not be processed).
//
// Real-time safety: the song is decoded on a background thread into a Song object; the
// message thread hands it to the audio thread through an atomic pointer, and takes the
// replaced one back through another (deleted on the message thread). The audio thread only
// reads samples, with linear interpolation for a different host sample rate.

#include <juce_audio_formats/juce_audio_formats.h>

#include <atomic>

namespace vm
{
struct Song
{
    juce::AudioBuffer<float> audio; // stereo
    double sampleRate = 44100.0;
    double seconds() const { return sampleRate > 0.0 ? audio.getNumSamples() / sampleRate : 0.0; }
};

class PreviewPlayer
{
public:
    static constexpr double kMaxSeconds = 900.0;

    ~PreviewPlayer()
    {
        delete current;
        delete next.exchange (nullptr);
        delete retired.exchange (nullptr);
    }

    // Message thread: decode a file (blocking; call it from a background thread, then hand()).
    static std::unique_ptr<Song> decode (const juce::File& file, juce::String& error)
    {
        juce::AudioFormatManager formats;
        formats.registerBasicFormats();
        std::unique_ptr<juce::AudioFormatReader> reader (formats.createReaderFor (file));
        if (reader == nullptr)
        {
            error = "cannot read " + file.getFileName() + " (WAV, AIFF, FLAC" + juce::String ((juce::SystemStats::getOperatingSystemType() & juce::SystemStats::MacOSX) != 0 ? ", MP3" : "") + ")";
            return nullptr;
        }
        const auto n = (int) juce::jmin<juce::int64> (reader->lengthInSamples, (juce::int64) (kMaxSeconds * reader->sampleRate));
        auto song = std::make_unique<Song>();
        song->sampleRate = reader->sampleRate;
        song->audio.setSize (2, juce::jmax (1, n));
        song->audio.clear();
        reader->read (&song->audio, 0, n, 0, true, true);
        if (reader->numChannels == 1)
            song->audio.copyFrom (1, 0, song->audio, 0, 0, n);
        return song;
    }

    // Message thread: replace the song (stops playback).
    void hand (std::unique_ptr<Song> song)
    {
        collect();
        stop();
        delete next.exchange (song.release()); // a song the audio thread never took
    }

    // Message thread: delete a song the audio thread let go (call regularly, e.g. on a timer).
    void collect() { delete retired.exchange (nullptr); }

    // Message thread: play [start, end) seconds of the song.
    void play (double startSeconds, double endSeconds)
    {
        startS.store (startSeconds);
        endS.store (endSeconds);
        request.fetch_add (1);
    }
    void stop() { stopRequest.fetch_add (1); }
    bool isPlaying() const noexcept { return playing.load(); }
    double positionSeconds() const noexcept { return position.load(); } // song time, while playing
    double playEndSeconds() const noexcept { return endS.load(); }

    // Audio thread: add the preview to the output (after the chain).
    template <typename Sample>
    void process (Sample* left, Sample* right, int n, double hostRate) noexcept
    {
        if (auto* s = next.exchange (nullptr))
        {
            auto* old = current;
            current = s;
            playingNow = false;
            retired.store (old); // the message thread collects it (at most one waits there)
        }
        // a stop first: a stop and a play in the same block mean "play" (hand() stops, then play())
        if (const int r = stopRequest.load(); r != seenStop)
        {
            seenStop = r;
            playingNow = false;
        }
        if (const int r = request.load(); r != seenRequest)
        {
            seenRequest = r;
            if (current != nullptr)
            {
                pos = juce::jmax (0.0, startS.load()) * current->sampleRate;
                end = juce::jmin ((double) current->audio.getNumSamples(), endS.load() * current->sampleRate);
                playingNow = pos < end;
            }
        }
        if (playingNow && current != nullptr && hostRate > 0.0)
        {
            const double step = current->sampleRate / hostRate;
            const float* l = current->audio.getReadPointer (0);
            const float* rr = current->audio.getReadPointer (1);
            const int last = current->audio.getNumSamples() - 1;
            for (int i = 0; i < n && playingNow; ++i)
            {
                const int k = (int) pos;
                const float frac = (float) (pos - k);
                const int k1 = juce::jmin (k + 1, last);
                const float a = l[k] + frac * (l[k1] - l[k]);
                const float b = rr[k] + frac * (rr[k1] - rr[k]);
                if (right != nullptr)
                {
                    left[i] += (Sample) a;
                    right[i] += (Sample) b;
                }
                else
                    left[i] += (Sample) (0.5f * (a + b));
                pos += step;
                if (pos >= end)
                    playingNow = false;
            }
            position.store (pos / current->sampleRate);
        }
        playing.store (playingNow);
    }

private:
    Song* current = nullptr; // audio thread only
    std::atomic<Song*> next { nullptr }, retired { nullptr };
    std::atomic<double> startS { 0.0 }, endS { 0.0 }, position { 0.0 };
    std::atomic<int> request { 0 }, stopRequest { 0 };
    std::atomic<bool> playing { false };
    int seenRequest = 0, seenStop = 0; // audio thread
    double pos = 0.0, end = 0.0;
    bool playingNow = false;
};
} // namespace vm
