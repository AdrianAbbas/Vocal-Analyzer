#pragma once
// In-plugin Analyze (milestone 6, SPEC option A): run the installed Python analyzer
// (`vocalmatch analyze ... --progress`) as a separate process on a background thread, follow
// its progress, and hand back the preset it wrote. Never touches the audio thread.

#include <juce_core/juce_core.h>

#include <mutex>

namespace vm
{
// The user's real home folder (a sandboxed host reports its container instead).
juce::File realHomeDirectory();

// Where install.sh / install-mac.sh put the `vocalmatch` command, in search order:
// $VOCALMATCH_ANALYZER, ~/.local/bin, ~/Library/Application Support/VocalMatch/venv/bin,
// ~/.local/share/vocalmatch/venv/bin, Homebrew, /usr/local/bin, then $PATH.
juce::Array<juce::File> analyzerCandidates();
juce::File findAnalyzer();

// Where analyses and captured takes are written: ~/Music/Vocal Match/<sub>, or the temporary
// folder when that cannot be created (sandboxed hosts).
juce::File vocalMatchFolder (const juce::String& sub);

struct AnalyzeRequest
{
    juce::File reference;  // the song (or a vocal stem)
    juce::File dry;        // the user's dry vocal (a file, or the captured take)
    juce::File outPreset;  // where the analyzer writes preset.json
    int refineEvals = 60;
    int passes = 2;
    juce::String separate = "auto";
    juce::Array<juce::Range<double>> regions; // seconds in the song; empty = all of it
    juce::StringArray singers;                // "A", "B", ...; empty = every singer together
    bool perSinger = false;                   // one preset per chosen singer
};

// `vocalmatch singers`: who sings when in the reference (for the Analyze panel).
struct ScanRequest
{
    juce::File reference;
    juce::File outJson;
    juce::String separate = "auto";
};

juce::String formatRegions (const juce::Array<juce::Range<double>>&); // "12.50-30.00,65.00-80.50"

class AnalyzerRun final : private juce::Thread
{
public:
    enum class State
    {
        idle,
        running,
        finished,
        failed,
        cancelled
    };

    AnalyzerRun();
    ~AnalyzerRun() override;

    // Starts the analyzer; false (and getError()) if it is already running or not installed.
    bool start (const AnalyzeRequest&);
    bool start (const ScanRequest&);
    void cancel();
    // Blocks until the run ends (VocalMatchRender --analyze); false on timeout.
    bool waitUntilDone (int timeoutMs);

    State getState() const noexcept { return state.load(); }
    float getProgress() const noexcept { return progress.load(); }
    juce::String getStage() const;
    juce::String getError() const;
    juce::String getLog() const;
    juce::File getPresetFile() const; // the file the run writes (preset, or the scan's JSON)

    static juce::StringArray argumentsFor (const juce::File& analyzer, const AnalyzeRequest&);
    static juce::StringArray argumentsFor (const juce::File& analyzer, const ScanRequest&);
    // The same command for Terminal, quoted (shown and copied when the host cannot start it).
    static juce::String shellCommandFor (const AnalyzeRequest&);
    static juce::String shellCommandFor (const ScanRequest&);

private:
    bool startWith (const juce::StringArray& args, const juce::File& output, const juce::Array<juce::File>& inputs);
    void run() override;
    void fail (const juce::String&);
    void handleLine (const juce::String&);

    juce::StringArray arguments;
    juce::File output;
    juce::File analyzer;
    std::atomic<State> state { State::idle };
    std::atomic<float> progress { 0.0f };
    mutable std::mutex lock;
    juce::String stage, error, log;
    juce::ChildProcess process;
};
} // namespace vm
