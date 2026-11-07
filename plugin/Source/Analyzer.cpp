#include "Analyzer.h"

#if JUCE_MAC || JUCE_LINUX || JUCE_BSD
    #include <pwd.h>
    #include <unistd.h>
#endif

namespace vm
{
juce::File realHomeDirectory()
{
#if JUCE_MAC || JUCE_LINUX || JUCE_BSD
    if (const auto* pw = getpwuid (getuid()); pw != nullptr && pw->pw_dir != nullptr)
        return juce::File (juce::String::fromUTF8 (pw->pw_dir));
#endif
    return juce::File::getSpecialLocation (juce::File::userHomeDirectory);
}

juce::Array<juce::File> analyzerCandidates()
{
    juce::Array<juce::File> out;
    const auto env = juce::SystemStats::getEnvironmentVariable ("VOCALMATCH_ANALYZER", {});
    if (env.isNotEmpty())
        out.add (juce::File (env));
    const auto home = realHomeDirectory();
    out.add (home.getChildFile (".local/bin/vocalmatch"));
    out.add (home.getChildFile ("Library/Application Support/VocalMatch/venv/bin/vocalmatch"));
    out.add (home.getChildFile (".local/share/vocalmatch/venv/bin/vocalmatch"));
    out.add (juce::File ("/opt/homebrew/bin/vocalmatch"));
    out.add (juce::File ("/usr/local/bin/vocalmatch"));
    for (const auto& dir : juce::StringArray::fromTokens (juce::SystemStats::getEnvironmentVariable ("PATH", {}), ":", {}))
        if (dir.isNotEmpty())
            out.add (juce::File (dir).getChildFile ("vocalmatch"));
    return out;
}

juce::File findAnalyzer()
{
    for (const auto& f : analyzerCandidates())
        if (f.existsAsFile())
            return f;
    return {};
}

juce::File vocalMatchFolder (const juce::String& sub)
{
    auto dir = realHomeDirectory().getChildFile ("Music/Vocal Match").getChildFile (sub);
    if (dir.createDirectory().wasOk() && dir.hasWriteAccess())
        return dir;
    dir = juce::File::getSpecialLocation (juce::File::tempDirectory).getChildFile ("Vocal Match").getChildFile (sub);
    dir.createDirectory();
    return dir;
}

AnalyzerRun::AnalyzerRun() : juce::Thread ("Vocal Match analyzer") {}

AnalyzerRun::~AnalyzerRun()
{
    cancel();
    stopThread (5000);
}

juce::String formatRegions (const juce::Array<juce::Range<double>>& regions)
{
    juce::StringArray parts;
    for (const auto& r : regions)
        parts.add (juce::String (r.getStart(), 2) + "-" + juce::String (r.getEnd(), 2));
    return parts.joinIntoString (",");
}

namespace
{
void addSelection (juce::StringArray& a, const AnalyzeRequest& r)
{
    if (! r.regions.isEmpty())
        a.addArray ({ "--regions", formatRegions (r.regions) });
    if (! r.singers.isEmpty())
        a.addArray ({ "--singer", r.singers.joinIntoString (",") });
    if (r.perSinger && ! r.singers.isEmpty())
        a.add ("--per-singer");
}
} // namespace

juce::StringArray AnalyzerRun::argumentsFor (const juce::File& exe, const AnalyzeRequest& r)
{
    juce::StringArray a;
    a.add (exe.getFullPathName());
    a.addArray ({ "analyze", "--reference", r.reference.getFullPathName(), "--dry", r.dry.getFullPathName(),
                  "--out", r.outPreset.getFullPathName(), "--refine", juce::String (r.refineEvals),
                  "--passes", juce::String (r.passes), "--separate", r.separate, "--progress" });
    addSelection (a, r);
    return a;
}

juce::StringArray AnalyzerRun::argumentsFor (const juce::File& exe, const ScanRequest& r)
{
    return { exe.getFullPathName(), "singers", "--reference", r.reference.getFullPathName(), "--json",
             r.outJson.getFullPathName(), "--separate", r.separate, "--progress" };
}

juce::String AnalyzerRun::shellCommandFor (const AnalyzeRequest& r)
{
    juce::String cmd = "vocalmatch analyze";
    auto q = [] (const juce::File& f) { return f.getFullPathName().quoted ('\''); };
    cmd << " --reference " << q (r.reference) << " --dry " << q (r.dry) << " --out " << q (r.outPreset);
    if (r.refineEvals != 60)
        cmd << " --refine " << r.refineEvals;
    if (! r.regions.isEmpty())
        cmd << " --regions " << formatRegions (r.regions);
    if (! r.singers.isEmpty())
        cmd << " --singer " << r.singers.joinIntoString (",") << (r.perSinger ? " --per-singer" : "");
    return cmd;
}

juce::String AnalyzerRun::shellCommandFor (const ScanRequest& r)
{
    auto q = [] (const juce::File& f) { return f.getFullPathName().quoted ('\''); };
    return "vocalmatch singers --reference " + q (r.reference) + " --json " + q (r.outJson);
}

bool AnalyzerRun::start (const AnalyzeRequest& r)
{
    if (! r.reference.existsAsFile() || ! r.dry.existsAsFile())
    {
        if (state.load() != State::running)
            fail ("Choose a reference song and a dry vocal first.");
        return false;
    }
    return startWith (argumentsFor (findAnalyzer(), r), r.outPreset, { r.reference, r.dry });
}

bool AnalyzerRun::start (const ScanRequest& r)
{
    if (! r.reference.existsAsFile())
    {
        if (state.load() != State::running)
            fail ("Choose a reference song first.");
        return false;
    }
    return startWith (argumentsFor (findAnalyzer(), r), r.outJson, { r.reference });
}

bool AnalyzerRun::startWith (const juce::StringArray& args, const juce::File& out, const juce::Array<juce::File>&)
{
    if (state.load() == State::running)
        return false;
    stopThread (2000);
    {
        std::lock_guard<std::mutex> g (lock);
        error = {};
        log = {};
        stage = "starting the analyzer";
    }
    arguments = args;
    output = out;
    progress = 0.0f;
    analyzer = findAnalyzer();
    if (! analyzer.existsAsFile())
    {
        fail ("The Vocal Match analyzer is not installed (run install-mac.sh from the download).");
        return false;
    }
    arguments.set (0, analyzer.getFullPathName());
    out.getParentDirectory().createDirectory();
    out.deleteFile();
    state = State::running;
    startThread();
    return true;
}

void AnalyzerRun::cancel()
{
    if (state.load() == State::running)
    {
        state = State::cancelled;
        signalThreadShouldExit();
        process.kill();
    }
}

bool AnalyzerRun::waitUntilDone (int timeoutMs) { return waitForThreadToExit (timeoutMs); }

juce::String AnalyzerRun::getStage() const
{
    std::lock_guard<std::mutex> g (lock);
    return stage;
}

juce::String AnalyzerRun::getError() const
{
    std::lock_guard<std::mutex> g (lock);
    return error;
}

juce::String AnalyzerRun::getLog() const
{
    std::lock_guard<std::mutex> g (lock);
    return log;
}

juce::File AnalyzerRun::getPresetFile() const { return output; }

void AnalyzerRun::fail (const juce::String& message)
{
    {
        std::lock_guard<std::mutex> g (lock);
        error = message;
    }
    state = State::failed;
}

void AnalyzerRun::handleLine (const juce::String& line)
{
    std::lock_guard<std::mutex> g (lock);
    if (line.startsWith ("PROGRESS "))
    {
        const auto rest = line.fromFirstOccurrenceOf ("PROGRESS ", false, false);
        progress = juce::jlimit (0.0f, 1.0f, rest.upToFirstOccurrenceOf (" ", false, false).getFloatValue());
        stage = rest.fromFirstOccurrenceOf (" ", false, false);
        return;
    }
    log << line << "\n";
    if (log.length() > 20000)
        log = log.getLastCharacters (16000);
}

void AnalyzerRun::run()
{
    if (! process.start (arguments, juce::ChildProcess::wantStdOut | juce::ChildProcess::wantStdErr))
    {
        fail ("This host did not let the plug-in start the analyzer (some hosts run plug-ins in a sandbox). "
              "Run the command below in Terminal, then Load preset.");
        return;
    }
    std::vector<char> lineBytes; // bytes of the current line (UTF-8 may split across reads)
    char buffer[1];
    while (! threadShouldExit())
    {
        // One byte at a time: readProcessOutput is fread, which waits until it has as many
        // bytes as asked for, and progress lines must arrive as they are written.
        const int n = process.readProcessOutput (buffer, 1);
        if (n <= 0)
        {
            if (! process.isRunning())
                break;
            wait (20);
            continue;
        }
        if (buffer[0] == '\n')
        {
            handleLine (juce::String::fromUTF8 (lineBytes.data(), (int) lineBytes.size()).trimEnd());
            lineBytes.clear();
        }
        else
            lineBytes.push_back (buffer[0]);
    }
    if (! lineBytes.empty())
        handleLine (juce::String::fromUTF8 (lineBytes.data(), (int) lineBytes.size()).trimEnd());
    if (state.load() == State::cancelled)
        return;
    process.waitForProcessToFinish (10000);
    const auto code = process.getExitCode();
    if (code == 0 && output.existsAsFile())
    {
        progress = 1.0f;
        state = State::finished;
    }
    else
    {
        const auto tail = juce::StringArray::fromLines (getLog().trimEnd());
        juce::String last;
        for (int i = juce::jmax (0, tail.size() - 3); i < tail.size(); ++i)
            last << tail[i] << " ";
        fail ("The analyzer stopped (exit " + juce::String ((int) code) + "): " + last.trim());
    }
}
} // namespace vm
