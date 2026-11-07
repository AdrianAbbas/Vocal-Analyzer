#include "Analyzer.h"

#include <algorithm>

#if JUCE_MAC || JUCE_LINUX || JUCE_BSD
    #include <pwd.h>
    #include <signal.h>
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

juce::File analyzerLogsFolder()
{
#if JUCE_MAC
    auto dir = realHomeDirectory().getChildFile ("Library/Logs/VocalMatch");
#else
    auto dir = realHomeDirectory().getChildFile (".local/share/vocalmatch/logs");
#endif
    if (dir.createDirectory().wasOk() && dir.hasWriteAccess())
        return dir;
    dir = juce::File::getSpecialLocation (juce::File::tempDirectory).getChildFile ("VocalMatch Logs");
    dir.createDirectory();
    return dir;
}

namespace
{
juce::String signalName (int sig)
{
    switch (sig)
    {
        case 4: return "illegal instruction";
        case 5: return "trace trap";
        case 6: return "abort";
        case 7: return "bus error";
        case 8: return "floating-point exception";
        case 9: return "killed";
#if JUCE_MAC
        case 10: return "bus error";
#endif
        case 11: return "segmentation fault";
        case 15: return "terminated";
        default: return "signal " + juce::String (sig);
    }
}

// Starts "$@" from the folder $1 in a clean, bounded environment, then prints
// VOCALMATCH-PID (for Cancel) and VOCALMATCH-EXIT <status> (128 + n for signal n).
const char* launcherScript = R"SH(cd "$1" 2>/dev/null || cd /tmp || exit 70
shift
unset PYTHONHOME PYTHONPATH PYTHONSTARTUP PYTHONEXECUTABLE PYTHONSAFEPATH VIRTUAL_ENV __PYVENV_LAUNCHER__ CONDA_PREFIX CONDA_DEFAULT_ENV CONDA_SHLVL CONDA_EXE CONDA_PYTHON_EXE DYLD_LIBRARY_PATH DYLD_INSERT_LIBRARIES DYLD_FRAMEWORK_PATH LD_PRELOAD KMP_DUPLICATE_LIB_OK
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 PYTHONFAULTHANDLER=1 PYTHONIOENCODING=utf-8 PYTHONUTF8=1
export OMP_NUM_THREADS="$VM_THREADS" MKL_NUM_THREADS="$VM_THREADS" OPENBLAS_NUM_THREADS="$VM_THREADS" VECLIB_MAXIMUM_THREADS="$VM_THREADS" NUMEXPR_NUM_THREADS="$VM_THREADS"
export TQDM_DISABLE=1 HF_HUB_DISABLE_PROGRESS_BARS=1
export PATH="/usr/bin:/bin:/usr/sbin:/sbin:${PATH:-}"
unset VM_THREADS
[ -n "${LANG:-}" ] || export LANG=en_US.UTF-8
"$@" &
pid=$!
echo "VOCALMATCH-PID $pid"
wait "$pid"
status=$?
echo "VOCALMATCH-EXIT $status"
exit "$status"
)SH";
} // namespace

juce::String describeFailure (const juce::String& errorLine, int exitStatus, bool sawExitStatus, bool fatalPythonError,
                              bool outputWritten, const juce::String& stage, const juce::String& lastLines)
{
    const auto during = stage.isNotEmpty() ? " while " + stage : juce::String();
    if (errorLine.isNotEmpty())
        return errorLine.trim();
    if (sawExitStatus && exitStatus == 9 + 128)
        return "The analyzer was stopped by the system" + during
               + ", most likely because the Mac ran out of memory. Close other apps and try again, or mark a shorter part of the song.";
    if (sawExitStatus && exitStatus > 128)
        return "The analyzer crashed (" + signalName (exitStatus - 128) + ")" + during + ". No preset was made.";
    if (fatalPythonError)
        return "The analyzer crashed (a fatal error in Python)" + during + ". No preset was made.";
    if (! sawExitStatus)
        return "The analyzer ended unexpectedly" + during + ". No preset was made.";
    if (exitStatus != 0)
        return "The analyzer stopped with an error (exit " + juce::String (exitStatus) + ")" + during
               + (lastLines.isNotEmpty() ? ": " + lastLines : juce::String (".")) ;
    if (! outputWritten)
        return "The analyzer finished without writing a preset" + during + ".";
    return "The analyzer failed" + during + ".";
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
    if (r.separate != "auto")
        cmd << " --separate " << r.separate;
    if (! r.regions.isEmpty())
        cmd << " --regions " << formatRegions (r.regions);
    if (! r.singers.isEmpty())
        cmd << " --singer " << r.singers.joinIntoString (",") << (r.perSinger ? " --per-singer" : "");
    return cmd;
}

juce::String AnalyzerRun::shellCommandFor (const ScanRequest& r)
{
    auto q = [] (const juce::File& f) { return f.getFullPathName().quoted ('\''); };
    return "vocalmatch singers --reference " + q (r.reference) + " --json " + q (r.outJson)
           + (r.separate != "auto" ? " --separate " + r.separate : juce::String());
}

bool AnalyzerRun::start (const AnalyzeRequest& r)
{
    if (! r.reference.existsAsFile() || ! r.dry.existsAsFile())
    {
        if (state.load() != State::running)
            fail ("Choose a reference song and a dry vocal first.");
        return false;
    }
    kind = "analyze";
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
    kind = "singers";
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
        errorLine = {};
        exitStatus = -1;
        fatalPythonError = false;
        stage = "starting the analyzer";
    }
    childPid = 0;
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
#if JUCE_MAC || JUCE_LINUX || JUCE_BSD
        // The analyzer itself (the wrapper's child): it stops its separation worker on SIGTERM.
        if (const int pid = childPid.load(); pid > 1)
            ::kill ((pid_t) pid, SIGTERM);
#endif
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

juce::File AnalyzerRun::getLogFile() const
{
    std::lock_guard<std::mutex> g (lock);
    return logFile;
}

void AnalyzerRun::writeLog (const juce::String& text)
{
    if (logStream != nullptr)
    {
        logStream->writeText (text, false, false, "\n");
        logStream->flush();
    }
}

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
    writeLog (line + "\n");
    std::lock_guard<std::mutex> g (lock);
    if (line.startsWith ("VOCALMATCH-PID "))
    {
        childPid = line.fromFirstOccurrenceOf (" ", false, false).getIntValue();
        return;
    }
    if (line.startsWith ("VOCALMATCH-EXIT "))
    {
        exitStatus = line.fromFirstOccurrenceOf (" ", false, false).getIntValue();
        return;
    }
    if (line.startsWith ("ERROR "))
        errorLine = line.fromFirstOccurrenceOf ("ERROR ", false, false);
    if (line.startsWith ("Fatal Python error") || line.contains ("Segmentation fault") || line.startsWith ("Current thread 0x"))
        fatalPythonError = true;
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
    // The full log of this run (older logs beyond the last 30 are removed).
    {
        const auto dir = analyzerLogsFolder();
        auto old = dir.findChildFiles (juce::File::findFiles, false, "*.log");
        std::sort (old.begin(), old.end(), [] (const juce::File& a, const juce::File& b)
                   { return a.getLastModificationTime() > b.getLastModificationTime(); });
        for (int i = 30; i < old.size(); ++i)
            old.getReference (i).deleteFile();
        auto file = dir.getNonexistentChildFile (kind + " " + juce::Time::getCurrentTime().formatted ("%Y-%m-%d %H.%M.%S"), ".log", false);
        std::lock_guard<std::mutex> g (lock);
        logFile = file;
        logStream = std::make_unique<juce::FileOutputStream> (file);
        if (! logStream->openedOk())
            logStream.reset();
    }
    juce::StringArray command;
#if JUCE_MAC || JUCE_LINUX || JUCE_BSD
    const int threads = juce::jlimit (1, 4, juce::SystemStats::getNumCpus());
    command.addArray ({ "/usr/bin/env", "VM_THREADS=" + juce::String (threads), "/bin/sh", "-c", launcherScript, "vocalmatch-launcher",
                        analyzerLogsFolder().getFullPathName() });
#endif
    command.addArray (arguments);
#ifdef JucePlugin_VersionString
    const juce::String version = JucePlugin_VersionString;
#else
    const juce::String version = "?";
#endif
    writeLog ("Vocal Match " + version + ", " + juce::SystemStats::getOperatingSystemName() + " ("
              + juce::SystemStats::getCpuModel() + ", " + juce::String (juce::SystemStats::getNumCpus()) + " cores, "
              + juce::String (juce::SystemStats::getMemorySizeInMegabytes()) + " MB)\n");
    writeLog ("host: " + juce::File::getSpecialLocation (juce::File::hostApplicationPath).getFullPathName() + "\n");
    writeLog ("command: " + arguments.joinIntoString (" ") + "\n\n");
    if (! process.start (command, juce::ChildProcess::wantStdOut | juce::ChildProcess::wantStdErr))
    {
        writeLog ("could not start the process\n");
        fail ("This host did not let the plug-in start the analyzer (some hosts run plug-ins in a sandbox). "
              "Run the command below in Terminal, then Load preset.");
        logStream.reset();
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
    {
        writeLog ("\ncancelled\n");
        logStream.reset();
        return;
    }
    process.waitForProcessToFinish (10000);
    bool sawExit;
    int status;
    {
        std::lock_guard<std::mutex> g (lock);
        sawExit = exitStatus >= 0;
        status = exitStatus;
    }
#if ! (JUCE_MAC || JUCE_LINUX || JUCE_BSD)
    sawExit = true;
    status = (int) process.getExitCode();
#endif
    const bool written = output.existsAsFile();
    if (sawExit && status == 0 && written && errorLine.isEmpty())
    {
        writeLog ("\nfinished: " + output.getFullPathName() + "\n");
        logStream.reset();
        progress = 1.0f;
        state = State::finished;
        return;
    }
    const auto tail = juce::StringArray::fromLines (getLog().trimEnd());
    juce::String last;
    for (int i = juce::jmax (0, tail.size() - 2); i < tail.size(); ++i)
        last << tail[i] << " ";
    juce::String message;
    {
        std::lock_guard<std::mutex> g (lock);
        message = describeFailure (errorLine, status, sawExit, fatalPythonError, written, stage, last.trim());
    }
    writeLog ("\nfailed: " + message + "\n");
    logStream.reset();
    output.deleteFile(); // never load a half-written result
    fail (message);
}
} // namespace vm
