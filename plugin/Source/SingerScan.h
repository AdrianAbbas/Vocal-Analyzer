#pragma once
// The result of `vocalmatch singers` (analysis/vocalmatch/singers.py): the vocal activity of
// the reference over time, the singers found (A, B, ...) and when each one sings.

#include <juce_core/juce_core.h>

#include <vector>

namespace vm
{
struct SingerScan
{
    struct Singer
    {
        juce::String id;
        double seconds = 0.0, share = 0.0, medianF0 = 0.0;
        juce::Range<double> preview;
    };
    struct Segment
    {
        double start = 0.0, end = 0.0;
        int singer = -1; // index into singers; -2 = singers overlap
    };

    juce::String reference; // the song's full path, as the analyzer saw it
    double durationS = 0.0, frameS = 0.05;
    std::vector<float> activity; // 0..1 per frame
    juce::Array<Singer> singers;
    std::vector<Segment> segments;
    double confidence = 0.0, overlapS = 0.0;
    bool unsure = false, separated = false;
    juce::String message;

    bool isValid() const { return durationS > 0.0; }

    static juce::Result parse (const juce::String& json, SingerScan& out)
    {
        out = {};
        juce::var v;
        if (const auto r = juce::JSON::parse (json, v); r.failed() || ! v.isObject())
            return juce::Result::fail ("not a singer scan");
        if ((int) v["version"] != 1)
            return juce::Result::fail ("unknown singer scan version");
        out.reference = v["reference"].toString();
        out.durationS = (double) v["duration_s"];
        out.frameS = juce::jmax (0.001, (double) v["frame_s"]);
        out.confidence = (double) v["confidence"];
        out.overlapS = (double) v["overlap_s"];
        out.unsure = (bool) v["unsure"];
        out.separated = (bool) v["separated"];
        out.message = v["message"].toString();
        if (const auto* a = v["activity"].getArray())
        {
            out.activity.reserve ((size_t) a->size());
            for (const auto& x : *a)
                out.activity.push_back ((float) (double) x / 100.0f);
        }
        if (const auto* a = v["singers"].getArray())
            for (const auto& s : *a)
            {
                Singer singer;
                singer.id = s["id"].toString();
                singer.seconds = (double) s["seconds"];
                singer.share = (double) s["share"];
                singer.medianF0 = (double) s["median_f0_hz"];
                if (const auto* p = s["preview"].getArray(); p != nullptr && p->size() == 2)
                    singer.preview = { (double) (*p)[0], (double) (*p)[1] };
                out.singers.add (singer);
            }
        if (const auto* a = v["segments"].getArray())
            for (const auto& s : *a)
                if (const auto* seg = s.getArray(); seg != nullptr && seg->size() == 3)
                {
                    Segment g { (double) (*seg)[0], (double) (*seg)[1], -1 };
                    const auto who = (*seg)[2].toString();
                    if (who == "overlap")
                        g.singer = -2;
                    else
                        for (int i = 0; i < out.singers.size(); ++i)
                            if (out.singers[i].id == who)
                                g.singer = i;
                    out.segments.push_back (g);
                }
        if (! out.isValid())
            return juce::Result::fail ("the singer scan has no duration");
        return juce::Result::ok();
    }
};
} // namespace vm
