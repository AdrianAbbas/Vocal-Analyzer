#pragma once
// The Vocal Match effect chain in C++: shared/CHAIN.md, the same algorithms as the Python
// reference chain (analysis/vocalmatch/chain/). No JUCE. Real-time safe: prepare() allocates,
// setParams() and process() never allocate or lock.
//
// Order: input gain -> EQ -> compressor -> saturation -> de-esser -> delay -> reverb
//        -> doubler/width -> output gain

#include "Biquad.h"

#include <array>
#include <cfenv>
#include <cstdint>
#include <functional>
#include <vector>

namespace vm
{
constexpr int kNumEqBands = 6;

// Python's round(): half to even (CHAIN.md: every ms -> samples conversion uses it).
inline long roundHalfEven (double x) noexcept
{
    const double r = std::nearbyint (x); // default rounding mode: to nearest, ties to even
    return (long) r;
}

// ms -> samples as in CHAIN.md: round(ms * fs / 1000), ms first rounded to 0.001 ms (parameters
// are stored as float: 375 ms comes back as 374.99997 and would round to the other sample).
inline long msToSamples (double ms, double fs) noexcept
{
    return roundHalfEven (std::round (ms * 1000.0) / 1000.0 * fs / 1000.0);
}

struct GainParams
{
    bool enabled = false;
    double gainDb = 0.0;
    bool operator== (const GainParams&) const = default;
};

struct EqBandParams
{
    bool enabled = false;
    BiquadType type = BiquadType::peak;
    double freqHz = 1000.0;
    double gainDb = 0.0;
    double q = 0.7071;
    bool operator== (const EqBandParams&) const = default;
};

struct EqParams
{
    bool enabled = false;
    std::array<EqBandParams, kNumEqBands> bands {};
    bool operator== (const EqParams&) const = default;
};

struct CompressorParams
{
    bool enabled = false;
    double thresholdDb = 0.0;
    double ratio = 1.0;
    double attackMs = 10.0;
    double releaseMs = 100.0;
    double kneeDb = 6.0;
    double makeupDb = 0.0;
    bool operator== (const CompressorParams&) const = default;
};

struct SaturationParams
{
    bool enabled = false;
    double driveDb = 0.0;
    double mixPct = 0.0;
    bool operator== (const SaturationParams&) const = default;
};

struct DeEsserParams
{
    bool enabled = false;
    double frequencyHz = 6500.0;
    double thresholdDb = 0.0;
    double reductionDb = 0.0;
    bool operator== (const DeEsserParams&) const = default;
};

struct DelayParams
{
    bool enabled = false;
    double timeMs = 250.0;
    double feedbackPct = 0.0;
    double mixPct = 0.0;
    double highcutHz = 8000.0;
    bool operator== (const DelayParams&) const = default;
};

struct ReverbParams
{
    bool enabled = false;
    double decayS = 1.5;
    double predelayMs = 0.0;
    double mixPct = 0.0;
    double dampingHz = 8000.0;
    bool operator== (const ReverbParams&) const = default;
};

struct DoublerParams
{
    bool enabled = false;
    double mixPct = 0.0;
    double widthPct = 100.0;
    bool operator== (const DoublerParams&) const = default;
};

struct ChainParams
{
    GainParams inputGain;
    EqParams eq;
    CompressorParams compressor;
    SaturationParams saturation;
    DeEsserParams deesser;
    DelayParams delay;
    ReverbParams reverb;
    DoublerParams doubler;
    GainParams outputGain;
    bool operator== (const ChainParams&) const = default;
};

// "Match amount" (shared/CHAIN.md): 0..1 scales every effect's departure from neutral.
inline ChainParams applyMatchAmount (ChainParams p, double amount) noexcept
{
    amount = std::min (std::max (amount, 0.0), 1.0);
    p.inputGain.gainDb *= amount;
    for (auto& b : p.eq.bands)
        b.gainDb *= amount;
    p.compressor.ratio = 1.0 + (std::max (p.compressor.ratio, 1.0) - 1.0) * amount;
    p.compressor.makeupDb *= amount;
    p.saturation.mixPct *= amount;
    p.deesser.reductionDb *= amount;
    p.delay.mixPct *= amount;
    p.reverb.mixPct *= amount;
    p.doubler.mixPct *= amount;
    p.doubler.widthPct = 100.0 + (p.doubler.widthPct - 100.0) * amount;
    p.outputGain.gainDb *= amount;
    return p;
}

// --------------------------------------------------------------------- dynamics helpers
constexpr double kLevelFloorDb = -120.0;
constexpr double kLevelFloorLin = 1e-6; // 10^(-120/20)

inline double levelDb (double linear) noexcept
{
    return 20.0 * std::log10 (std::max (std::abs (linear), kLevelFloorLin));
}

inline double timeConstantCoeff (double timeMs, double fs) noexcept
{
    return std::exp (-1.0 / (std::max (timeMs, 1e-3) * 1e-3 * fs));
}

// Static soft-knee gain computer (Giannoulis et al. 2012): gain *reduction* in dB, >= 0.
inline double gainReductionDb (double x, double thresholdDb, double ratio, double kneeDb) noexcept
{
    ratio = std::max (ratio, 1.0);
    const double slope = 1.0 / ratio - 1.0;
    const double over = x - thresholdDb;
    double y = x;
    if (kneeDb > 0.0)
    {
        const double half = kneeDb / 2.0;
        if (std::abs (over) <= half)
            y = x + slope * (over + half) * (over + half) / (2.0 * kneeDb);
        else if (over > half)
            y = thresholdDb + over / ratio;
    }
    else if (over > 0.0)
    {
        y = thresholdDb + over / ratio;
    }
    return x - y;
}

// Log-domain "smooth branching" ballistics on the gain reduction.
struct BranchingSmoother
{
    double attack = 0.0, release = 0.0, s = 0.0;

    void set (double attackMs, double releaseMs, double fs) noexcept
    {
        attack = timeConstantCoeff (attackMs, fs);
        release = timeConstantCoeff (releaseMs, fs);
    }

    double process (double gr) noexcept
    {
        if (gr > s)
            s = attack * s + (1.0 - attack) * gr;
        else
            s = release * s + (1.0 - release) * gr;
        return s;
    }
};

// ------------------------------------------------------------------- reverb constants
constexpr int kFdnLines = 8;
constexpr std::array<double, kFdnLines> kFdnDelaysMs { 29.7, 37.1, 41.1, 43.7, 53.0, 59.9, 67.7, 73.3 };
constexpr double kMaxPredelayMs = 250.0;
constexpr double kMaxDelayMs = 2000.0;
constexpr double kDeEsserAttackMs = 0.5, kDeEsserReleaseMs = 40.0, kDeEsserQ = 0.7071;

// 8x8 Sylvester Hadamard signs: (-1)^popcount(j & k) (scipy.linalg.hadamard(8)).
constexpr int kHadamardSign[kFdnLines][kFdnLines] = {
    { 1, 1, 1, 1, 1, 1, 1, 1 },     { 1, -1, 1, -1, 1, -1, 1, -1 }, { 1, 1, -1, -1, 1, 1, -1, -1 }, { 1, -1, -1, 1, 1, -1, -1, 1 },
    { 1, 1, 1, 1, -1, -1, -1, -1 }, { 1, -1, 1, -1, -1, 1, -1, 1 }, { 1, 1, -1, -1, -1, -1, 1, 1 }, { 1, -1, -1, 1, -1, 1, 1, -1 },
};

// Doubler voices: (base delay ms, depth ms, LFO rate Hz) for L and R.
struct DoublerVoice
{
    double baseMs, depthMs, rateHz;
};
constexpr std::array<DoublerVoice, 2> kDoublerVoices { DoublerVoice { 14.0, 1.6, 0.31 }, DoublerVoice { 21.0, 2.1, 0.23 } };

// ------------------------------------------------------------------------------ the chain
class Chain
{
public:
    // Allocates every buffer. Call from prepareToPlay, never from the audio thread.
    void prepare (double sampleRate, int maxBlockSize)
    {
        fs = sampleRate;
        blockSize = std::max (maxBlockSize, 16);
        bufL.assign ((size_t) blockSize, 0.0);
        bufR.assign ((size_t) blockSize, 0.0);
        for (auto& d : delayBuf)
            d.assign ((size_t) std::max<long> (1, roundHalfEven (kMaxDelayMs * fs / 1000.0)) + 1, 0.0);
        predelayBuf.assign ((size_t) roundHalfEven (kMaxPredelayMs * fs / 1000.0) + 1, 0.0);
        for (int i = 0; i < kFdnLines; ++i)
        {
            fdnLen[(size_t) i] = std::max<long> (1, roundHalfEven (kFdnDelaysMs[(size_t) i] * fs / 1000.0));
            fdnBuf[(size_t) i].assign ((size_t) fdnLen[(size_t) i], 0.0);
        }
        const double maxVoiceMs = std::max (kDoublerVoices[0].baseMs + kDoublerVoices[0].depthMs,
                                             kDoublerVoices[1].baseMs + kDoublerVoices[1].depthMs);
        doublerLen = (int) std::ceil (maxVoiceMs * fs / 1000.0) + 4;
        for (auto& h : doublerHist)
            h.assign ((size_t) doublerLen, 0.0);
        updateCoefficients();
        reset();
    }

    void reset() noexcept
    {
        for (auto& ch : eqState)
            for (auto& s : ch)
                s.reset();
        compSmoother.s = 0.0;
        for (auto& ch : deessState)
            for (auto& s : ch)
                s.reset();
        deessSmoother.s = 0.0;
        for (auto& d : delayBuf)
            std::fill (d.begin(), d.end(), 0.0);
        delayLp = { 0.0, 0.0 };
        delayPos = 0;
        std::fill (predelayBuf.begin(), predelayBuf.end(), 0.0);
        predelayPos = 0;
        for (auto& b : fdnBuf)
            std::fill (b.begin(), b.end(), 0.0);
        fdnPos.fill (0);
        fdnLp.fill (0.0);
        for (auto& h : doublerHist)
            std::fill (h.begin(), h.end(), 0.0);
        doublerPos = 0;
        sampleIndex = 0;
    }

    // Real-time safe: recomputes coefficients only.
    void setParams (const ChainParams& p) noexcept
    {
        if (p == params)
            return;
        params = p;
        updateCoefficients();
    }

    const ChainParams& getParams() const noexcept { return params; }

    // Current gain reduction in dB (meters): the compressor's and the de-esser's smoothed
    // reduction, 0 when the effect is off.
    double compressorReductionDb() const noexcept { return params.compressor.enabled ? compSmoother.s : 0.0; }
    double deesserReductionDb() const noexcept
    {
        return params.deesser.enabled && params.deesser.reductionDb > 0.0 ? deessSmoother.s : 0.0;
    }
    double getSampleRate() const noexcept { return fs; }

    // Stereo in place. Any length: processed in chunks of the prepared block size.
    template <typename Sample>
    void processStereo (Sample* left, Sample* right, int numSamples) noexcept
    {
        for (int start = 0; start < numSamples; start += blockSize)
        {
            const int n = std::min (blockSize, numSamples - start);
            for (int i = 0; i < n; ++i)
            {
                bufL[(size_t) i] = (double) left[start + i];
                bufR[(size_t) i] = (double) right[start + i];
            }
            processBlock (bufL.data(), bufR.data(), n);
            for (int i = 0; i < n; ++i)
            {
                left[start + i] = (Sample) bufL[(size_t) i];
                right[start + i] = (Sample) bufR[(size_t) i];
            }
        }
    }

    // Mono in place: the input feeds both channels and the output is the mid (L+R)/2.
    template <typename Sample>
    void processMono (Sample* data, int numSamples) noexcept
    {
        for (int start = 0; start < numSamples; start += blockSize)
        {
            const int n = std::min (blockSize, numSamples - start);
            for (int i = 0; i < n; ++i)
                bufL[(size_t) i] = bufR[(size_t) i] = (double) data[start + i];
            processBlock (bufL.data(), bufR.data(), n);
            for (int i = 0; i < n; ++i)
                data[start + i] = (Sample) (0.5 * (bufL[(size_t) i] + bufR[(size_t) i]));
        }
    }

    // Seconds of output after the input stops: the delay's echoes and the reverb's tail
    // (down 60 dB), and the doubler's longest voice.
    static double tailSeconds (const ChainParams& params) noexcept
    {
        double tail = 0.0;
        const auto& d = params.delay;
        if (d.enabled && d.mixPct > 0.0)
        {
            const double fb = d.feedbackPct / 100.0;
            const double repeats = fb > 1e-3 ? std::ceil (60.0 / (-20.0 * std::log10 (fb))) : 1.0;
            tail += d.timeMs / 1000.0 * std::min (repeats + 1.0, 200.0);
        }
        const auto& r = params.reverb;
        if (r.enabled && r.mixPct > 0.0)
            tail += r.predelayMs / 1000.0 + r.decayS + 0.1;
        if (params.doubler.enabled && params.doubler.mixPct > 0.0)
            tail += 0.025;
        return tail;
    }

private:
    void updateCoefficients() noexcept
    {
        numActiveBands = 0;
        for (int b = 0; b < kNumEqBands; ++b)
        {
            const auto& band = params.eq.bands[(size_t) b];
            if (! band.enabled || std::equal_to<double>() (band.gainDb, 0.0)) // skipped, as in Python
                continue;
            activeBand[(size_t) numActiveBands] = b;
            eqCoeffs[(size_t) b] = rbj (band.type, band.freqHz, band.gainDb, band.q, fs);
            ++numActiveBands;
        }
        compSmoother.set (params.compressor.attackMs, params.compressor.releaseMs, fs);
        inputGainLin = dbToLin (params.inputGain.gainDb);
        outputGainLin = dbToLin (params.outputGain.gainDb);

        satDrive = dbToLin (params.saturation.driveDb);

        deessLp = rbj (BiquadType::lowpass, params.deesser.frequencyHz, 0.0, kDeEsserQ, fs);
        deessHp = rbj (BiquadType::highpass, params.deesser.frequencyHz, 0.0, kDeEsserQ, fs);
        deessSmoother.set (kDeEsserAttackMs, kDeEsserReleaseMs, fs);

        delayLen = (int) std::min<long> ((long) delayBuf[0].size() - 1,
                                         std::max<long> (1, msToSamples (params.delay.timeMs, fs)));
        if (delayLen < 1)
            delayLen = 1;
        delayPos %= delayLen;
        delayA = onePoleCoeff (params.delay.highcutHz, fs);

        double gSum = 0.0;
        for (int i = 0; i < kFdnLines; ++i)
        {
            fdnGain[(size_t) i] = std::pow (10.0, -3.0 * (double) fdnLen[(size_t) i] / (fs * std::max (params.reverb.decayS, 1e-3)));
            gSum += fdnGain[(size_t) i];
        }
        const double gMean = gSum / kFdnLines;
        fdnNorm = std::sqrt (std::max (1.0 - gMean * gMean, 0.0));
        fdnA = onePoleCoeff (params.reverb.dampingHz, fs);
        predelayLen = (int) std::min<long> ((long) predelayBuf.size() - 1, std::max<long> (0, msToSamples (params.reverb.predelayMs, fs)));
    }

    void processBlock (double* L, double* R, int n) noexcept
    {
        const auto& p = params;
        if (p.inputGain.enabled)
            applyGain (L, R, n, inputGainLin);
        if (p.eq.enabled)
            processEq (L, R, n);
        if (p.compressor.enabled)
            processCompressor (L, R, n);
        if (p.saturation.enabled && p.saturation.mixPct > 0.0)
            processSaturation (L, R, n);
        if (p.deesser.enabled && p.deesser.reductionDb > 0.0)
            processDeEsser (L, R, n);
        if (p.delay.enabled && p.delay.mixPct > 0.0)
            processDelay (L, R, n);
        if (p.reverb.enabled && p.reverb.mixPct > 0.0)
            processReverb (L, R, n);
        if (p.doubler.enabled)
            processDoubler (L, R, n);
        if (p.outputGain.enabled)
            applyGain (L, R, n, outputGainLin);
        sampleIndex += n;
    }

    static void applyGain (double* L, double* R, int n, double g) noexcept
    {
        for (int i = 0; i < n; ++i)
        {
            L[i] *= g;
            R[i] *= g;
        }
    }

    void processEq (double* L, double* R, int n) noexcept
    {
        for (int k = 0; k < numActiveBands; ++k)
        {
            const int b = activeBand[(size_t) k];
            const auto& c = eqCoeffs[(size_t) b];
            auto& sl = eqState[0][(size_t) b];
            auto& sr = eqState[1][(size_t) b];
            for (int i = 0; i < n; ++i)
            {
                L[i] = sl.process (c, L[i]);
                R[i] = sr.process (c, R[i]);
            }
        }
    }

    void processCompressor (double* L, double* R, int n) noexcept
    {
        const auto& c = params.compressor;
        for (int i = 0; i < n; ++i)
        {
            const double x = levelDb (std::max (std::abs (L[i]), std::abs (R[i])));
            const double s = compSmoother.process (gainReductionDb (x, c.thresholdDb, c.ratio, c.kneeDb));
            const double g = dbToLin (c.makeupDb - s);
            L[i] *= g;
            R[i] *= g;
        }
    }

    // wet = tanh(g x) / g, y = (1 - m) x + m wet
    void processSaturation (double* L, double* R, int n) noexcept
    {
        const double m = params.saturation.mixPct / 100.0, g = satDrive;
        for (int i = 0; i < n; ++i)
        {
            L[i] = (1.0 - m) * L[i] + m * (std::tanh (g * L[i]) / g);
            R[i] = (1.0 - m) * R[i] + m * (std::tanh (g * R[i]) / g);
        }
    }

    // LR4 split; the high band is turned down while it is above the threshold.
    void processDeEsser (double* L, double* R, int n) noexcept
    {
        const auto& d = params.deesser;
        for (int i = 0; i < n; ++i)
        {
            double low[2], high[2];
            const double in[2] = { L[i], R[i] };
            for (int c = 0; c < 2; ++c)
            {
                auto& st = deessState[(size_t) c];
                low[c] = st[1].process (deessLp, st[0].process (deessLp, in[c]));
                high[c] = st[3].process (deessHp, st[2].process (deessHp, in[c]));
            }
            const double level = levelDb (std::max (std::abs (high[0]), std::abs (high[1])));
            const double gr = std::min (std::max (level - d.thresholdDb, 0.0), d.reductionDb);
            const double g = std::pow (10.0, -deessSmoother.process (gr) / 20.0);
            L[i] = low[0] + g * high[0];
            R[i] = low[1] + g * high[1];
        }
    }

    // r = buf[n - D], w = LP(r), buf[n] = x + fb w, y = (1 - m) x + m w
    void processDelay (double* L, double* R, int n) noexcept
    {
        const double m = params.delay.mixPct / 100.0, fb = params.delay.feedbackPct / 100.0, a = delayA;
        double* io[2] = { L, R };
        for (int i = 0; i < n; ++i)
        {
            for (int c = 0; c < 2; ++c)
            {
                auto& buf = delayBuf[(size_t) c];
                const double x = io[c][i];
                const double r = buf[(size_t) delayPos];
                delayLp[(size_t) c] = a * r + (1.0 - a) * delayLp[(size_t) c];
                const double w = delayLp[(size_t) c];
                buf[(size_t) delayPos] = x + fb * w;
                io[c][i] = (1.0 - m) * x + m * w;
            }
            if (++delayPos >= delayLen)
                delayPos = 0;
        }
    }

    // 8-line FDN, Hadamard feedback, input = pre-delayed mid.
    void processReverb (double* L, double* R, int n) noexcept
    {
        static constexpr double s8 = 0.35355339059327373; // 1 / sqrt(8)
        static constexpr int signL[kFdnLines] = { 1, -1, 1, -1, 1, -1, 1, -1 };
        static constexpr int signR[kFdnLines] = { 1, 1, -1, -1, 1, 1, -1, -1 };
        const double m = params.reverb.mixPct / 100.0, a = fdnA;
        const size_t preSize = predelayBuf.size();
        for (int i = 0; i < n; ++i)
        {
            const double mid = 0.5 * (L[i] + R[i]);
            double u;
            if (predelayLen == 0)
                u = mid;
            else
            {
                // predelayBuf holds the last preSize inputs; read the one predelayLen ago.
                const size_t readPos = (predelayPos + preSize - (size_t) predelayLen) % preSize;
                u = predelayBuf[readPos];
            }
            predelayBuf[predelayPos] = mid;
            if (++predelayPos >= preSize)
                predelayPos = 0;

            double f[kFdnLines];
            for (int k = 0; k < kFdnLines; ++k)
            {
                const double r = fdnBuf[(size_t) k][(size_t) fdnPos[(size_t) k]];
                fdnLp[(size_t) k] = a * r + (1.0 - a) * fdnLp[(size_t) k];
                f[k] = fdnGain[(size_t) k] * fdnLp[(size_t) k];
            }
            double wetL = 0.0, wetR = 0.0;
            for (int k = 0; k < kFdnLines; ++k)
            {
                wetL += signL[k] * s8 * f[k];
                wetR += signR[k] * s8 * f[k];
            }
            for (int j = 0; j < kFdnLines; ++j)
            {
                double acc = 0.0; // row j of the Sylvester Hadamard matrix / sqrt(8)
                for (int k = 0; k < kFdnLines; ++k)
                    acc += kHadamardSign[j][k] * s8 * f[k];
                auto& buf = fdnBuf[(size_t) j];
                buf[(size_t) fdnPos[(size_t) j]] = acc + s8 * u;
                if (++fdnPos[(size_t) j] >= fdnLen[(size_t) j])
                    fdnPos[(size_t) j] = 0;
            }
            L[i] = (1.0 - m) * L[i] + m * (wetL * fdnNorm);
            R[i] = (1.0 - m) * R[i] + m * (wetR * fdnNorm);
        }
    }

    // Two modulated-delay voices (linear interpolation) + mid/side width.
    void processDoubler (double* L, double* R, int n) noexcept
    {
        const double m = params.doubler.mixPct / 100.0, w = params.doubler.widthPct / 100.0;
        double* io[2] = { L, R };
        const int len = doublerLen;
        for (int i = 0; i < n; ++i)
        {
            const double t = (double) (sampleIndex + i);
            for (int c = 0; c < 2; ++c)
            {
                auto& hist = doublerHist[(size_t) c];
                const double x = io[c][i];
                hist[(size_t) doublerPos] = x; // hist[pos] = x[t]
                if (m > 0.0)
                {
                    const auto& v = kDoublerVoices[(size_t) c];
                    const double delay = (v.baseMs + v.depthMs * std::sin (2.0 * kPi * v.rateHz * t / fs)) * fs / 1000.0;
                    const double p = t - delay;
                    double voice = 0.0;
                    if (p >= 0.0)
                    {
                        const double k = std::floor (p);
                        const double frac = p - k;
                        const long back0 = (long) (t - k); // samples back from t
                        const double x0 = hist[(size_t) ((doublerPos - back0 % len + len) % len)];
                        const double x1 = hist[(size_t) ((doublerPos - (back0 - 1) % len + len) % len)];
                        voice = x0 + frac * (x1 - x0);
                    }
                    io[c][i] = x + m * voice;
                }
            }
            if (++doublerPos >= len)
                doublerPos = 0;
            if (! std::equal_to<double>() (w, 1.0))
            {
                const double mid = 0.5 * (L[i] + R[i]);
                const double side = 0.5 * (L[i] - R[i]) * w;
                L[i] = mid + side;
                R[i] = mid - side;
            }
        }
    }

    double fs = 48000.0;
    int blockSize = 512;
    ChainParams params;
    std::vector<double> bufL, bufR;
    int64_t sampleIndex = 0;

    double inputGainLin = 1.0, outputGainLin = 1.0;
    std::array<BiquadCoeffs, kNumEqBands> eqCoeffs {};
    std::array<int, kNumEqBands> activeBand {};
    int numActiveBands = 0;
    std::array<std::array<BiquadState, kNumEqBands>, 2> eqState {};
    BranchingSmoother compSmoother;

    double satDrive = 1.0;

    BiquadCoeffs deessLp, deessHp;
    std::array<std::array<BiquadState, 4>, 2> deessState {}; // lp, lp, hp, hp per channel
    BranchingSmoother deessSmoother;

    std::array<std::vector<double>, 2> delayBuf;
    std::array<double, 2> delayLp { 0.0, 0.0 };
    int delayLen = 1, delayPos = 0;
    double delayA = 1.0;

    std::vector<double> predelayBuf;
    size_t predelayPos = 0;
    int predelayLen = 0;
    std::array<std::vector<double>, kFdnLines> fdnBuf;
    std::array<long, kFdnLines> fdnLen {};
    std::array<long, kFdnLines> fdnPos {};
    std::array<double, kFdnLines> fdnLp {};
    std::array<double, kFdnLines> fdnGain {};
    double fdnNorm = 0.0, fdnA = 1.0;

    std::array<std::vector<double>, 2> doublerHist;
    int doublerLen = 1, doublerPos = 0;
};
} // namespace vm
