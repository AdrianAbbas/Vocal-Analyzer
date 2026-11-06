#pragma once
// The Vocal Match effect chain in C++: shared/CHAIN.md, the same algorithms as the Python
// reference chain (analysis/vocalmatch/chain/). No JUCE. Real-time safe: prepare() allocates,
// setParams() and process() never allocate or lock.
//
// Order: input gain -> EQ -> compressor -> output gain
// (saturation, de-esser, delay, reverb and doubler/width join as they are estimated).

#include "Biquad.h"

#include <array>
#include <functional>
#include <vector>

namespace vm
{
constexpr int kNumEqBands = 6;

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

struct ChainParams
{
    GainParams inputGain;
    EqParams eq;
    CompressorParams compressor;
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

// ------------------------------------------------------------------------------ the chain
class Chain
{
public:
    // Allocates the scratch buffers. Call from prepareToPlay, never from the audio thread.
    void prepare (double sampleRate, int maxBlockSize)
    {
        fs = sampleRate;
        blockSize = std::max (maxBlockSize, 16);
        bufL.assign ((size_t) blockSize, 0.0);
        bufR.assign ((size_t) blockSize, 0.0);
        updateCoefficients();
        reset();
    }

    void reset() noexcept
    {
        for (auto& ch : eqState)
            for (auto& s : ch)
                s.reset();
        compSmoother.s = 0.0;
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

    // Seconds of output after the input stops (none yet: no delay or reverb).
    double tailSeconds() const noexcept { return 0.0; }

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
    }

    void processBlock (double* L, double* R, int n) noexcept
    {
        if (params.inputGain.enabled)
            applyGain (L, R, n, inputGainLin);
        if (params.eq.enabled)
            processEq (L, R, n);
        if (params.compressor.enabled)
            processCompressor (L, R, n);
        if (params.outputGain.enabled)
            applyGain (L, R, n, outputGainLin);
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

    double fs = 48000.0;
    int blockSize = 512;
    ChainParams params;
    std::vector<double> bufL, bufR;

    double inputGainLin = 1.0, outputGainLin = 1.0;
    std::array<BiquadCoeffs, kNumEqBands> eqCoeffs {};
    std::array<int, kNumEqBands> activeBand {};
    int numActiveBands = 0;
    std::array<std::array<BiquadState, kNumEqBands>, 2> eqState {};
    BranchingSmoother compSmoother;
};
} // namespace vm
