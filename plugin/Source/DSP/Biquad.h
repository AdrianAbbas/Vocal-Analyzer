#pragma once
// RBJ "Audio EQ Cookbook" biquads, exactly as analysis/vocalmatch/chain/biquad.py and
// shared/CHAIN.md define them: Q-form alpha, shelves with 2*sqrt(A)*alpha, normalised by a0,
// run as transposed direct form II (what scipy.signal.sosfilt runs). Double precision.
// No JUCE, no allocation.

#include <algorithm>
#include <cmath>

namespace vm
{
constexpr double kPi = 3.14159265358979323846;
constexpr double kMaxFreqFraction = 0.49;

inline double dbToLin (double db) noexcept { return std::pow (10.0, db / 20.0); }

inline double clampFreq (double freqHz, double fs) noexcept
{
    return std::min (std::max (freqHz, 1.0), kMaxFreqFraction * fs);
}

enum class BiquadType
{
    lowShelf = 0, // the order of the plug-in's EQ band "type" choice
    peak = 1,
    highShelf = 2,
    highpass = 3,
    lowpass = 4
};

struct BiquadCoeffs
{
    double b0 = 1.0, b1 = 0.0, b2 = 0.0, a1 = 0.0, a2 = 0.0;
};

inline BiquadCoeffs rbj (BiquadType type, double freqHz, double gainDb, double q, double fs) noexcept
{
    const double f0 = clampFreq (freqHz, fs);
    q = std::max (q, 1e-3);
    const double A = std::pow (10.0, gainDb / 40.0);
    const double w0 = 2.0 * kPi * f0 / fs;
    const double cw = std::cos (w0), sw = std::sin (w0);
    const double alpha = sw / (2.0 * q);
    double b0, b1, b2, a0, a1, a2;
    switch (type)
    {
        case BiquadType::peak:
            b0 = 1.0 + alpha * A;
            b1 = -2.0 * cw;
            b2 = 1.0 - alpha * A;
            a0 = 1.0 + alpha / A;
            a1 = -2.0 * cw;
            a2 = 1.0 - alpha / A;
            break;
        case BiquadType::lowShelf:
        {
            const double sq = 2.0 * std::sqrt (A) * alpha;
            b0 = A * ((A + 1) - (A - 1) * cw + sq);
            b1 = 2.0 * A * ((A - 1) - (A + 1) * cw);
            b2 = A * ((A + 1) - (A - 1) * cw - sq);
            a0 = (A + 1) + (A - 1) * cw + sq;
            a1 = -2.0 * ((A - 1) + (A + 1) * cw);
            a2 = (A + 1) + (A - 1) * cw - sq;
            break;
        }
        case BiquadType::highShelf:
        {
            const double sq = 2.0 * std::sqrt (A) * alpha;
            b0 = A * ((A + 1) + (A - 1) * cw + sq);
            b1 = -2.0 * A * ((A - 1) + (A + 1) * cw);
            b2 = A * ((A + 1) + (A - 1) * cw - sq);
            a0 = (A + 1) - (A - 1) * cw + sq;
            a1 = 2.0 * ((A - 1) - (A + 1) * cw);
            a2 = (A + 1) - (A - 1) * cw - sq;
            break;
        }
        case BiquadType::highpass:
            b0 = (1.0 + cw) / 2.0;
            b1 = -(1.0 + cw);
            b2 = (1.0 + cw) / 2.0;
            a0 = 1.0 + alpha;
            a1 = -2.0 * cw;
            a2 = 1.0 - alpha;
            break;
        case BiquadType::lowpass:
        default:
            b0 = (1.0 - cw) / 2.0;
            b1 = 1.0 - cw;
            b2 = (1.0 - cw) / 2.0;
            a0 = 1.0 + alpha;
            a1 = -2.0 * cw;
            a2 = 1.0 - alpha;
            break;
    }
    return { b0 / a0, b1 / a0, b2 / a0, a1 / a0, a2 / a0 };
}

// Transposed direct form II, one channel.
struct BiquadState
{
    double z1 = 0.0, z2 = 0.0;

    void reset() noexcept { z1 = z2 = 0.0; }

    double process (const BiquadCoeffs& c, double x) noexcept
    {
        const double y = c.b0 * x + z1;
        z1 = c.b1 * x - c.a1 * y + z2;
        z2 = c.b2 * x - c.a2 * y;
        return y;
    }
};

// One-pole low-pass  y += a (x - y),  a = 1 - exp(-2 pi fc / fs)  (delay and reverb damping).
inline double onePoleCoeff (double cutoffHz, double fs) noexcept
{
    return 1.0 - std::exp (-2.0 * kPi * clampFreq (cutoffHz, fs) / fs);
}
} // namespace vm
