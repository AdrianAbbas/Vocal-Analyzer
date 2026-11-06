# Test audio

- `private/` — your own reference songs and vocals for listening tests.
  Git-ignored: copyrighted material and personal recordings never go in the repo.
- The automated tests need no audio files: they generate a deterministic
  synthetic vocal (`vocalmatch.synth.synthetic_vocal`). To listen to it:
  `python -m vocalmatch synth --out tests/audio/private/synthetic.wav`.
