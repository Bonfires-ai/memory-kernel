# Releasing

1. Run the **release-check** workflow (`.github/workflows/release-check.yml`, started by hand) and wait for it to pass:
   it runs the test suite on the trf default parser, which CI skips.
2. Run the engine tests on a machine with the pinned FCG Go (`pytest -m needs_fcg` with `MK_DATA_DIR` set).
3. Set the date in the CHANGELOG heading, then tag.
