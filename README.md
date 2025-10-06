**NanitosNASA2025
**
Exoplanet candidate triage with NASA Kepler light curves and a lightweight adversarial classifier. We use Lightkurve to fetch/stitch/flatten Kepler time series, run BLS to phase-fold potential transits, and train a dual-head 1D-CNN: one head scores real vs. fake, the other predicts FALSE POSITIVE / CANDIDATE / CONFIRMED. A small generator provides synthetic negatives to sharpen the discriminator.
