# NanitosNASA2025

Exoplanet candidate triage with **NASA Kepler** light curves and a lightweight **Generative Adversarial Network**.  
We use **Lightkurve** to fetch/stitch/flatten time series, run **BLS** to phase-align transit signals, and train a **dual-head 1D-CNN**: one head scores **real vs. fake**, the other predicts **FALSE POSITIVE / CANDIDATE / CONFIRMED**.  
A small generator produces synthetic negatives to sharpen the discriminator.

---

##  About this Project

This repository was created for the **2025 NASA Space Apps Challenge** —  
**[“A World Away: Hunting for Exoplanets with AI.”](https://www.spaceappschallenge.org/2025/challenges/a-world-away-hunting-for-exoplanets-with-ai/)**

- **Team:** Nanitos  
- **Local Event:** Ensenada, Mexico  
- **Members:** Saúl Eduardo Pérez Herrera, José Emiliano Villalobos, Ximena León Gómez, Diego Ángel Noriega Ruiz, Roberto Carlos Garcés Labastida, Carolina Valdivia Padilla
- **Report:** Take a look at the report of this project entitled [Nanitos_Report.pdf](https://github.com/Saperz4002/NanitosNASA2025/blob/main/Nanitos_Report.pdf)
---

##  Features

- **MAST/Kepler via Lightkurve:** long-cadence LCs, quarter stitching, flattening, outlier removal, BLS.
- **Dual-head discriminator:** real/fake (sigmoid) + 3-class softmax (FP/Candidate/Confirmed).
- **Adversarial negatives:** generator outputs fake light curves for robust training.
- **Probe script:** load a saved run and predict for a given **KIC/kepid** or Kepler name.

---

## Live GUI

**Try the GUI here ➜ [findexoplanetsnanitos.us/backmod.html](https://findexoplanetsnanitos.us/backmod.html)**
