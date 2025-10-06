# NanitosNASA2025

Exoplanet candidate triage with **NASA Kepler** light curves and a lightweight **adversarial classifier**.  
We use **Lightkurve** to fetch/stitch/flatten time series, run **BLS** to phase-align transit signals, and train a **dual-head 1D-CNN**: one head scores **real vs. fake**, the other predicts **FALSE POSITIVE / CANDIDATE / CONFIRMED**.  
A small generator produces synthetic negatives to sharpen the discriminator.

---

## ✨ Features
- 🔭 **MAST/Kepler via Lightkurve**: long-cadence LCs, quarter stitching, flattening, outlier removal, BLS.
- 🧠 **Dual-head discriminator**: real/fake (sigmoid) + 3-class softmax (FP/Candidate/Confirmed).
- 🧪 **Adversarial negatives**: generator outputs fake light curves for robust training.
- 🧰 **Probe script**: load a saved run and predict for a given **KIC/kepid** or Kepler name.

---

## 📦 Installation

```bash
# Python 3.10+ recommended
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

pip install -U pip
pip install lightkurve tensorflow scikit-learn numpy pandas matplotlib tqdm
