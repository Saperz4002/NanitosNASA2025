import pandas as pd
import numpy as np
import lightkurve as lk
import tensorflow as tf
from tensorflow.keras import layers, Model, Input

def class_from_disposition(d: str) -> int:
    return 1 if str(d).upper() == 'CONFIRMED' else 0

def class_from_disposition(d: str) -> int:
    d = str(d).upper()
    if d == 'CONFIRMED':
        return 2
    if d == 'CANDIDATE':
        return 1
    return 0

def normalize_flux(flux):
    flux_min, flux_max = np.min(flux), np.max(flux)
    if flux_max - flux_min == 0:
        return np.zeros_like(flux)
    return 2 * (flux - flux_min) / (flux_max - flux_min) - 1

def get_real_transit_curve(name):
    try:
        result = lk.search_lightcurve(name, author='Kepler', cadence='long')
        if len(result) == 0:
            return None
        lc = result[:2].download_all().stitch().flatten(window_length=901).remove_outliers()
        period = np.linspace(1, 20, 1000)
        bls = lc.to_periodogram(method='bls', period=period, frequency_factor=500)
        planet_period = bls.period_at_max_power
        t0 = bls.transit_time_at_max_power
        duration = bls.duration_at_max_power
        model = bls.get_transit_model(period=planet_period, transit_time=t0, duration=duration)
        folded_model = model.fold(planet_period, t0)
        flux = normalize_flux(folded_model.flux.value)
        if len(flux) != CURVE_LENGTH:
            flux = np.pad(flux, (0, CURVE_LENGTH - len(flux)), mode='constant') if len(flux) < CURVE_LENGTH else flux[:CURVE_LENGTH]
        return flux.reshape(CURVE_LENGTH, 1), folded_model, planet_period, t0
    except Exception as e:
        print(f"⚠️ Error with {name}: {e}")
        return None

def build_generator():
    noise = Input(shape=(LATENT_DIM,))
    x = layers.Dense(800)(noise)
    x = layers.LeakyReLU(LEAKY_RELU_SLOPE)(x)
    x = layers.Dense(1600)(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(LEAKY_RELU_SLOPE)(x)
    x = layers.Dense(CURVE_LENGTH)(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(LEAKY_RELU_SLOPE)(x)
    x = layers.Dense(CURVE_LENGTH, activation='tanh')(x)
    x = layers.Reshape((CURVE_LENGTH, 1))(x)
    return Model(noise, x, name="Generator")

def build_discriminator():
    input_curve = Input(shape=(CURVE_LENGTH, 1))
    x = layers.Conv1D(32, 7, padding='same')(input_curve)
    x = layers.LeakyReLU(LEAKY_RELU_SLOPE)(x)
    x = layers.MaxPooling1D(pool_size=2, strides=1, padding='same')(x)
    x = layers.Conv1D(32, 7, padding='same')(x)
    x = layers.LeakyReLU(LEAKY_RELU_SLOPE)(x)
    x = layers.MaxPooling1D(pool_size=2, strides=1, padding='same')(x)
    x = layers.Flatten()(x)
    x = layers.Dense(128)(x); x = layers.LeakyReLU(LEAKY_RELU_SLOPE)(x)
    x = layers.Dense(64)(x);  x = layers.LeakyReLU(LEAKY_RELU_SLOPE)(x)
    realness = layers.Dense(1, activation='sigmoid', name='realness_output')(x)
    class_output = layers.Dense(3, activation='softmax', name='class_output')(x)
    return Model(input_curve, [realness, class_output], name="Discriminator")

def predict_kepid(kepid_or_name, realness_threshold=0.5):
    x = get_real_transit_curve_for_infer(kepid_or_name, CURVE_LENGTH)
    if x is None:
        return
    x = np.expand_dims(x, axis=0)
    if isinstance(discriminator, tf.keras.layers.TFSMLayer):
        predictions = discriminator(x)
        pred_realness = predictions.get('realness_output')
        pred_class_probs = predictions.get('class_output')
        if pred_realness is None or pred_class_probs is None:
            print("Error: Could not find 'realness_output' or 'class_output' in TFSMLayer outputs.")
            print("Available outputs:", predictions.keys())
            return
        pred_realness = pred_realness.numpy()
        pred_class_probs = pred_class_probs.numpy()
    elif isinstance(discriminator, tf.keras.Model):
        pred_realness, pred_class_probs = discriminator.predict(x, verbose=0)
    else:
        print("Error: Discriminator model loaded in an unexpected format.")
        return
