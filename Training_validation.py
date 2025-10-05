import re
import numpy as np
import pandas as pd
import lightkurve as lk
import tensorflow as tf
from tensorflow.keras import layers, Model, Input
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

LATENT_DIM = 100
CURVE_LENGTH = 3197
BATCH_SIZE = 8
EPOCHS = 150
LEAKY_RELU_SLOPE = 0.2
NOISE_STDDEV = 0.2
MIN_REAL_SAMPLES = 10

csv_path = "Kepler_num.csv"
df = pd.read_csv(csv_path)
rename_map = {}
for col in df.columns:
    lc = col.lower()
    if lc == 'kepid': rename_map[col] = 'kepid'
    if lc == 'koi_disposition': rename_map[col] = 'koi_disposition'
df = df.rename(columns=rename_map)
assert {'kepid','koi_disposition'} <= set(df.columns)

n_total = 200
n_confirmed = int(n_total*0.35)
n_candidate = int(n_total*0.30)
n_false_pos = n_total - n_confirmed - n_candidate

sub_conf = df[df['koi_disposition'].str.upper()=='CONFIRMED']
sub_cand = df[df['koi_disposition'].str.upper()=='CANDIDATE']
sub_fp   = df[df['koi_disposition'].str.upper()=='FALSE POSITIVE']
sample_confirmed = sub_conf.sample(n=min(n_confirmed,len(sub_conf)), random_state=42)
sample_candidate = sub_cand.sample(n=min(n_candidate,len(sub_cand)), random_state=42)
sample_falsepos  = sub_fp.sample(n=min(n_false_pos ,len(sub_fp)),   random_state=42)

sample_50 = pd.concat([sample_confirmed,sample_candidate,sample_falsepos], ignore_index=True)
sample_50 = sample_50.sample(frac=1, random_state=42).reset_index(drop=True)
sample_50 = sample_50.dropna(subset=['kepid']).copy()
sample_50 = sample_50.loc[~sample_50['kepid'].duplicated(keep='first')]

targets_labels = []
for _, row in sample_50.iterrows():
    try:
        kid = int(row['kepid'])
    except:
        continue
    disp = str(row['koi_disposition']).upper()
    cls = 2 if disp=='CONFIRMED' else 1 if disp=='CANDIDATE' else 0
    targets_labels.append((kid, cls))

noise_in = Input(shape=(LATENT_DIM,))
g = layers.Dense(800)(noise_in)
g = layers.LeakyReLU(LEAKY_RELU_SLOPE)(g)
g = layers.Dense(1600)(g)
g = layers.BatchNormalization()(g)
g = layers.LeakyReLU(LEAKY_RELU_SLOPE)(g)
g = layers.Dense(CURVE_LENGTH)(g)
g = layers.BatchNormalization()(g)
g = layers.LeakyReLU(LEAKY_RELU_SLOPE)(g)
g = layers.Dense(CURVE_LENGTH, activation='tanh')(g)
g = layers.Reshape((CURVE_LENGTH,1))(g)
generator = Model(noise_in, g, name="Generator")

inp = Input(shape=(CURVE_LENGTH,1))
d = layers.Conv1D(32,7,padding='same')(inp)
d = layers.LeakyReLU(LEAKY_RELU_SLOPE)(d)
d = layers.MaxPooling1D(pool_size=2,strides=1,padding='same')(d)
d = layers.Conv1D(32,7,padding='same')(d)
d = layers.LeakyReLU(LEAKY_RELU_SLOPE)(d)
d = layers.MaxPooling1D(pool_size=2,strides=1,padding='same')(d)
d = layers.Flatten()(d)
d = layers.Dense(128)(d); d = layers.LeakyReLU(LEAKY_RELU_SLOPE)(d)
d = layers.Dense(64)(d);  d = layers.LeakyReLU(LEAKY_RELU_SLOPE)(d)
realness = layers.Dense(1, activation='sigmoid', name='realness_output')(d)
class_out = layers.Dense(3, activation='softmax', name='class_output')(d)
discriminator = Model(inp, [realness, class_out], name="Discriminator")

discriminator.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=4e-5, beta_1=0.5),
    loss={'realness_output':'binary_crossentropy','class_output':'sparse_categorical_crossentropy'}
)

real_curves = []
y_class_list = []
for target_kic, target_cls in targets_labels:
    try:
        tgt = f'KIC {target_kic}'
        result = lk.search_lightcurve(tgt, author='Kepler', cadence='long')
        if len(result)==0: 
            continue
        lc = result[:2].download_all().stitch().flatten(window_length=901).remove_outliers()
        period = np.linspace(1,20,1000)
        bls = lc.to_periodogram(method='bls', period=period, frequency_factor=500)
        planet_period = bls.period_at_max_power
        t0 = bls.transit_time_at_max_power
        duration = bls.duration_at_max_power
        model = bls.get_transit_model(period=planet_period, transit_time=t0, duration=duration)
        folded_model = model.fold(planet_period, t0)
        flux = folded_model.flux.value.astype(np.float32)
        fmin, fmax = np.min(flux), np.max(flux)
        flux = np.zeros_like(flux) if fmax-fmin==0 else (2*(flux-fmin)/(fmax-fmin)-1).astype(np.float32)
        if len(flux) < CURVE_LENGTH: flux = np.pad(flux,(0,CURVE_LENGTH-len(flux)),mode='constant')
        else: flux = flux[:CURVE_LENGTH]
        curve = flux.reshape(CURVE_LENGTH,1)
        real_curves.append(curve)
        y_class_list.append(target_cls)
        if len(real_curves) >= 300:
            break
    except Exception as e:
        print(f"Error {target_kic}: {e}")

if len(real_curves) < MIN_REAL_SAMPLES:
    raise RuntimeError(f"Only {len(real_curves)} curves; need ≥ {MIN_REAL_SAMPLES}.")

real_data   = np.array(real_curves, dtype=np.float32)
real_labels = np.ones((len(real_data),1), dtype=np.float32)
real_classes= np.array(y_class_list, dtype=np.int32)

X_train, X_val, y_real_train, y_real_val, y_class_train, y_class_val = train_test_split(
    real_data, real_labels, real_classes, test_size=0.3, random_state=42, stratify=real_classes
)

fixed_noise = tf.random.normal((1, LATENT_DIM), stddev=NOISE_STDDEV)

for epoch in range(EPOCHS):
    idx = np.random.randint(0, X_train.shape[0], BATCH_SIZE)
    real_batch = X_train[idx]
    real_labels_batch = y_real_train[idx].astype(np.float32)
    real_classes_batch = y_class_train[idx].astype(np.int32)

    noise = tf.random.normal((BATCH_SIZE, LATENT_DIM), stddev=NOISE_STDDEV)
    fake_data = generator(noise).numpy().astype(np.float32)
    fake_labels = np.zeros((BATCH_SIZE,1), dtype=np.float32)
    fake_classes_placeholder = np.zeros((BATCH_SIZE,), dtype=np.int32)

    X_total        = np.concatenate([real_batch, fake_data], axis=0).astype(np.float32)
    y_realness_all = np.concatenate([real_labels_batch, fake_labels], axis=0).astype(np.float32)
    y_class_all    = np.concatenate([real_classes_batch, fake_classes_placeholder], axis=0).astype(np.int32)

    w_realness = np.ones((X_total.shape[0],), dtype=np.float32)
    w_class    = np.concatenate([np.ones((BATCH_SIZE,),dtype=np.float32),
                                 np.zeros((BATCH_SIZE,),dtype=np.float32)], axis=0)

    loss = discriminator.train_on_batch(X_total, [y_realness_all, y_class_all], sample_weight=[w_realness, w_class])
    if (epoch+1) % 10 == 0:
        print(f"Epoch {epoch+1}/{EPOCHS} - Losses: {loss}")

pred_realness, pred_classes_prob = discriminator.predict(X_val)
pred_classes = np.argmax(pred_classes_prob, axis=1)
print("\nValidation — 3-class report (0=FP, 1=Candidate, 2=Confirmed)")
print(classification_report(y_class_val, pred_classes, digits=4))
print("Confusion Matrix:")
print(confusion_matrix(y_class_val, pred_classes))
