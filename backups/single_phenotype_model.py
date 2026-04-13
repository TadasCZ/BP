from pandas_plink import read_plink
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import math

import tensorflow as tf
from tensorflow.keras import layers, models, regularizers

from scipy import stats
from sklearn.impute import SimpleImputer
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from phenotype_levels import phenotype_levels

fam_path = "/storage/plzen4-ntis/home/tadsova/rice_data/base_filtered_v0.7_0.8_10kb_1_0.8_50_1.fam"
pheno_path = "phenotypes/all_phenotypes.txt"
prefix = "/storage/plzen4-ntis/home/tadsova/rice_data/base_filtered_v0.7_0.8_10kb_1_0.8_50_1"

"""
lsen_levels = [1, 3, 5, 7, 9]

1 - VERY EARLY (leaves dried before grain mature)
3 - EARLY (all leaves have lost green colour)
5 - INTERMEDIATE (one leaf still green)
7 - LATE (two or more leaves still green)
9 - VERY LATE (all leaves still green)

phenotype = "LSEN"
"""

def snp_matrix(prefix):
    (bim, fam, G) = read_plink(prefix, verbose=False)

    X = G.compute().T.astype(np.float32) # sample x snp
    missing_count = np.isnan(X).sum()
    print(f"Celkem chybějících genotypů: {missing_count}")

    # nahrazení NaN za modus -> řešení?
    if missing_count > 0:
        modes = stats.mode(X, axis=0, nan_policy='omit', keepdims=True).mode[0]
        modes = np.nan_to_num(modes, nan=0)
        
        inds = np.where(np.isnan(X))
        X[inds] = np.take(modes, inds[1])
    
    X = X.astype(np.int8)
    print("Genotype matrix shape:", X.shape)

    return X, bim, fam

def prepare_phenotype_data(pheno_path, fam_path, phenotype):
    pheno = pd.read_csv(pheno_path, sep="\t")
    fam = pd.read_csv(
        fam_path, 
        sep=r"\s+", 
        header=None, 
        names=["FID", "IID", "father", "mother", "sex", "trait"]
    )
    
    pheno_subset = pheno[["3K_DNA_IRIS_UNIQUE_ID", phenotype]].rename(
        columns={"3K_DNA_IRIS_UNIQUE_ID": "IID", phenotype: "VALUE"}
    )
    
    fam["IDX"] = range(len(fam))

    merged_data = fam[["IID", "IDX"]].merge(pheno_subset, on="IID").dropna(subset=["VALUE"])

    merged_data = merged_data[["IDX", "IID", "VALUE"]]
    merged_data["VALUE"] = merged_data["VALUE"].astype(int)

    print(f"Počet vzorků '{phenotype}' po filtraci: {len(merged_data)}")
    
    return merged_data

def analyze_phenotype_distribution(df, phenotype_name):
    print(f"\nDistribuce hodnot pro: {phenotype_name}")

    counts = df["VALUE"].value_counts().sort_index()
    
    print(counts)
    print(f"Celkem: {counts.sum()} vzorků")
    
    return counts

def onehot_encode_phenotype(df, levels, prefix):
    onehot_data = pd.get_dummies(df["VALUE"], prefix=prefix, dtype=int)

    for lvl in levels:
        col = f"{prefix}_{lvl}"
        if col not in onehot_data.columns:
            onehot_data[col] = 0

    column_order = [f"{prefix}_{l}" for l in levels]
    onehot_data = onehot_data[column_order]

    final_df = pd.concat([
        df[["IDX", "IID"]].reset_index(drop=True), 
        onehot_data.reset_index(drop=True)
    ], axis=1)

    print(f'\nOne-hot kódování provedeno.')

    return final_df

def get_clean_labels(df_onehot, levels, prefix):
    label_cols = []
    for l in levels:
        label_cols.append(prefix + "_" + str(l))
    y = df_onehot[label_cols].values
    
    print(f"Tvar matice fenotypů: {y.shape}")
    print(y)
    return y

def align_genotypes_to_phenotypes(X, df_pheno):
    keep_indices = df_pheno["IDX"].values

    print(f"\nPůvodní počet vzorků v X: {X.shape[0]}")
    print(f"Počet vzorků po filtraci: {len(keep_indices)}")
    print(f"Prvních 10 vybraných indexů řádků: {keep_indices[:10]}")
    
    X_filtered = X[keep_indices, :]
    
    print(f"Nová velikost matice X: {X_filtered.shape}")
    
    return X_filtered

def onehot_encode_genotypes(X):
    num_samples, num_snps = X.shape
    X_onehot = np.zeros((num_samples, num_snps, 3), dtype=np.int8)
    
    for val in [0, 1, 2]:
        X_onehot[:, :, val] = (X == val)

    return X_onehot.reshape(num_samples, -1)

X, bim, fam = snp_matrix(prefix)
print(X)

# připravím si jen hodnoty jednoho fenotypu
df_lsen = prepare_phenotype_data(pheno_path, fam_path, phenotype)

# analýza phenotypu
dist_summary = analyze_phenotype_distribution(df_lsen, phenotype)

# převedení fenotypových dat na onehot
df_onehot = onehot_encode_phenotype(df_lsen, lsen_levels, phenotype)
#print(df_onehot.head())

# připravené čisté fenotypy
y_phenotypes = get_clean_labels(df_onehot, lsen_levels, phenotype)

# seřazené vzorky dle fenotypů
X_ready_for_onehot = align_genotypes_to_phenotypes(X, df_lsen)

# genotypy na onehot
X_onehot = onehot_encode_genotypes(X_ready_for_onehot)

def create_train_test_split(X, y, test_size=0.2, random_state=42):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=test_size, 
        random_state=random_state,
        shuffle=True   # zkusit změnit na false -> aby se zachovalo pořadí?
    )
    
    print("Dokončené rozdělení dat")
    print(f"X_train: {X_train.shape}, y_train: {y_train.shape}")
    print(f"X_test:  {X_test.shape}, y_test:  {y_test.shape}")
    
    return X_train, X_test, y_train, y_test

# train/test data
X_train, X_test, y_train, y_test = create_train_test_split(X_onehot, y_phenotypes)

def build_genomic_model(input_dim, num_classes):
    model = models.Sequential([
        layers.Input(shape=(input_dim,)),

        #layers.Dense(128, activation='relu'),
        layers.Dense(1024),
        layers.BatchNormalization(),
        layers.Activation("relu"),
        #layers.Dropout(0.5),
        
        layers.Dense(1024),
        layers.BatchNormalization(),
        layers.Activation("relu"),
        #layers.Dropout(0.3),
        
        layers.Dense(num_classes, activation='softmax')
    ])

    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-9)
    
    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

input_size = X_train.shape[1]
classes_count = y_train.shape[1]

model = build_genomic_model(input_size, classes_count)
model.summary()

history = model.fit(
    X_train, y_train,
    epochs=50,
    batch_size=64,
    validation_split=0.1,
    verbose=1
)

def plot_history(history):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(acc) + 1)

    plt.figure(figsize=(12, 5))

    # Accuracy
    plt.subplot(1, 2, 1)
    plt.plot(epochs, acc, 'bo-', label='Trénovací acc')
    plt.plot(epochs, val_acc, 'ro-', label='Validační acc')
    plt.title('Přesnost trénování a validace')
    plt.xlabel('Epochs')
    plt.ylabel('Acc')
    plt.legend()

    # Loss
    plt.subplot(1, 2, 2)
    plt.plot(epochs, loss, 'bo-', label='Trénovací loss')
    plt.plot(epochs, val_loss, 'ro-', label='Validační loss')
    plt.title('Ztráta trénování a validace')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    plt.show()

plot_history(history)

test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
print(f"\nFinální přesnost na testovacích datech: {test_acc:.4f}")

def plot_confusion_matrix(model, X_test, y_test, levels):
    # nejvyšší ppst
    y_pred_probs = model.predict(X_test)
    y_pred = np.argmax(y_pred_probs, axis=1)
    
    # z onehot na indexy
    y_true = np.argmax(y_test, axis=1)
    
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=levels, yticklabels=levels)
    
    plt.title('Matice záměn')
    plt.ylabel('Skutečná hodnota')
    plt.xlabel('Predikovaná hodnota')
    plt.show()

plot_confusion_matrix(model, X_test, y_test, [1, 3, 5, 7, 9])


def plot_prediction_confidence(model, X_test, y_test, levels, num_samples=3):
    indices = np.random.choice(len(X_test), num_samples, replace=False)
    predictions = model.predict(X_test[indices])
    true_labels = np.argmax(y_test[indices], axis=1)

    plt.figure(figsize=(15, 5))

    for i, idx in enumerate(indices):
        plt.subplot(1, num_samples, i + 1)
        
        colors = ['gray'] * len(levels)
        pred_idx = np.argmax(predictions[i])
        
        colors[pred_idx] = 'orange' 
        
        bars = plt.bar(range(len(levels)), predictions[i], color=colors, edgecolor='black', alpha=0.7)
        
        if pred_idx == true_labels[i]:
            bars[pred_idx].set_edgecolor('green')
            bars[pred_idx].set_linewidth(3)
        else:
            bars[true_labels[i]].set_edgecolor('red')
            bars[true_labels[i]].set_linewidth(3)
            bars[true_labels[i]].set_label('Skutečnost')

        plt.xticks(range(len(levels)), levels)
        plt.ylim(0, 1)
        plt.title(f"Vzorek č. {idx}\nSkutečnost: {levels[true_labels[i]]}")
        plt.ylabel('Pravděpodobnost (Jistota)')
        plt.xlabel('Hladina fenotypu')
        
    plt.tight_layout()
    plt.show()

plot_prediction_confidence(model, X_test, y_test, [1, 3, 5, 7, 9])
