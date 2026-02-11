from pandas_plink import read_plink
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import math

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.callbacks import EarlyStopping

from scipy import stats
from sklearn.impute import SimpleImputer
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from phenotype_levels import phenotype_levels_used
from data_preparation import run_phenotype_pipeline, snp_matrix

print(tf.config.list_physical_devices('GPU'))

fam_path = "/storage/plzen4-ntis/home/tadsova/rice_data/base_filtered_v0.7_0.8_10kb_1_0.8_50_1.fam"
pheno_path = "phenotypes/all_phenotypes.txt"
prefix = "/storage/plzen4-ntis/home/tadsova/rice_data/base_filtered_v0.7_0.8_10kb_1_0.8_50_1"

# výběr fenotypu
pheno = "PTH"

# masky
training_mask_probability = 0.9
test_mask_probability = 0.0

X = snp_matrix(prefix)
#print(X)

X_ready, y_ready, current_levels = run_phenotype_pipeline(pheno, X, pheno_path, fam_path)

def create_train_val_test_split(X, y, test_size=0.15, val_size=0.15, random_state=42):
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        shuffle=True
    )

    val_ratio = val_size / (1.0 - test_size)

    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val,
        y_train_val,
        test_size=val_ratio,
        random_state=random_state,
        shuffle=True
    )
    
    print("Dokončené rozdělení dat")
    print(f"X_train: {X_train.shape}, y_train: {y_train.shape}")
    print(f"X_val:   {X_val.shape},   y_val:   {y_val.shape}")
    print(f"X_test:  {X_test.shape},  y_test:  {y_test.shape}")
    
    return X_train, X_val, X_test, y_train, y_val, y_test

# train/test data
X_train, X_val, X_test, y_train, y_val, y_test = create_train_val_test_split(X_ready, y_ready)

class GenomicDataGenerator(tf.keras.utils.Sequence):
    def __init__(self, X, y, batch_size=32, mask_prob=0.9, shuffle=True):
        self.X = X
        self.y = y
        self.batch_size = batch_size
        self.mask_prob = mask_prob
        self.shuffle = shuffle
        self.indices = np.arange(len(self.X))
        self.on_epoch_end()

    def __len__(self):
        return int(np.floor(len(self.X) / self.batch_size))

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)

    def __getitem__(self, index):
        batch_indices = self.indices[index*self.batch_size:(index+1)*self.batch_size]
        X_batch = self.X[batch_indices]
        y_batch = self.y[batch_indices]

        num_samples, num_snps = X_batch.shape
        X_out = np.zeros((num_samples, num_snps, 4), dtype=np.float32)
        
        mask = np.random.rand(num_samples, num_snps) < self.mask_prob

        for val in [0, 1, 2]:
            X_out[:, :, val] = (X_batch == val) & (~mask)
        
        X_out[:, :, 3] = mask
        
        return X_out.reshape(num_samples, -1), y_batch

  def build_genomic_model(input_dim, num_classes, learning_rate=1e-6):
    model = models.Sequential([
        layers.Input(shape=(input_dim,)),

        layers.Dense(256),
        layers.BatchNormalization(),
        layers.Activation("relu"),
        #layers.Dropout(0.5),
        
        layers.Dense(128),
        layers.BatchNormalization(),
        layers.Activation("relu"),
        #layers.Dropout(0.3),
        
        layers.Dense(num_classes, activation='softmax')
    ])

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    
    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['categorical_accuracy']
    )
    
    return model

train_gen = GenomicDataGenerator(X_train, y_train, mask_prob=training_mask_probability)
val_gen   = GenomicDataGenerator(X_val, y_val, mask_prob=test_mask_probability, shuffle=False)
test_gen = GenomicDataGenerator(X_test, y_test, mask_prob=0.0, shuffle=False)

num_snps = X_train.shape[1]
classes_count = y_train.shape[1]

model = build_genomic_model(num_snps * 4, classes_count)

early_stop = EarlyStopping(
    monitor='val_loss',
    patience=15,  
    restore_best_weights=True
)

history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=50,
    callbacks=[early_stop],
    verbose=1
)

def plot_history(history):
    acc = history.history['categorical_accuracy']
    val_acc = history.history['val_categorical_accuracy']
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

test_loss, test_acc = model.evaluate(test_gen, verbose=0)
print(f"\nFinální přesnost na testovacích datech: {test_acc:.4f}")

def plot_confusion_matrix(model, generator, levels):
    y_pred_probs = model.predict(generator)
    y_pred = np.argmax(y_pred_probs, axis=1)
    
    y_true_list = []
    for i in range(len(generator)):
        _, y_batch = generator[i]
        y_true_list.append(np.argmax(y_batch, axis=1))
    
    y_true = np.concatenate(y_true_list)
    
    y_true = y_true[:len(y_pred)]
    
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=levels, yticklabels=levels)
    
    plt.title('Matice záměn (z generátoru s maskou)')
    plt.ylabel('Skutečná hodnota')
    plt.xlabel('Predikovaná hodnota')
    plt.show()

plot_confusion_matrix(model, test_gen, current_levels)

y_pred_probs = model.predict(test_gen, verbose=0)

y_true_list = []
for i in range(len(test_gen)):
    _, y_batch = test_gen[i]
    y_true_list.append(y_batch)

y_true = np.concatenate(y_true_list)

avg_confidence = np.mean(np.max(y_pred_probs, axis=1))
actual_accuracy = np.mean(np.argmax(y_pred_probs, axis=1) == np.argmax(y_true, axis=1))

print(f"Průměrná jistota modelu: {avg_confidence:.2%}")
print(f"Reálná přesnost modelu: {actual_accuracy:.2%}")

diff = avg_confidence - actual_accuracy
print(f"Rozdíl (Overconfidence gap): {diff:.2%}")
