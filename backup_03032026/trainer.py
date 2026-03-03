from pandas_plink import read_plink
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import math

import os
import csv
import json
import gc

import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, Callback

from scipy import stats
from sklearn.impute import SimpleImputer
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

from phenotype_levels import phenotype_levels_used
from data_preparation import run_phenotype_pipeline

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

class ProgressiveMaskingCallback(Callback):
    def __init__(self, generator, start_mask, end_mask, total_epochs):
        super().__init__()
        self.generator = generator
        self.start_mask = start_mask
        self.end_mask = end_mask
        self.total_epochs = total_epochs

    def on_epoch_begin(self, epoch, logs=None):
        new_mask_prob = self.start_mask + (self.end_mask - self.start_mask) * (epoch / self.total_epochs)
        
        new_mask_prob = min(new_mask_prob, self.end_mask)
        
        self.generator.mask_prob = new_mask_prob

def build_genomic_model(input_dim, num_classes, learning_rate=1e-6):
    model = models.Sequential([
        layers.Input(shape=(input_dim,)),
        
        layers.Dense(256, kernel_regularizer=regularizers.l2(0.01)),
        layers.BatchNormalization(),
        layers.Activation("relu"),

        layers.Dense(128, kernel_regularizer=regularizers.l2(0.01)),
        layers.BatchNormalization(),
        layers.Activation("relu"),
        
        layers.Dense(num_classes, activation='softmax')
    ])

    loss = tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1)
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    
    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['categorical_accuracy']
    )
    
    return model

def log_final_results(phenotype, pheno_lvls, apriori_acc, test_acc, test_loss, total_samples, filename="phenotype_benchmarks.csv"):
    file_exists = os.path.isfile(filename)
    num_classes = len(pheno_lvls)
    lvls_str = "-".join(map(str, pheno_lvls))
    
    with open(filename, mode='a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow([
                "phenotype",
                "num_classes",
                "levels_map",
                "apriori_acc", 
                "model_acc", 
                "test_loss", 
                "total_samples"
            ])
        
        writer.writerow([
            phenotype,
            num_classes,
            lvls_str,
            f"{apriori_acc:.4f}", 
            f"{test_acc:.4f}", 
            f"{test_loss:.4f}", 
            total_samples
        ])
    
    print(f"Výsledky pro {phenotype} uloženy.")

def save_history_json(history, phenotype_name):
    history_dict = {}
    for key, values in history.history.items():
        history_dict[key] = [float(v) for v in values]

    with open(f"history_data/history_data_2/{phenotype_name}_history.json", "w") as f:
        json.dump(history_dict, f, indent=4)

def train_single_phenotype(pheno_name, X_data, pheno_path, fam_path, results_file, config):
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    
    try:
        X_ready, y_ready, levels, apriori_acc = run_phenotype_pipeline(pheno_name, X_data, pheno_path, fam_path)
        total_n = len(y_ready)
        X_train, X_val, X_test, y_train, y_val, y_test = create_train_val_test_split(X_ready, y_ready)
        
        y_int = np.argmax(y_ready, axis=1)
        cw = compute_class_weight('balanced', classes=np.unique(y_int), y=y_int)
        cw_dict = dict(enumerate(cw))

        train_gen = GenomicDataGenerator(X_train, y_train, mask_prob=config['START_MASK'])
        val_gen   = GenomicDataGenerator(X_val, y_val, mask_prob=0.0, shuffle=False)
        test_gen  = GenomicDataGenerator(X_test, y_test, mask_prob=0.0, shuffle=False)

        model = build_genomic_model(X_train.shape[1] * 4, y_train.shape[1])
        
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, verbose=1),
            ProgressiveMaskingCallback(generator=train_gen, start_mask=config['START_MASK'], end_mask=config['END_MASK'], total_epochs=50)
        ]

        history = model.fit(
            train_gen,
            validation_data=val_gen,
            epochs=50, 
            callbacks=callbacks,
            class_weight=cw_dict,
            verbose=1
        )

        save_history_json(history, pheno_name)
        test_loss, test_acc = model.evaluate(test_gen, verbose=0)
        
        log_final_results(
            phenotype=pheno_name,
            pheno_lvls=levels,
            apriori_acc=apriori_acc,
            test_acc=test_acc,
            test_loss=test_loss,
            total_samples=total_n,
            filename=results_file
        )
        
    finally:
        K.clear_session()
        gc.collect()