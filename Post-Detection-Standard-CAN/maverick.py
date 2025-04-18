#!/usr/bin/env python3

# !pip3 install pyyaml h5py

import os
import gc
import time
import random
import numpy as np
import xgboost as xgb
import tensorflow as tf
from pathlib import Path
from keras import backend as K
import matplotlib.pyplot as plt
from collections import Counter
from tensorflow.keras import layers, Model
from tensorflow.keras.optimizers import Nadam
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
from sklearn.ensemble import RandomForestClassifier
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import accuracy_score, matthews_corrcoef, f1_score
from sklearn.metrics import roc_curve, auc, RocCurveDisplay, confusion_matrix

batchsize = 256

def set_seeds(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    tf.random.set_seed(seed)
    np.random.seed(seed)
    
def set_global_determinism(seed):
    set_seeds(seed=seed)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.config.threading.set_intra_op_parallelism_threads(1)

def convert(number, unit='micro'):
    if unit == 'micro':
        result = number * 1e6
    elif unit == 'milli':
        result = number * 1e3
    elif unit == 'nano':
        result = number * 1e9
    else:
        raise ValueError("Invalid unit. Choose from 'micro', 'milli', or 'nano'.")
    return round(result, 3)

class Maverick:
    def __init__(self, model, X_train = None, y_train = None, latent_shape = 20, lra=0.001, epochs = 50, load_filename = "", save_filename = "autoencoder_model.h5"):
        set_seeds(seed=42)
        set_global_determinism(seed=42)
        file_path = Path(save_filename)
        if file_path.exists() and file_path.suffix == '.h5':
            load_filename = save_filename
        if not (isinstance(model, RandomForestClassifier) or isinstance(model, xgb.XGBClassifier)):
            raise ValueError("The provided model should be either a Sklearn Random Forest or XGBoost model.")
        else:
            tf.keras.backend.clear_session()
            self.model = model  
        print("Initializing Maverick...")
        if load_filename != "":
            print("Loading File:", load_filename)
            def root_mean_squared_error(y_true, y_pred):
                difference = K.clip(K.abs(y_pred - y_true), 0.0, 1.0)
                return K.sqrt(K.mean(K.square(difference)))
            autoencoder = load_model(load_filename, custom_objects={'root_mean_squared_error': root_mean_squared_error})
            self.autoencoder = autoencoder
        else:
            if X_train is None or y_train is None:
                raise ValueError("Kindly provide the training set and labels.")
            ytrain_pred = model.predict(X_train) 
            xref_mask = ytrain_pred == y_train
            xref = X_train[xref_mask]
            dtype = np.uint16
            idref = model.apply(xref).astype(dtype)
            input_shape = idref.shape[1]
            # Define the encoder
            input_data = tf.keras.Input(shape=input_shape)
            encoded = layers.Dense(input_shape // 2, activation='relu')(input_data)
            hidden = layers.Dense(input_shape // 4, activation='relu')(encoded)
            latent = layers.Dense(latent_shape, activation='relu')(hidden)
            encoder = Model(inputs = input_data, outputs = latent, name = 'encoder')
            # Define the decoder
            latent_input = tf.keras.Input(shape=(latent_shape,))
            hidden_decoder = layers.Dense(input_shape // 4, activation='relu')(latent_input)
            decoded = layers.Dense(input_shape // 2, activation='relu')(hidden_decoder)
            output_data = layers.Dense(input_shape, activation='sigmoid')(decoded)
            decoder = Model(inputs=latent_input, outputs=output_data, name='decoder')
            # Create an instance of MinMaxScaler
            scaler = MinMaxScaler()
            idrefn = scaler.fit_transform(idref)
            del idref, X_train, y_train, xref, xref_mask, ytrain_pred
            gc.collect()
            # Define the autoencoder by combining the encoder and decoder
            autoencoder_input = tf.keras.Input(shape=(input_shape,))
            encoded_auto = encoder(autoencoder_input)
            decoded_auto = decoder(encoded_auto)
            autoencoder = Model(inputs=autoencoder_input, outputs=decoded_auto, name='autoencoder')
            def root_mean_squared_error(y_true, y_pred):
                difference = K.clip(K.abs(y_pred - y_true), 0.0, 1.0)
                return K.sqrt(K.mean(K.square(difference)))
            # Compile the autoencoder model
            autoencoder.compile(optimizer=Nadam(learning_rate=lra), loss = root_mean_squared_error)
            # Train the model
            t1 = time.time()
            autoencoder.fit(idrefn, idrefn, epochs=epochs, batch_size=batchsize, shuffle=True, validation_split=0.15)
            t2 = time.time()
            con_time = t2-t1
            autoencoder.save(save_filename)
            self.autoencoder = autoencoder
            self.con_time = con_time
        print("Initialization Complete!")
        
    def anomaly_scores(self, samples):
        dtype = np.uint16
        scaler = MinMaxScaler()
        idquery = self.model.apply(samples).astype(dtype)
        idqueryn = scaler.fit_transform(idquery)
        reconstructed_data = self.autoencoder.predict(idqueryn, batch_size = 256)
        rmse_per_sample = np.sqrt(np.mean(np.square(idqueryn - reconstructed_data), axis=1))
        del idquery, idqueryn
        gc.collect()
        return rmse_per_sample

    def get_roc_curve(self, samples, labels_sym):
        scores = self.anomaly_scores(samples)
        fpr, tpr, _ = roc_curve(labels_sym, scores, pos_label=1)
        roc_auc = auc(fpr, tpr)
        display = RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc, estimator_name='Maverick')
        display.plot()
        plt.show()
        
    def predict(self, samples, threshold):
        t0 = time.time()
        scores = self.anomaly_scores(samples)
        predictions = []
        preds = self.model.predict(samples)
        for distance, pred in zip(scores, preds):
            if distance <= threshold:
                predictions.append(pred)
            else:
                predictions.append(abs(1-pred))
        tn = time.time()
        pred_time = tn-t0
        self.pred_time = pred_time
        return predictions
        
    def get_metrics(self, samples, labels, threshold): 
        start_time = time.time()
        predictions = self.predict(samples, threshold)
        end_time = time.time()
        print("Maverick Average Prediction Time: ", convert((end_time - start_time)/len(samples)), "μs\n")
        print("Accuracy: ", round(accuracy_score(predictions, labels), 3))
        print("F1-Score: ", round(f1_score(predictions, labels), 3))
        print("MCC: ", round(matthews_corrcoef(predictions, labels), 3))
        tn, fp, fn, tp = confusion_matrix(predictions, labels).ravel()
        tpr = tp / (tp + fn)  # True positive rate
        tnr = tn / (tn + fp)  # True negative rate
        fpr = fp / (fp + tn)  # False positive rate
        fnr = fn / (fn + tp)  # False negative rate
        print(f"True Positive Rate (TPR): {tpr:.3f}")
        print(f"True Negative Rate (TNR): {tnr:.3f}")
        print(f"False Positive Rate (FPR): {fpr:.3f}")
        print(f"False Negative Rate (FNR): {fnr:.3f}")
        self.avg_time = convert((end_time - start_time)/len(samples))
            
    def plot_rmse_scores(self, samples, labels_sym):
        scores = self.anomaly_scores(samples)
        # Plot RMSE scores
        plt.figure(figsize=(10, 6))
        plt.plot(scores, markersize=1)
        plt.title('RMSE Scores for Each Sample')
        plt.xlabel('Sample Index')
        plt.ylabel('RMSE')
        plt.grid(True)
        plt.axvline(x=Counter(labels_sym)[0], color='red', linestyle='--', linewidth=1)
        plt.show()