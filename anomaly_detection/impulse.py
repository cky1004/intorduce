from model import *
import pandas as pd
import numpy as np


def main():
    data = {
        'data_set'
    }

    # feature
    input_size = 1
    latent_dim = 5

    # 오토 인코더 초기화
    ae = AutoencoderAnomalyDetection(input_size, latent_dim)
    # Col calculate
    _MDS = ['VALUE']
    # Preprocess the data
    data_tensor, re_df = ae.preprocess_data(data,_MDS)

    # Train
    ae.train_autoencoder(data_tensor)

    # Detect anomalies
    anomalies = ae.detect_anomalies(data_tensor)

    print("Anomalies detected at indices:", anomalies)

if __name__ == "__main__":
    main()
