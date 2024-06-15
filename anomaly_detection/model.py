from keras.models import Model
from keras.layers import Input, Dense
from keras.optimizers import Adam
import pandas as pd
import numpy as np

class AutoencoderAnomalyDetection:
    def __init__(self, input_size, latent_dim, learning_rate=0.001):
        self.input_size = input_size
        self.latent_dim = latent_dim
        self.learning_rate = learning_rate
        self.model = self.build_autoencoder()
        self.slope_dict = {}

    def preprocess_data(self, data,_MDS):
        df = pd.DataFrame(data)
        df['CREATED_TIME_datetime'] = pd.to_datetime(df['CREATED_TIME'])

        re_df = pd.DataFrame()
        for name, group in df.groupby(pd.Grouper(key='CREATED_TIME_datetime', freq='72H')):
            self._slope_cal(self, _MDS, group)
            group['grm_72H'] = name
            group['ROWNUM'] = np.arange(1, len(group) + 1)
            group['slope'] = self.slope_dict['slope']
            group['bias'] = self.slope_dict['bias']
            re_df = pd.concat([re_df, group])

        data_values = re_df[_MDS].values.astype(np.float32)
        data_tensor = np.reshape(data_values, (-1, 1))

        return data_tensor

    def _slope_cal(self, _MDS, df):
        intercepts = [
            (name, 
             np.poly1d(np.polyfit(np.arange(1, len(group[_MDS[0]]) + 1), group[_MDS[0]], 1))[1],
             np.poly1d(np.polyfit(np.arange(1, len(group[_MDS[0]]) + 1), group[_MDS[0]], 1))[0])
            for name, group in df
        ]

        intercepts_t_s = [t[1] for t in intercepts]
        intercepts_t_b = [t[2] for t in intercepts]

        mean_last_slope = np.mean(intercepts_t_s)
        mean_last_bias = np.mean(intercepts_t_b)

        self.slope_dict['slope'] = mean_last_slope
        self.slope_dict['bias'] = mean_last_bias

        return self.slope_dict

    def build_autoencoder(self):
        input_layer = Input(shape=(self.input_size,))
        encoded = Dense(64, activation='relu')(input_layer)
        encoded = Dense(32, activation='relu')(encoded)
        encoded = Dense(self.latent_dim, activation='relu')(encoded)
        decoded = Dense(32, activation='relu')(encoded)
        decoded = Dense(64, activation='relu')(decoded)
        decoded = Dense(self.input_size, activation='sigmoid')(decoded)

        autoencoder = Model(inputs=input_layer, outputs=decoded)
        autoencoder.compile(optimizer=Adam(learning_rate=self.learning_rate), loss='mse')
        
        return autoencoder

    def train_autoencoder(self, data_tensor, num_epochs=50, batch_size=64):
        self.model.fit(data_tensor, data_tensor, epochs=num_epochs, batch_size=batch_size, shuffle=True, verbose=1)

    def detect_anomalies(self, new_data_tensor, threshold_multiplier=3):
        reconstructed_data = self.model.predict(new_data_tensor)
        mse = np.mean(np.power(new_data_tensor - reconstructed_data, 2), axis=1)

        threshold = np.mean(mse) + threshold_multiplier * np.std(mse)
        anomalies = np.where(mse > threshold)[0]

        return anomalies