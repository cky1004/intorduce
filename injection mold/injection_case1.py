import numpy as np
import pandas as pd
from urllib.parse import quote
import argparse
import tensorflow as tf
from preprocess_data import *
from b_trainer import Trainer
from util import *
import json

pd.set_option('mode.chained_assignment', None)
np.seterr(divide='ignore')

parser = argparse.ArgumentParser(description="Argparse Tutorial")
parser.add_argument("--setting")
args = parser.parse_args()

db = dict(
    server="localhost",
    port="xx",
    user="postgres",
    password="xx",
    database="xx",
    raw_data_table="xx",
)

class CNNModel(Trainer):
    def __init__(self, setting, input_shape, num_classes):
        super().__init__(setting)
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.epochs = setting['epoch']
        self.batch_size = setting['batch_size']
        self.model = self.build_model()
        self.model.compile(
            loss="mse",
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
            metrics=["mae", tf.keras.metrics.MeanAbsoluteError()],
        )

    def build_model(self):
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Conv2D(32, (3, 3), activation='linear', padding='same', input_shape=self.input_shape))
        model.add(tf.keras.layers.LeakyReLU(alpha=0.1))
        model.add(tf.keras.layers.MaxPooling2D((2, 2), padding='same'))
        model.add(tf.keras.layers.Dropout(0.30))
        model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='linear', padding='same'))
        model.add(tf.keras.layers.LeakyReLU(alpha=0.1))
        model.add(tf.keras.layers.MaxPooling2D((2, 2), padding='same'))
        model.add(tf.keras.layers.Flatten())
        model.add(tf.keras.layers.Dense(128))
        model.add(tf.keras.layers.LeakyReLU(alpha=0.1))
        model.add(tf.keras.layers.Dense(1, activation='elu'))
        return model

def main(setting):
    seed_everything(42)

    raw_data = Db_connect(db)
    df_base = raw_data.dbConnect('xxx')
    df_base_total = scale_df(df_base)
    augmented_df = augment_df(df_base_total)
    reshaped_df = reshape_df(augmented_df)

    model_data = reshaped_df.groupby(["MC_CODE", "ITEM_CODE"]).get_group(('호기명', '제품명'))
    (X_train, Y_train), (X_val, Y_val), (X_test, Y_test) = prepare_data(model_data)

    trainer = CNNModel(setting, input_shape=(X_train.shape[1], X_train.shape[2], 1), num_classes=1)
    
    es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=50)
    mc = tf.keras.callbacks.ModelCheckpoint('best_model.h5', monitor='val_loss', mode='min', verbose=1, save_best_only=True)
    
    print(f"Model fit starting")
    
    trainer.fit(
        (X_train, Y_train),
        validation_data=(X_val, Y_val),
        epochs=trainer.epochs,
        callbacks=[es, mc],
        batch_size=trainer.batch_size,
    )
    print(f"Fit completed")

    train_metric = get_model_metrics(
        model=trainer.model,
        task=ModelTask.REGRESSION,
        phase="train",
        X=X_train,
        y_true=Y_train,
    )
    
    test_metric = get_model_metrics(
        model=trainer.model,
        task=ModelTask.REGRESSION,
        phase="val",
        X=X_test,
        y_true=Y_test,
    )
    
    metric = {**train_metric, **test_metric}
    trainer.log_metric(metric)
    trainer.log_model("model", ModelType.TENSORFLOW)
    trainer.save_train_results()

if __name__ == "__main__":
    print('시작')
    main(json.loads(args.setting))
