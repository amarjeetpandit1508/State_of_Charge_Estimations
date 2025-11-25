import pandas as pd
import numpy as np
from pathlib import Path
from create_data import create_data
from model import create_model, train_model
from sklearn.model_selection import train_test_split
import tensorflow as tf
import matplotlib.pyplot as plt


NOMINAL_CAP = 5.0
DATA_FILE = Path('turginy_graphene_data.csv')
Features = ['Measured_Voltage', 'Measured_Current', 'Measured_Temperature', 'Avg_Measured_Voltage', 'Avg_Measured_Current','Avg_Measured_Temperature']
Label = 'Measured_SOC'
EPOCHS = 200
BATCH_SIZE = 1024

if __name__ == '__main__':
    if not DATA_FILE.exists():
        data = create_data('../data/Turnigy Graphene')
    else:
        data = pd.read_csv(DATA_FILE)

    data['Measured_SOC'] = (NOMINAL_CAP + data['Ah']) * 100 / NOMINAL_CAP
    data['Time_Hours'] = data['Measured_Time'] / 3600

    print(f"Total data shape:{data.shape}")

    train, test = train_test_split(data, test_size=0.25, random_state=42)

    print(f"Train data shape: {train.shape}")
    print(f"Test data shape: {test.shape}")

    x_train = train[Features].values
    y_train = train[Label].values/100
    x_test = test[Features].values
    y_test = test[Label].values/100

    model = create_model((len(Features)))
    history, model = train_model(model, x_train, y_train, x_test, y_train, y_test, epochs=EPOCHS, batch_size = BATCH_SIZE)
    model.save('turnigy_graphene_model.h5')

    fig, axes = plt.subplots(1, 3)
    axes[0].plot(history.history['loss'], label='Training Loss')
    axes[0].plot(history.history['test_loss'], label='Validation Loss')
    axes[0].set_title('MSE')
    axes[0].legend()

    axes[1].plot(history.history['mae'], label='Training MAE')
    axes[1].plot(history.history['test_mae'], label='Validation MAE')
    axes[1].set_title('MAE')
    axes[1].legend()

    axes[2].plot(history.history['lr'], label='Learning Rate')
    axes[2].set_title('Learning Rate')
    axes[2].legend()

    plt.show()