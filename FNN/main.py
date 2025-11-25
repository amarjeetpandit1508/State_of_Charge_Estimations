import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from utils import *
from create_data import preprocess_data
from train import Features


DATA_FILE = 'D:/AA Study/1. Python/Battery Projects/2. SoC Estimation/data/25 degC/04-20-19_05.34 780_LA92_25degC_Turnigy_Graphene.mat'


if __name__ == '__main__':

    LiPoly = load_data(DATA_FILE)
    LiPoly = preprocess_data(LiPoly)

    # Battery Capacity in Ah taken from Data
    Nominal_Cap = 5

    # Caculating SOC using coloumb counting for comparison
    LiPoly['Measured_SOC'] = 100 * (1 + (LiPoly['Ah'] / Nominal_Cap))

    # Converting Seconds to Hours
    LiPoly['Time_Hours'] = LiPoly['Measured_Time'] / 3600

    model = keras.models.load_model('D:/AA Study/1. Python/Battery Projects/2. SoC Estimation/FNN/best_turnigy_graphene_model.h5', compile=False)
    SOC_Estimated = model.predict(LiPoly[Features].values)*100
    SOC_Estimated = np.squeeze(SOC_Estimated)
    SOC_Estimated = simple_exponential_smoothing(SOC_Estimated, 0.2)

    epsilon = 1e-6  # avoid divide-by-zero
    percentage_error = np.abs((LiPoly['Measured_SOC'] - SOC_Estimated) * 100 / (LiPoly['Measured_SOC'] + epsilon))
    mean_error = np.mean(percentage_error)
    print(f"Mean percentage error: {mean_error:.2f}%")

    # mae = np.mean(np.abs(LiPoly['Measured_SOC'] - SOC_Estimated))
    # print(f"Mean absolute SOC error: {mae:.2f} %")

    # Plot the results
    fig, axes = plt.subplots(1, 2)

    # SOC comparison
    axes[0].plot(LiPoly['Time_Hours'], LiPoly['Measured_SOC'], label='Measured SOC')
    axes[0].plot(LiPoly['Time_Hours'], SOC_Estimated, label='Estimated SOC')
    axes[0].set_ylabel('SOC [%]')
    axes[0].set_xlabel('Time [Hours]')
    axes[0].set_title('Measured vs. Estimated SOC')
    axes[0].legend()
    axes[0].grid(True)

    # SOC error
    axes[1].plot(LiPoly['Time_Hours'], LiPoly['Measured_SOC'] - SOC_Estimated, color='orange')
    axes[1].set_ylabel('SOC Error [%]')
    axes[1].set_xlabel('Time [Hours]')
    axes[1].set_title('SOC Error')
    axes[1].grid(True)

    plt.show()
