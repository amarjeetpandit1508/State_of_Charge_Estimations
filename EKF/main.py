import numpy as np
import matplotlib.pyplot as plt
from utils import *
from ekf import run_ekf


Data = 'D:/AA Study/1. Python/Battery Projects/2. SoC Estimation/data/25 degC/04-20-19_05.34 780_LA92_25degC_Turnigy_Graphene.mat'
BATTERY_MODEL = 'D:/AA Study/1. Python/Battery Projects/2. SoC Estimation/data/BatteryModel.csv'
SOC_OCV = 'D:/AA Study/1. Python/Battery Projects/2. SoC Estimation/data/SOC-OCV.csv'


if __name__ == '__main__':

    LiPoly = load_data(Data)

    # LiPoly Capacity from data (in AH)
    nominal_capacity = 5.0
    
    # Calculate the SoC using coloumb counting for comparison
    # Charge +ve, Discharge -ve in DATA
    # Charge -ve, Discharge +ve in Formula
    LiPoly['Measured_SOC'] = (nominal_capacity + LiPoly['Ah']) * 100 / nominal_capacity
    # LiPoly['Measured_SOC'] = 100 * (1 - np.abs(LiPoly['Ah']) / nominal_capacity)

    # Resampling the data
    LiPoly = LiPoly.iloc[0::10]

    # Charging: -ve and discharging: +ve
    LiPoly['Measured_Current_R'] = LiPoly['Measured_Current'] * (-1)

    # Converting seconds to hours
    LiPoly['Time_Hours'] = LiPoly['Measured_Time'] / 3600

    # Load LiPoly Model and SoC-OCV data
    battery_model = pd.read_csv(BATTERY_MODEL)
    soc_ocv = pd.read_csv(SOC_OCV)

    print(battery_model.head())
    print(soc_ocv.head())

    # Calling run_ekf function for estimation
    SOC_Estimated, Vt_Estimated, Vt_Error = run_ekf(LiPoly, battery_model, soc_ocv)
    SOC_Estimated = simple_exponential_smoothing(SOC_Estimated, 0.25)

    percentage_error = np.abs((LiPoly['Measured_SOC'] - SOC_Estimated * 100) / LiPoly['Measured_SOC']) * 100
    print('Mean percentage error: {:.2f}%'.format(np.mean(percentage_error)))


    # Plotting the results
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))  # Set figure size

    # Terminal voltage plot
    axes[0, 0].plot(LiPoly['Time_Hours'], LiPoly['Measured_Voltage'], label='Measured Voltage')
    axes[0, 0].plot(LiPoly['Time_Hours'], Vt_Estimated, label='Estimated Voltage')
    axes[0, 0].set_ylabel('Terminal Voltage [V]')
    axes[0, 0].set_title('Measured vs. Estimated Terminal Voltage')
    axes[0, 0].legend()
    axes[0, 0].grid(True)

    # Terminal voltage error
    axes[0, 1].plot(LiPoly['Time_Hours'], Vt_Error, color='red')
    axes[0, 1].set_ylabel('Voltage Error [V]')
    axes[0, 1].set_title('Terminal Voltage Error')
    axes[0, 1].grid(True)

    # SOC comparison
    axes[1, 0].plot(LiPoly['Time_Hours'], LiPoly['Measured_SOC'], label='Measured SOC')
    axes[1, 0].plot(LiPoly['Time_Hours'], SOC_Estimated * 100., label='Estimated SOC (EKF)')
    axes[1, 0].set_ylabel('SOC [%]')
    axes[1, 0].set_xlabel('Time [Hours]')
    axes[1, 0].set_title('Measured vs. Estimated SOC')
    axes[1, 0].legend()
    axes[1, 0].grid(True)

    # SOC error
    axes[1, 1].plot(LiPoly['Time_Hours'], LiPoly['Measured_SOC'] - SOC_Estimated * 100., color='orange')
    axes[1, 1].set_ylabel('SOC Error [%]')
    axes[1, 1].set_xlabel('Time [Hours]')
    axes[1, 1].set_title('SOC Error')
    axes[1, 1].grid(True)

    plt.tight_layout()  # Prevent overlap
    plt.show()




