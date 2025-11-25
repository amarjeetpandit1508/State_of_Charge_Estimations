# main.py
import numpy as np
import matplotlib.pyplot as plt
from utils import load_data
from ekf_op import run_ekf
import pandas as pd
import matplotlib.ticker as mtick

Data = r'D:/AA Study/1. Python/Battery Projects/2. SoC Estimation/data/25 degC/04-20-19_05.34 780_LA92_25degC_Turnigy_Graphene.mat'
BATTERY_MODEL = r'D:/AA Study/1. Python/Battery Projects/2. SoC Estimation/data/BatteryModel.csv'
SOC_OCV = r'D:/AA Study/1. Python/Battery Projects/2. SoC Estimation/data/SOC-OCV.csv'

if __name__ == '__main__':

    LiPoly = load_data(Data)
    nominal_capacity = 5.0
    LiPoly['Measured_SOC'] = (nominal_capacity + LiPoly['Ah']) * 100 / nominal_capacity

    # Downsample to reduce computation (your original code used step 10)
    LiPoly = LiPoly.iloc[0::10].reset_index(drop=True)

    # Correct sign so charging/discharging matches EKF expectation
    LiPoly['Measured_Current_R'] = LiPoly['Measured_Current'] * (-1)

    LiPoly['Time_Hours'] = LiPoly['Measured_Time'] / 3600.0

    # Load battery model and SOC-OCV
    battery_model = pd.read_csv(BATTERY_MODEL)
    soc_ocv = pd.read_csv(SOC_OCV)

    # Run EKF (now returns params as well)
    SOC_est, Vt_est, Vt_err, Params = run_ekf(LiPoly, battery_model, soc_ocv)

    # Compute percentage error vs measured SOC (measured_SOC already in %)
    percent_error = np.abs((LiPoly['Measured_SOC'].values - SOC_est * 100.0) / (LiPoly['Measured_SOC'].values + 1e-12)) * 100.0
    print('Mean percentage error: {:.3f}%'.format(np.nanmean(percent_error)))


    # --- Extract time ---
    time_hours = LiPoly['Time_Hours'].values

    # ------------------------------
    # 1. Voltage, Voltage Error, SOC
    # ------------------------------
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

    # --- a) Terminal Voltage ---
    axes[0].plot(time_hours, LiPoly['Measured_Voltage'].values, label='Measured Voltage', alpha=0.6)
    axes[0].plot(time_hours, Vt_est, '--', label='Estimated Voltage', alpha=0.8)
    axes[0].set_ylabel('Voltage [V]')
    axes[0].set_title('Terminal Voltage')
    axes[0].grid(True)
    axes[0].legend()

    # --- b) Voltage Error ---
    axes[1].plot(time_hours, Vt_err, color='r')
    axes[1].set_ylabel('Voltage Error [V]')
    axes[1].set_title('Voltage Residual')
    axes[1].grid(True)

    # --- c) SOC ---
    axes[2].plot(time_hours, LiPoly['Measured_SOC'].values, label='Measured SOC')
    axes[2].plot(time_hours, SOC_est * 100.0, '--', label='EKF Estimated SOC')
    axes[2].set_xlabel('Time [hours]')
    axes[2].set_ylabel('SOC [%]')
    axes[2].set_title('State of Charge (SOC)')
    axes[2].grid(True)
    axes[2].legend()

    plt.tight_layout()
    plt.show()

    # ------------------------------
    # 2. Online Parameter Estimates
    # ------------------------------
    param_names = ['R0', 'R1', 'R2', 'C1', 'C2']
    num_params = len(param_names)

    fig, axes = plt.subplots(2, 3, figsize=(14, 6), sharex=True)
    axes = axes.flatten()  # flatten 2x3 grid for easier indexing

    for i in range(num_params):
        ax = axes[i]
        ax.plot(time_hours, Params[:, i])
        ax.set_title(param_names[i])
        ax.set_ylabel(param_names[i])
        ax.grid(True)
        
        # Format y-axis for C1 and C2 to avoid scientific notation
        if param_names[i] in ['C1', 'C2']:
            ax.yaxis.set_major_formatter(mtick.ScalarFormatter())
            ax.ticklabel_format(style='plain', axis='y')
            ax.yaxis.get_major_formatter().set_useOffset(False)

    # Hide unused subplot (bottom right corner)
    axes[-1].axis('off')

    # Set common x-label for all parameter plots
    for ax in axes[:num_params]:
        ax.set_xlabel('Time [hours]')

    fig.suptitle("Online Parameter Estimates", fontsize=14)
    plt.tight_layout()
    plt.show()
