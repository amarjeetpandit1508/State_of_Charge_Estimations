import scipy.io
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error


def load_data(mat_file: str) -> pd.DataFrame:
    """Load MATLAB .mat data into a pandas DataFrame."""
    data = scipy.io.loadmat(mat_file)
    meas = data['meas']
    dtypes = meas.dtype

    # Convert MATLAB structured array to Python dictionary
    meas = np.squeeze(meas).tolist()
    data_dict = {}
    for i, name in enumerate(dtypes.names):
        data_dict[name] = np.array(meas[i]).squeeze()

    # Convert to DataFrame
    df = pd.DataFrame(data_dict)

    # Rename columns for clarity
    df.rename(columns={
        'Time': 'Measured_Time',
        'Voltage': 'Measured_Voltage',
        'Current': 'Measured_Current',
        'Battery_Temp_degC': 'Measured_Temperature',
    }, inplace=True)

    return df


def compute_metrics(preds, labels):
    """Compute regression error metrics."""
    return {
        'mse': mean_squared_error(labels, preds),
        'mae': mean_absolute_error(labels, preds)
    }


def simple_exponential_smoothing(arr, alpha):
    """Apply simple exponential smoothing to a 1D numpy array."""
    arr = np.asarray(arr)
    smoothed = np.zeros(len(arr))
    smoothed[0] = arr[0]

    for i in range(1, len(arr)):
        smoothed[i] = alpha * arr[i] + (1 - alpha) * smoothed[i - 1]

    return smoothed
