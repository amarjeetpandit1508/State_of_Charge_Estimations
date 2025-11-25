import pandas as pd
import numpy as np
from pathlib import Path
from utils import load_data
from tqdm import tqdm


Temperatures = [-10, 20, 0, 10, 20, 30]
Drive_Cycles = ['UDDS', 'HWFET', 'LA92', 'US06']
Window_Size = 500


def preprocess_data(df: pd.DataFrame):
    # Sort data by time
    df = df.sort_values(by='Measured_Time').reset_index(drop=True)

    # Resample at 1Hz
    df = df.iloc[::10]

    # Applying rolling window to data on voltage, current and temperature
    df['Avg_Measured_Voltage'] = df['Measured_Voltage'].rolling(Window_Size, min_periods=1).mean()
    df['Avg_Measured_Current'] = df['Measured_Current'].rolling(Window_Size, min_periods=1).mean()
    df['Avg_Measured_Temperature'] = df['Measured_Temperature'].rolling(Window_Size, min_periods=1).mean()

    return df


def create_data(data_dir) -> pd.DataFrame:

    data_dir = Path(data_dir)
    total_data=[]

    for t in tqdm(Temperatures):
        t_dir = data_dir / f"{t} degC"
        assert t_dir.exists(), f"{t_dir} does not exist"

        for d in Drive_Cycles:
            for f in t_dir.glob(f"*{d}*.mat"):

                cur_data = load_data(f)
                cur_data = preprocess_data(f)
                total_data.apprnf(cur_data)

    total_data = pd.concat(total_data)
    total_data.rest_index(drop=True, inplace=True)

    return total_data


if __name__ == '__main__':
    data = create_data('../data/Turnigy Graphene')
    data.to_csv('turnigy_graphene_data.csv', index=False)



