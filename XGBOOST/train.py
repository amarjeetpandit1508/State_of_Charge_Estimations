import pickle
import pandas as pd
import numpy as np
from pathlib import Path
from xgboost import XGBRegressor
from create_data import create_data
from utils import compute_metrics
from sklearn.model_selection import train_test_split

Nominal_Cap = 5.0
Data_File = Path('turnigy_graphene_data.csv')
Features = ['Measured_Voltage', 'Measured_Current', 'Measured_Temperature', 'Avg_Measured_Voltage', 'Avg_Measured_Current','Avg_Measured_Temperature']
Label = 'Measured_SOC'
Save_File = Path('best_model.pkl')

if __name__ == '__main__':
    
    if not Data_File.exists():
        data = create_data(r"D:\AA Study\1. Python\Battery Projects\2. SoC Estimation\data")
        data.to_csv(Data_File, index=False)
    else:
        data = pd.read_csv(Data_File)

    data[Label] = (Nominal_Cap + data['Ah']) * 100 / Nominal_Cap
    # data['Measured_SOC'] = 100 * (1 - np.abs(data['Ah']) / Nominal_Cap)

    train, test = train_test_split(data, test_size=0.25, random_state=42)

    x_train = train[Features].values
    y_train = train[Label].values/100
    x_test = test[Features].values
    y_test = test[Label].values/100

    model = XGBRegressor(n_estimators=1000)
    model.fit(x_train, y_train)

    train_metrics = compute_metrics(model.predict(x_train), y_train)
    test_metrics = compute_metrics(model.predict(x_test), y_test)

    print(f"Train Metrics: {train_metrics}")
    print(f"Test Metrics: {test_metrics}")

    pickle.dump(model, open(Save_File, 'wb'))





