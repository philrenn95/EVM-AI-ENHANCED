import os
import pandas as pd
import numpy as np

def FITBIT_GROUNDTRUTH():
    fit_bit_path = "/FITBIT/"

    participant_list = os.listdir(os.getcwd() + fit_bit_path)

    fit_bitData = {}

    for i in participant_list:
        tmp_path = os.getcwd() + os.sep + fit_bit_path + os.sep +  i
        TimeStamp = []
        Heartrate = []
        if '.DS_Store' in tmp_path:
             pass
        else:
            data = pd.read_csv(tmp_path)

        TimeStamp = np.array(data.iloc[:, 0].tolist())
        Heartrate = np.array(data.iloc[:, 1].tolist())

        combined_array = np.column_stack((TimeStamp, Heartrate))

        fit_bitData[i[:5]] = combined_array

    return fit_bitData
    




if __name__ == "__main__":
    fit_bitData = fit_bit_GROUNDTRUTH()