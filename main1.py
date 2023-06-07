from typing import NoReturn, Tuple
import numpy as np
import pandas as pd
pio.templates.default = "simple_white"

# stages - LA is level 3 - locally advanced, it is in range 3a 3b 3c, so we will make it average - 3b
# meaning every time we have
FEATURED_TO_DROP = ['id-hushed_internalpatientid', 'Form Name', ]

all_features = [' Hospital', 'User Name', 'אבחנה-Age', 'אבחנה-Basic stage',
       'אבחנה-Diagnosis date', 'אבחנה-Her2', 'אבחנה-Histological diagnosis',
       'אבחנה-Histopatological degree', 'אבחנה-Ivi -Lymphovascular invasion',
       'אבחנה-KI67 protein', 'אבחנה-Lymphatic penetration',
       'אבחנה-M -metastases mark (TNM)', 'אבחנה-Margin Type',
       'אבחנה-N -lymph nodes mark (TNM)', 'אבחנה-Nodes exam',
       'אבחנה-Positive nodes', 'אבחנה-Side', 'אבחנה-Stage',
       'אבחנה-Surgery date1', 'אבחנה-Surgery date2', 'אבחנה-Surgery date3',
       'אבחנה-Surgery name1', 'אבחנה-Surgery name2', 'אבחנה-Surgery name3',
       'אבחנה-Surgery sum', 'אבחנה-T -Tumor mark (TNM)', 'אבחנה-Tumor depth',
       'אבחנה-Tumor width', 'אבחנה-er', 'אבחנה-pr',
       'surgery before or after-Activity date',
       'surgery before or after-Actual activity',
       'id-hushed_internalpatientid']
print(len(all_features))

def load_data(filename: str) -> pd.DataFrame:
    """
    load data from a csv file filename.csv into a pandas dataframe
    """
    # TODO DROPNA?
    df = pd.read_csv(filename).drop_duplicates()
    df = df.drop(FEATURED_TO_DROP, axis=1)
    # mission 1: drop the not important ones and add other important ones

    # mission 2: preprocess the data to be able to use it in the model,
    # and use it wisely todo add descrioption of specific activities!
    return df
