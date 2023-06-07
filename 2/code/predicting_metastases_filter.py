from typing import NoReturn, Tuple
import numpy as np
import pandas as pd
# pio.templates.default = "simple_white"

# stages - LA is level 3 - locally advanced, it is in range 3a 3b 3c, so we will make it average - 3b
# meaning every time we have

# new columns:
# diagnostic date year - (2023 - age),
#  אבחנה-Positive nodes /אבחנה-Nodes exam -

FEATURED_TO_DROP = ['id-hushed_internalpatientid', ' Form Name', ' Hospital', 'User Name', 'אבחנה-Basic stage',
                    'אבחנה-Margin Type', 'אבחנה-Side', 'אבחנה-Surgery date1', 'אבחנה-Surgery date2',
                    'אבחנה-Surgery date3',
                    'אבחנה-Surgery name1', 'אבחנה-Surgery name2', 'אבחנה-Surgery name3', 'אבחנה-Surgery sum',
                    'אבחנה-er', 'אבחנה-pr', 'אבחנה-KI67 protein', 'אבחנה-Positive nodes', 'אבחנה-Nodes exam',
                    'surgery before or after-Activity date',
                    'surgery before or after-Actual activity', 'אבחנה-Ivi -Lymphovascular invasion',
                    'אבחנה-Lymphatic penetration', 'אבחנה-M -metastases mark (TNM)', 'אבחנה-N -lymph nodes mark (TNM)'
    , 'אבחנה-T -Tumor mark (TNM)', 'אבחנה-Diagnosis date', 'אבחנה-Her2']

CATEGORICAL = ['אבחנה-Histological diagnosis']

all_features = ['Form Name', 'Hospital', 'User Name', 'אבחנה-Age', 'אבחנה-Basic stage',
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


def hot_one(df, category):
    df = pd.get_dummies(df, prefix=category + "_", columns=[category], dtype=int)
    return df


def translate_degree(stage):
    """
    translate the stage of the cancer to level of sevirity
    """
    # undiffined cases we set to avrage
    stage = str(stage)
    if stage == "Null":
        return None

    if stage[1] == "X":
        return None
    else:
        return int(stage[1])


def translate_stage_4_levels(stage):
    """
    translate the stage of the cancer to level of sevirity
    """
    # undiffined cases we set to avrage
    stage = str(stage)
    if stage == "Not yet Established":
        return None
    if stage == "nan":
        return None
    elif stage == "LA":
        return 2

    stage = stage.replace("Stage", "")
    return int(stage[0])


def translate_stage_17_levels(stage):
    """
    translate the stage of the cancer to level of sevirity
    """
    # undiffined cases we set to avrage
    stage = str(stage)
    if stage == "Not yet Established":
        return None
    if stage == "nan":
        return None
    elif stage == "LA":
        return 8

    stage = stage.replace("Stage", "")
    var = 4 * int(stage[0])
    if len(stage) == 2:
        return var + 1 + ord(stage[1]) - ord("a")
    else:
        return var


# mission 2: preprocess the data to be able to use it in the model,
# and use it wisely todo add descrioption of specific activities!


def filter_data(df) -> pd.DataFrame:
    """
    load data from a csv file filename.csv into a pandas dataframe
    """
    # TODO DROPNA?

    # add the percentage of positive nodes
    df["percentage_nodes"] = df['אבחנה-Positive nodes'] / df['אבחנה-Nodes exam']
    df = df.replace(np.inf, None)

    # the age he got the diagnosis is crucial
    df['אבחנה-Diagnosis date'] = pd.to_datetime(df['אבחנה-Diagnosis date'], format='%d/%m/%Y %H:%M')
    df["age_diagnosted"] = df['אבחנה-Diagnosis date'].dt.year - (2023 - df['אבחנה-Age'])
    # drop the features that we've decided not to use

    # do on-hot on the categorical features
    for cat_feature in CATEGORICAL:
        df = hot_one(df, cat_feature)

    # manage with the cancer stages
    df["אבחנה-Stage"] = df["אבחנה-Stage"].apply(translate_stage_17_levels)
    df["אבחנה-Histopatological degree"] = df["אבחנה-Histopatological degree"].apply(translate_degree)

    df = df.drop(FEATURED_TO_DROP, axis=1)

    return df


# def main():
#     df = load_data("2/data/train.feats.csv")
#     print(df.keys())
#
#
# if __name__ == '__main__':
#     main()
