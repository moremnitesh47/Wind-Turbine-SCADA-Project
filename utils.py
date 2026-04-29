import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler


merge_features = [
    [18, 19, 20, 21],               # Generator cooling air temperatures (Environmental)
    [27, 28, 29, 30, 31, 32, 33, 34],  # Yaw motor currents (Yaw/Axis)
    [36, 38],                       # DC link voltages (Electrical/Converter)
    [49, 50],                       # Hydraulic azimuth circuit pressures A/B (Hydraulic System)
    [51, 52],                       # Hydraulic memory pressures A/B (Hydraulic System)
    [54, 55],                       # Rotor brake pressures A/B (Pitch/Rotor Brake)
    [56, 57],                       # Line currents (Power input phases)
    [58, 59, 60],                   # Line voltages (Power input phases)
    [62, 63, 64],                   # Axis temperatures 1–3 (Pitch/Yaw/Axis)
    [66, 67, 68],                   # Internal consumption currents L1–L3 (Electrical)
    [69, 70],                       # Internal power consumption L2, L3 (Electrical)
    [71, 72, 73],                   # Internal consumption voltages L1–L3 (Electrical)
    [90, 91],                       # Nacelle vibration sensors (Vibration/Mechanical)
    [92, 93],                       # Transverse vibration sensors (Vibration/Mechanical)
    [97, 98, 99],                   # Gearbox filter pollution indicators (Gear Oil/Mechanical)
    [100, 101, 102, 103, 104, 105], # Rotor blade & motor positions (Pitch system)
    [107, 108],                     # Internal consumption power (Electrical)
    [109, 110],                     # Pressure diff. filter stage A (Environmental/Mechanical)
    [111, 112],                     # Pressure diff. filter stage B (Environmental/Mechanical)
    [113, 114],                     # Pressure diff. spinner (Environmental/Mechanical)
    [115, 116],                     # Spinner pressure (Environmental)
    [117, 118],                     # Gearbox oil pressure inputs (Mechanical)
    [123, 124, 125, 126],           # Wind direction sensors (Environmental/Wind)
    [127, 128, 129, 130, 131, 132], # HV grid currents and generator RMS currents (Electrical)
    [133, 134, 135],                # Axis line-to-line RMS currents (Electrical/Axis)
    [136, 137, 138],                # Generator RMS voltages (Electrical)
    [139, 140, 141],                # HV grid phase voltages (Electrical)
    [144, 145, 146, 147],           # Rotor and shaft speeds (Mechanical)
    [148, 149],                     # Axis motor RPM (Mechanical/Axis)
    [151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166],                     # Planetary bearing temperatures (Mechanical)
    [168, 169],                     # Axial bearing temperatures (Mechanical)
    [173, 174],                     # Generator cooling air inlet temps (Environmental)
    [178, 179],                     # Hydraulic oil tank temperatures (Hydraulic System)
    [182, 185],                     # Axis 3 motor temp PT100 and Axis 3 KTY84 (Mechanical)
    [181, 184],                     # Axis 2 motor temp PT100 and KTY84 (Mechanical)
    [180, 183],                     # Axis 1 motor temp PT100 and KTY84 (Mechanical)
    [186, 187],                     # Gearbox oil temperatures (Mechanical)
    [189, 190],                     # Gearbox oil inlet temperatures (Mechanical)
    [191, 192],                     # Transformer oil temperatures (Electrical)
    [194, 195],                     # Rotor bearing inner ring temps (Mechanical)
    [196, 197, 198],                # Rotor bearing temperatures 1–3 (Mechanical)
    [199, 200, 201, 202, 203, 204],                      # Stator winding temperatures (Electrical)
    [206, 207],                     # Electrical cabinet temps 1–2 (Environmental)
    [208, 209],                     # Axis cooling element temps 1–2 (Mechanical)
    [216, 221, 224],                # Axis 3 battery voltages (Pitch system)
    [219, 222],                     # Axis 1 battery voltages (Pitch system)
    [220, 223],                     # Axis 2 battery voltages (Pitch system)
    [228, 229],                     # Generator inlet cooling water temps (Environmental)
    [231, 232],                     # Cooling system pressures (Environmental)
    [233, 234],                     # Generator outlet water temps (Environmental)
    [2, 5, 6, 17],                  # Power/converter readings (Electrical)
    [235, 236, 237]                 # Wind speed sensors (Environmental/Wind)
]

drop_features = [
    "sensor_1_avg",                 # Generator acceleration avg (rarely used)
    "sensor_7_std",                # Ambient temperature std (low variation)
    "sensor_9_avg", "sensor_9_std",  # Battery charge 1 (redundant)
    "sensor_10_avg", "sensor_10_std", # Battery charge 2 (redundant)
    "sensor_16_std",               # Converter angle std (low signal strength)
    "sensor_39_std",               # Electrical cabinet temp std (low variation)
    "sensor_77_std", "sensor_78_std", "sensor_79_std", "sensor_80_std",  # Cooler motor current stds (low variance)
    "sensor_81_avg",               # Motor current hydraulic pump A avg (redundant)
    "sensor_90_merged_std",        # Nacelle vibration std (redundant)
    "sensor_92_merged_std",        # Vibration transverse 1 std (redundant)
    "sensor_95_avg", "sensor_95_std",  # Filter pump pressure (rarely active)
    "sensor_96_avg",               # Pollution indicator pump 2 avg
    "sensor_97_merged_std",        # Gearbox filter A std
    "sensor_167_std",              # Transformer container air temp std
    "sensor_178_merged_std",       # Hydraulic oil temp 1 std
    "sensor_188_std",              # Transformer oil temp std
    "sensor_191_merged_std",       # Main transformer oil temp 1 std
    "sensor_193_std",              # Platform temperature std
    "sensor_196_merged_std",       # Rotor bearing temp 1 std
    "sensor_205_std",              # Electrical cabinet temp std
    "sensor_206_merged_std",       # Electrical cabinet 1 std
    "sensor_218_std",              # 24V nacelle voltage std
    "sensor_225_std",              # Water conductivity std
    "sensor_227_std"               # Water pressure PT2 std
]






def check_correlation(df: pd.DataFrame, numbers: list[int]) -> pd.DataFrame:
    """
    Compute and return the Pearson correlation matrix for the specified sensor columns.

    :param df: DataFrame containing columns like 'sensor_1_avg', 'sensor_2_avg', …
    :param numbers: List of sensor numbers to include in the correlation.
    :return: A DataFrame of pairwise correlations among those sensor columns.
    """
    # Build list of sensor column names
    sensors = [f"sensor_{n}_avg" for n in numbers]

    # Verify that each column exists in the DataFrame
    missing = set(sensors) - set(df.columns)
    if missing:
        raise KeyError(f"Missing columns in DataFrame: {missing}")

    # Compute and return the correlation matrix
    corr_matrix = df[sensors].corr()
    return corr_matrix

###########################################################################################################################
def merge_sensor(df, sensors):

    """
    Merge average and std columns of specified sensor group into a single average and std column.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe with sensor readings.
    
    sensors : list of int
        List of sensor identifiers to be merged.
    
    Returns
    -------
    pd.DataFrame
        Updated dataframe with merged sensor columns.
    """
    if sensors == [2, 5, 6, 17]:
        names_avg = ['power_' + str(number) + '_avg' for number in sensors]
        names_std = ['power_' + str(number) + '_std' for number in sensors]
        new_name_avg = 'power_' + str(sensors[0]) + '_merged_avg'
        new_name_std = 'power_' + str(sensors[0]) + '_merged_std'
    elif sensors== [235, 236, 237]:
        names_avg = ['wind_speed_' + str(number) + '_avg' for number in sensors]
        names_std = ['wind_speed_' + str(number) + '_std' for number in sensors]
        new_name_avg = 'wind_speed_' + str(sensors[0]) + '_merged_avg'
        new_name_std = 'wind_speed_' + str(sensors[0]) + '_merged_std'
    else:
        names_avg = ['sensor_' + str(number) + '_avg' for number in sensors]
        names_std = ['sensor_' + str(number) + '_std' for number in sensors]
        new_name_avg = 'sensor_' + str(sensors[0]) + '_merged_avg'
        new_name_std = 'sensor_' + str(sensors[0]) + '_merged_std'
    df[new_name_avg] = df[names_avg].mean(axis=1)
    df[new_name_std] = df[names_std].mean(axis=1)
    df = df.drop(names_avg, axis=1)
    df = df.drop(names_std, axis=1)
    return df

def drop_minmax(df):

    """
    Drop all columns containing 'min' or 'max' in their names.
    
    Parameters
    ----------
    df : pd.DataFrame
    
    Returns
    -------
    pd.DataFrame
    """
    columns= df.columns
    for col in columns:
        if 'min' in col or 'max' in col:
            df.drop(col, axis=1, inplace=True)
    return df


def drop_sensors(df, drop_list):
    df.drop(columns=drop_list, inplace=True)
    return df




def preprocess_pipeline(
    df,
    merge_sensors=merge_features,
    drop_sensors_list=drop_features,
    scaled=True
               ):
    
    """
    Drop specified sensor columns from the dataframe.
    
    Parameters
    ----------
    df : pd.DataFrame
    drop_list : list of str
        List of column names to be dropped.
    
    Returns
    -------
    pd.DataFrame
    """

    if scaled:
        scaler = StandardScaler()
        all_numeric = df.select_dtypes(include=[np.number]).columns
        numeric_cols = all_numeric.difference(['status_type_id','asset_id','id'])
        df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

    df = drop_minmax(df)

    for sensor_group in merge_sensors:
        df = merge_sensor(df, sensor_group)

    # Now drop_sensors still refers to the function,
    # and drop_sensors_list is your list of columns to drop.
    df = drop_sensors(df, drop_list=drop_sensors_list)

    return df


########################################################################################################################

import numpy as np
from collections import defaultdict, deque
from collections import defaultdict
import pandas as pd

def get_correlated_sensor(df, threshold=0.95, exclude_cols=None):
    """
    Return maximal groups of sensors (cliques) where each pair has
    |corr| >= threshold.

    Parameters
    ----------
    df : pandas.DataFrame
        Input dataframe.
    threshold : float, default=0.95
        Correlation cutoff.
    exclude_cols : list of str, optional
        Columns to ignore. Defaults to ['train_test', 'time_stamp', 'asset_id'].

    Returns
    -------
    cliques : list of list of str
        Each inner list is a maximal clique of sensor names.
    """
    # 1) set default excludes
    if exclude_cols is None:
        exclude_cols = ['train_test', 'time_stamp', 'asset_id']

    # 2) compute absolute correlation matrix on feature columns
    features = list(df.columns.difference(exclude_cols))
    corr = df[features].corr().abs()

    # 3) build adjacency sets for edges where corr>=threshold
    adj = {f: set() for f in features}
    for i, f_i in enumerate(features):
        for j in range(i+1, len(features)):
            f_j = features[j]
            if corr.at[f_i, f_j] >= threshold:
                adj[f_i].add(f_j)
                adj[f_j].add(f_i)

    # 4) Bron–Kerbosch to find all maximal cliques
    def bron_kerbosch(R, P, X):
        if not P and not X:
            yield R
        for v in list(P):
            yield from bron_kerbosch(R | {v},
                                     P & adj[v],
                                     X & adj[v])
            P.remove(v)
            X.add(v)

    all_cliques = [
        sorted(list(clique))
        for clique in bron_kerbosch(set(), set(features), set())
        if len(clique) > 1
    ]

    return all_cliques
