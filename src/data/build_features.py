import os.path as path
import pandas as pd
import numpy as np
from tabulate import tabulate


def build_connection_features(df, is_attack=True, range=5):
    # Check if Event ID 1 found in range
    mask = df['EventID'].eq(1).rolling(range).sum().ge(1)
    df.loc[mask, 'EventID_1'] = '1'
    df['EventID_1'].fillna(0, inplace=True)

    # Check if Event ID 2 found in range
    mask = df['EventID'].eq(2).rolling(range).sum().ge(1)
    df.loc[mask, 'EventID_2'] = '1'
    df['EventID_2'].fillna(0, inplace=True)

    # Check if Event ID 3 found in range
    mask = df['EventID'].eq(3).rolling(range).sum().ge(1)
    df.loc[mask, 'EventID_3'] = '1'
    df['EventID_3'].fillna(0, inplace=True)

    # Check if Event ID 12 found in range
    mask = df['EventID'].eq(12).rolling(range).sum().ge(1)
    df.loc[mask, 'EventID_12'] = '1'
    df['EventID_12'].fillna(0, inplace=True)

    # Check if Event ID 13 found in range
    mask = df['EventID'].eq(13).rolling(range).sum().ge(1)
    df.loc[mask, 'EventID_13'] = '1'
    df['EventID_13'].fillna(0, inplace=True)

    # Check if Event ID 14 found in range
    mask = df['EventID'].eq(14).rolling(range).sum().ge(1)
    df.loc[mask, 'EventID_14'] = '1'
    df['EventID_14'].fillna(0, inplace=True)

    # Count number of unique Event ID's within range
    df['EventID'].fillna(0, inplace=True)
    df['num_unique_IDs'] = df['EventID'].rolling(range, min_periods=1).apply(lambda x: len(np.unique(x))).astype(int)

    # Count number of unique ProcessId's within range
    df['ProcessId'].fillna(0, inplace=True)
    try:
        df['num_unique_IDs'] = pd.to_numeric(df['ProcessId']).rolling(range, min_periods=1).apply(lambda x: len(np.unique(x))).astype(int)
    except ValueError:
        pass

    # Count number of unique ServiceName's within range (optional column)
    try:
        if 'ServiceName' in df.columns:
            df['num_unique_service_names'] = pd.to_numeric(df['ServiceName']).rolling(range, min_periods=1).apply(lambda x: len(np.unique(x))).astype(int)
    except ValueError:
        pass

    df['is_attack'] = is_attack

    return df


def build_time_features(df, is_attack=True, range=5):
    # Check if Event ID 1 found in range
    mask = df['EventID'].eq(1).rolling(range, on=df.index).sum().ge(1)
    df.loc[mask, 'EventID_1'] = '1'
    df['EventID_1'].fillna(0, inplace=True)

    # Check if Event ID 2 found in range
    mask = df['EventID'].eq(2).rolling(range, on=df.index).sum().ge(1)
    df.loc[mask, 'EventID_2'] = '1'
    df['EventID_2'].fillna(0, inplace=True)

    # Check if Event ID 3 found in range
    mask = df['EventID'].eq(3).rolling(range, on=df.index).sum().ge(1)
    df.loc[mask, 'EventID_3'] = '1'
    df['EventID_3'].fillna(0, inplace=True)

    # Check if Event ID 12 found in range
    mask = df['EventID'].eq(12).rolling(range, on=df.index).sum().ge(1)
    df.loc[mask, 'EventID_12'] = '1'
    df['EventID_12'].fillna(0, inplace=True)

    # Check if Event ID 13 found in range
    mask = df['EventID'].eq(13).rolling(range, on=df.index).sum().ge(1)
    df.loc[mask, 'EventID_13'] = '1'
    df['EventID_13'].fillna(0, inplace=True)

    # Check if Event ID 14 found in range
    mask = df['EventID'].eq(14).rolling(range, on=df.index).sum().ge(1)
    df.loc[mask, 'EventID_14'] = '1'
    df['EventID_14'].fillna(0, inplace=True)

    # Count number of unique Event ID's within range
    df['EventID'].fillna(0, inplace=True)

    df['num_unique_IDs'] = df['EventID'].rolling(
        range, on=df.index, min_periods=1).apply(lambda x: len(np.unique(x))).astype(int)

    # Count number of unique ProcessId's within range
    df['ProcessId'].fillna(0, inplace=True)
    try:
        df['num_unique_IDs'] = pd.to_numeric(df['ProcessId']).rolling(
            range, on=df.index, min_periods=1).apply(lambda x: len(np.unique(x))).astype(int)
    except ValueError:
        pass

    # Count number of unique ServiceName's within range (optional column)
    try:
        if 'ServiceName' in df.columns:
            df['num_unique_service_names'] = pd.to_numeric(df['ServiceName']).rolling(
                range, on=df.index, min_periods=1).apply(lambda x: len(np.unique(x))).astype(int)
    except ValueError:
        pass

    df['is_attack'] = is_attack

    return df


def write_to_file(df, attack):
    file_path = path.abspath(path.join(__file__, "../..")) + "\\features\\" + attack + ".txt"
    with open(file_path, "w+", encoding="utf-8") as f:
        f.write(tabulate(df, headers='keys'))


def write_csv_to_file(df, attack):
    file_path = path.abspath(path.join(__file__, "../../..")) + "\\data\\datasets_with_features\\" + attack + ".csv"
    df.to_csv(file_path)


def create_dataset(path):
    df = pd.read_csv(path)
    df['UtcTime'] = pd.to_datetime(df['UtcTime']).apply(lambda x: x.replace(microsecond=0))
    # df.drop_duplicates('UtcTime', keep='first', inplace=True)
    df.set_index(['UtcTime'], inplace=True)
    df = df.sort_index()

    try:
        df.groupby('SourceHostname')
    except KeyError:
        pass
    # TODO: Group by SourceHostName when finding rolling values
    return df


def main():
    dir = path.abspath(path.join(__file__, "../../..")) + "\\data\\processed\\"
    attacks = ["apt_sim", "kerberoasting", "brute_force", "dc_shadow", "dc_sync",
               "golden_ticket", "password_spraying", "remote_process_injection", "normal_events"]

    for attack in attacks:
        file_path = dir + attack + ".csv"
        inital_df = create_dataset(file_path)
        if attack == "normal_events":
            df = build_connection_features(inital_df, is_attack=False)
        else:
            df = build_connection_features(inital_df)
        write_csv_to_file(df, attack)


if __name__ == "__main__":
    main()

