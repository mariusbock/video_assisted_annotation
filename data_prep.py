import pandas as pd
import numpy as np
from datetime import datetime


def extract_elan_labels(data, subject, path):
    timestamps = pd.to_timedelta(data['timestamps'])
    data = data.reset_index(drop=True)
    labels = pd.read_csv(path, delimiter='\t', index_col=0, header=None, lineterminator='\n')
    labels = labels.reset_index()
    columns = ['layer', 'drop1', 'start', 'start[ms]', 'end', 'end[ms]', 'length', 'length[ms]', 'label']
    labels.columns = columns
    labels = labels.drop(['drop1'], axis=1)
    starting_point = pd.Timedelta(hours=0, minutes=0, seconds=0, microseconds=0).to_timedelta64()
    for row in labels.iterrows():
        start_time = row[1]['start']
        end_time = row[1]['end']
        dt_start = datetime.strptime(start_time, "%H:%M:%S.%f").time()
        timedelta_start = pd.Timedelta(hours=dt_start.hour, minutes=dt_start.minute, seconds=dt_start.second,
                                       microseconds=dt_start.microsecond).to_timedelta64()
        dt_end = datetime.strptime(end_time, "%H:%M:%S.%f").time()
        timedelta_end = pd.Timedelta(hours=dt_end.hour, minutes=dt_end.minute, seconds=dt_end.second,
                                     microseconds=dt_end.microsecond).to_timedelta64()

        start = starting_point + timedelta_start
        end = starting_point + timedelta_end
        start_index = np.abs(np.subtract(timestamps, start)).argmin()
        end_index = np.abs(np.subtract(timestamps, end)).argmin()
        try:
            data[subject].loc[start_index:end_index] = row[1]['label']
        except:
            print('error')

    return data

def get_gui_label_timestamps(file_type):
    import glob

    raw_data_folder = 'raw/'
    annotations = 'annotations/'
    gt = "groundtruth" + file_type
    annotations_wear_mad = glob.glob(annotations + "wear/*" + file_type + ".csv")
    annotations_wear_mad.sort()
    annotations_wear_elan = glob.glob(annotations + "wear/*" + file_type + ".txt")
    annotations_wear_elan.sort()
    annotations_wetlab_mad = glob.glob(annotations + "wetlab/*" + file_type + ".csv")
    annotations_wetlab_mad.sort()
    annotations_wetlab_elan = glob.glob(annotations + "wetlab/*" + file_type + ".txt")
    annotations_wetlab_elan.sort()
    wetlab_data = pd.read_csv(raw_data_folder + "wetlab" + file_type + ".csv", names=['timestamps', 'sbj', 'acc_x', 'acc_y', 'acc_z', gt, 'groundtruth1'], index_col=0).drop('groundtruth1', axis=1)
    wetlab_data.fillna('null_class', inplace=True)
    wear_data = pd.read_csv(raw_data_folder + "wear" + file_type + ".csv", index_col=0, names=['timestamps', 'right_arm_acc_x','right_arm_acc_y','right_arm_acc_z', 'labels']).rename({'labels': gt}, axis=1)
    wear_data.fillna('null_class', inplace=True)

    # wetlab
    for i, j in enumerate(annotations_wetlab_mad):
        j = j.split('.')[0].split('/')[-1]
        wetlab_data[j] = np.full(fill_value="null_class", shape=(len(wetlab_data[gt]), 1))
        labels = pd.read_csv(annotations_wetlab_mad[i])
        for n in range(0, labels.shape[0]):
            start = labels.start[n]
            end = labels.end[n]
            description = labels.description[n][2:-3]
            wetlab_data[j].loc[start:end] = description

    for i, j in enumerate(annotations_wear_mad):
        j = j.split('.')[0].split('/')[-1]
        wear_data[j] = np.full(fill_value="null_class", shape=(len(wear_data[gt]), 1))
        labels = pd.read_csv(annotations_wear_mad[i])
        for n in range(0, labels.shape[0]):
            start = labels.start[n]
            end = labels.end[n]
            description = labels.description[n][2:-3]
            wear_data[j].loc[start:end] = description

    for i, j in enumerate(annotations_wetlab_elan):
        j = j.split('.')[0].split('/')[-1]                
        wetlab_data[j] = np.full(fill_value="null_class", shape=(len(wetlab_data[gt]), 1))
        wetlab_data = extract_elan_labels(wetlab_data, j, annotations_wetlab_elan[i])

    for i, j in enumerate(annotations_wear_elan):
        j = j.split('.')[0].split('/')[-1]
        wear_data[j] = np.full(fill_value="null_class", shape=(len(wear_data[gt]), 1))
        wear_data = extract_elan_labels(wear_data, j, annotations_wear_elan[i])
    
    return wetlab_data.reset_index(drop=True), wear_data.reset_index(drop=True)

