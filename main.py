import pandas as pd
import data_prep as dp
import numpy as np
import matplotlib.pyplot as plt
import os
import json
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap

pd.options.mode.chained_assignment = None 
np.seterr(divide='ignore', invalid='ignore')


def compare_timelines(gt, timeline_1, name, cmap1, n_classes):
    """Creates color-coded visualization of participant's annotated timeline and ground truth annotations. Saves visualization as .png file.
    
    Parameters
    ----------
    gt : list
        List of ground truth labels (per-record basis)
    timeline_1 : list
        List of participant's annotation labels (per-record basis)
    name : str
        File name of created png file
    cmap1 : list
        Hex values of colors to be used for each label
    n_classes : int
        Number of classes in dataset
    """
    import matplotlib.pyplot as plt
    from matplotlib.ticker import StrMethodFormatter
    
    # plot 1:
    fig, (gt_ax, ax1) = plt.subplots(2, 1, sharex=True, figsize=(6, 3), layout="constrained")
    gt_ax.set_yticks([])
    gt_ax.pcolor([gt], cmap=cmap1, vmin=0, vmax=n_classes)    
    ax1.set_yticks([])
    ax1.pcolor([timeline_1], cmap=cmap1, vmin=0, vmax=n_classes)
    
    plt.gca().xaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))  # No decimal places
    plt.xticks([])
    plt.savefig(os.path.join('plots', name + '.png'))
    plt.close()


def Filter(string, substr):
    """Filters list of strings for whether each string contains a substring.
    
    Parameters
    ----------
    string : list
        List of strings to be filtered
    substr : str
        Substring which should be contained in strings
    Returns
    -------
    list
        List of strings which contained substring
    """
    return [str for str in string if any(sub in str for sub in substr)]


def compute_scores(data, name, gt_column, subjs, experts, novices, mad, elan, label_dict=None, color_map=None):
    """Computes the F1-score, Cohens Kappa and NULL-class accuracy of each session of each participant. Writes results to two JSON-formatted text files.
    
    Parameters
    ----------
    data : list
        DataFrame containing the sensor data, ground truth and participants' annotations
    name : str
        Dataset name to be used
    gt_column : str
        Column name which contains ground truth labels
    subjs : str
        List of all subjects ids
    experts : list
        List of session ids which were commenced by experts
    novices : list
        List of session ids which were commenced by novices
    mad : list
        List of session ids which were commenced using the MAD-GUI
    elan : list
        List of session ids which were commenced using the ELAN player
    label_dict : dict, optional
        Label dictionary of label string and corresponding id
    color_map : list, optional
        Color map for labels which should be applied during visualization        
    """
    from sklearn.metrics import cohen_kappa_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
    result = dict()
    all_gt = np.array([])
    all_pred = np.array([])
    mad_gt = np.array([])
    mad_pred = np.array([])
    elan_gt = np.array([])
    elan_pred = np.array([])
    experts_gt = np.array([])
    experts_pred = np.array([])
    novices_gt = np.array([])
    novices_pred = np.array([])

    if not os.path.exists('plots'):
        os.mkdir('plots')
        
    for i, subj in enumerate(subjs):
        file_type = '_'.join(subj.split('_')[-2:])
        gt = np.squeeze(data[[gt_column + '_' + file_type]].replace(label_dict).fillna(0).values)
        cc = np.squeeze(data[[subj]].replace(label_dict).fillna(0).values)
        compare_timelines(gt, cc, str(int(i)) + '_' + name + '_' + subj, color_map, len(label_dict))
        conf_mat_sbj = confusion_matrix(gt, cc, normalize='true', labels=range(len(label_dict)))
        result[subj + '_f1_score'] = f1_score(gt, cc, average='macro', labels=np.unique(np.concatenate((gt, cc))))
        result[subj + '_cohens'] = cohen_kappa_score(gt, cc, labels=np.unique(np.concatenate((gt, cc))))
        result[subj + '_cohens'] = cohen_kappa_score(gt, cc, labels=np.unique(np.concatenate((gt, cc))))
        result[subj + '_null_accuracy'] = (conf_mat_sbj.diagonal()/conf_mat_sbj.sum(axis=1))[0]

        if subj in experts:
            experts_gt = np.append(experts_gt, gt)
            experts_pred = np.append(experts_pred, cc)
        elif subj in novices:
            novices_gt = np.append(novices_gt, gt)
            novices_pred = np.append(novices_pred, cc)
        if subj in mad:
            mad_gt = np.append(mad_gt, gt)
            mad_pred = np.append(mad_pred, cc)
        elif subj in elan:
            elan_gt = np.append(elan_gt, gt)
            elan_pred = np.append(elan_pred, cc)
            
        all_gt = np.append(all_gt, gt)
        all_pred = np.append(all_pred, cc)
            
    # experts
    conf_mat_ex = confusion_matrix(experts_gt, experts_pred, normalize='true', labels=range(len(label_dict)))
    _, ax = plt.subplots(figsize=(15, 15), layout="constrained")
    ax.set_title('Confusion Matrix (Experts)')
    conf_disp_ex = ConfusionMatrixDisplay(confusion_matrix=conf_mat_ex, display_labels=label_dict.keys())    
    conf_disp_ex.plot(ax=ax, xticks_rotation='vertical', colorbar=False)
    plt.savefig(os.path.join('plots', 'experts_'+ name + '.png'))    
    plt.close()
    
    # novices
    conf_mat_nov = confusion_matrix(novices_gt, novices_pred, normalize='true', labels=range(len(label_dict)))
    _, ax = plt.subplots(figsize=(15, 15), layout="constrained")
    ax.set_title('Confusion Matrix (Novices)')
    conf_disp_nov = ConfusionMatrixDisplay(confusion_matrix=conf_mat_nov, display_labels=label_dict.keys())    
    conf_disp_nov.plot(ax=ax, xticks_rotation='vertical', colorbar=False)
    plt.savefig(os.path.join('plots', 'novices_'+ name + '.png'))  
    plt.close()

    # mad
    conf_mat_mad = confusion_matrix(mad_gt, mad_pred, normalize='true', labels=range(len(label_dict)))
    result['mad' + '_f1_score'] = f1_score(mad_gt, mad_pred, average='macro', labels=range(len(label_dict)), zero_division=0)
    _, ax = plt.subplots(figsize=(15, 15), layout="constrained")
    ax.set_title('Confusion Matrix (MAD-GUI)')
    conf_disp_mad = ConfusionMatrixDisplay(confusion_matrix=conf_mat_mad, display_labels=label_dict.keys())    
    conf_disp_mad.plot(ax=ax, xticks_rotation='vertical', colorbar=False)
    plt.savefig(os.path.join('plots', 'mad_'+ name + '.png'))    
    plt.close()
    
    # elan
    conf_mat_elan = confusion_matrix(elan_gt, elan_pred, normalize='true', labels=range(len(label_dict)))
    result['elan' + '_f1_score'] = f1_score(elan_gt, elan_pred, average='macro', labels=range(len(label_dict)), zero_division=0)
    _, ax = plt.subplots(figsize=(15, 15), layout="constrained")
    ax.set_title('Confusion Matrix (ELAN)')
    conf_disp_elan = ConfusionMatrixDisplay(confusion_matrix=conf_mat_elan, display_labels=label_dict.keys())    
    conf_disp_elan.plot(ax=ax, xticks_rotation='vertical', colorbar=False)
    plt.savefig(os.path.join('plots', 'elan_'+ name + '.png'))    
    plt.close()

    # all
    conf_mat_all = confusion_matrix(all_gt, all_pred, normalize='true', labels=range(len(label_dict)))
    result['all' + '_f1_score'] = f1_score(all_gt, all_pred, average='macro', labels=range(len(label_dict)), zero_division=0)
    _, ax = plt.subplots(figsize=(15, 15), layout="constrained")
    ax.set_title('Confusion Matrix (All)')
    conf_disp_all = ConfusionMatrixDisplay(confusion_matrix=conf_mat_all, display_labels=label_dict.keys())    
    conf_disp_all.plot(ax=ax, xticks_rotation='vertical', colorbar=False)
    plt.savefig(os.path.join('plots', 'all_'+ name + '.png'))
    plt.close()
    
    with open('overall_results_' + name + '.txt', 'w') as file:
        file.write(json.dumps(result, indent=3))


if __name__ == '__main__':
    raw_data_folder = 'sensor_data'
    annotations = 'annotations'
    nov = ['sbj_0', 'sbj_1', 'sbj_2', 'sbj_3', 'sbj_4', 'sbj_5', 'sbj_6', 'sbj_7', 'sbj_8', 'sbj_9']
    ex = ['sbj_10', 'sbj_11', 'sbj_12', 'sbj_13', 'sbj_14']
    wetlab_0_10, wear_0_10 = dp.get_gui_label_timestamps('_0_10', raw_data_folder, annotations)
    wetlab_10_20, wear_10_20 = dp.get_gui_label_timestamps('_10_20', raw_data_folder, annotations)

    wetlab = pd.concat((wetlab_0_10, wetlab_10_20), axis=1)
    wear = pd.concat((wear_0_10, wear_10_20), axis=1)

    label_dict_wetlab = {
        'null_class': 0,
        'pouring': 1,
        'transfer': 2,
        'pipetting': 3,
        'stirring': 4,
        'cutting': 5,
        'pestling': 6,
        'mixing': 7,
    }
    label_dict_wear = {
        'null_class': 0,
        'jogging (sidesteps)': 1,
        'bench-dips': 2,
        'stretching (shoulders)': 3,
        'jogging (butt-kicks)': 4,
        'burpees': 5,
        'lunges': 6,
    }

    wear_subjs = [col for col in wear.columns if (('mad' in col) or ('elan' in col))]
    wetlab_subjs = [col for col in wetlab.columns if (('mad' in col) or ('elan' in col))]
    expert_subjs = Filter(wear_subjs + wetlab_subjs, ex)
    novice_subjs = Filter(wear_subjs + wetlab_subjs, nov)
    mad_subjs = Filter(wear_subjs + wetlab_subjs, ['mad'])
    elan_subjs = Filter(wear_subjs + wetlab_subjs, ['elan'])
    colors1 = sns.color_palette(palette="colorblind", n_colors=len(label_dict_wear) + len(label_dict_wetlab)).as_hex()
    colors1[0] = '#F8F8F8'
    colors1[len(label_dict_wear)] = '#F8F8F8'
    cmap_wear = LinearSegmentedColormap.from_list(name="Wear", colors=colors1[:len(label_dict_wear)], N=len(label_dict_wear))
    cmap_wetlab = LinearSegmentedColormap.from_list(name="Wetlab", colors=colors1[len(label_dict_wear):], N=len(label_dict_wetlab))

    compute_scores(wear, "wear", "groundtruth", wear_subjs, experts=expert_subjs, novices=novice_subjs, mad=mad_subjs, elan=elan_subjs ,label_dict=label_dict_wear, color_map=cmap_wear)
    compute_scores(wetlab, "wetlab", "groundtruth", wetlab_subjs, experts=expert_subjs, novices=novice_subjs, mad=mad_subjs, elan=elan_subjs ,label_dict=label_dict_wetlab, color_map=cmap_wetlab)
