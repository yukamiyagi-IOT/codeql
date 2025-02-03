# -*- coding: utf-8 -*-
"""
Created on Thur 23 Jun 2022
@author: 1862994
"""

import os
import math
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures


def _get_unit_of_target(target):
    target_unit = {
        "Damper_RL": "mm",
        "Damper_RR": "mm",
        "Damper_3rd_Rr": "mm",
        "Damper_FL": "mm",
        "Damper_FR": "mm",
        "Damper_3rd_Fr": "mm",
        "RH_Rear": "mm",
        "RH_Front": "mm",
        "Car_Speed": "km/h",
    }
    
    return target_unit[target]


def _get_parityplot_threshold_val(target):
    target_threshold_val = {
        'Damper_FL': 0.5, 
        'Damper_FR': 0.5,
        'Damper_3rd_Fr': 1, 
        'Damper_RL': 1.5, 
        'Damper_RR': 1.5, 
        'Damper_3rd_Rr': 1.5,
        'RH_Front': 2, 
        'RH_Rear': 4,
        'Car_Speed': 15,
    }
    
    threshold_val = target_threshold_val[target]
    return threshold_val


def _get_parityplot_text_pos(minx, maxx):
    text_x = (minx + maxx) * 0.45
    text_y_offset = (maxx - minx) * 0.06
        
    return text_x, text_y_offset


def actual_predicted_lineplot(RUN_ID, setting_no, target, modelname, filepath, y_test, y_pred, std_pred=None):
    
    unit = _get_unit_of_target(target)
    
    df = pd.DataFrame(y_test)
    df.rename(columns={target: "actual"}, inplace=True)
    df["predicted"] = y_pred.reshape(-1)
    df["stdev"] = std_pred
    
    new_df = df.iloc[0:len(df):10, :] # interval: 0.1s (10*0.01s)
    new_df = new_df.sort_index()
    
    x_time = new_df.index
    time_ticks = len(x_time) - 1
    plt.figure(figsize = (20, 4))
    _title = "{} - {} ({})".format(modelname, target, unit)
    plt.title(_title, fontsize=16)
    plt.xticks(np.arange(0, len(x_time), time_ticks), x_time[::time_ticks])

    plt.plot(new_df['actual'], label='actual')
    plt.plot(new_df['predicted'], label='predicted')
    plt.legend()

    if modelname.startswith("rbfBRR"):
        plt.fill_between(
            x_time,
            new_df['predicted'] - 1.96 * new_df['stdev'],
            new_df['predicted'] + 1.96 * new_df['stdev'],
            color="tab:orange",
            alpha=0.3,
        label=r"95% confidence interval",
        )
        plt.legend(['actual', 'predicted', "95% confidence interval"])
        
    # save the figure.
    dir_path = "../../Output/framework_of_modelling/{}/setting-no-{}/prediction_line_plots/{}".format(RUN_ID, setting_no, modelname)
    dir_exists = os.path.exists(dir_path)
    if not dir_exists:
        os.makedirs(dir_path)
        
    plot_fn = filepath.split("/")[-1].split(".csv")[0]
    plt.savefig(
        "{}/predlineplot-{}-{}-{}-{}-setting{}.png".format(
            dir_path, target, modelname, plot_fn, RUN_ID, setting_no),
        bbox_inches='tight',
        dpi=150
    )

    plt.close()
    
  
def parity_plot(RUN_ID, setting_no, target, modelname, filepath, actual, predicted):     
    
    diff = _get_parityplot_threshold_val(target)
    unit = _get_unit_of_target(target)

    minx = min(min(actual), min(predicted))
    maxx = max(max(actual), max(predicted))
    
    if (maxx - minx) > 1e4:
        print("{}\n-> {} model of {}: the predicted values are out of reasonable range. The parity plots are thus not plotted.".format(filepath, modelname, target))
        return

    plt.figure(figsize = (6, 6))
    _title = "{} - {} ({})".format(modelname, target, unit)
    plt.title(_title, fontsize=16)
    plt.xlabel("Actual", fontsize=16)
    plt.ylabel("Predicted", fontsize=16)
    plt.xlim(minx, maxx)
    plt.ylim(minx, maxx)
    
    n_points = len(actual)
    n_points_hit = 0
    for i in range(n_points):
        if (predicted[i] >= (actual[i] - diff)) & (predicted[i] <= (actual[i] + diff)):
            n_points_hit += 1
    hit_ratio = n_points_hit / n_points
    
    # middle line
    plt.plot(np.arange(minx, maxx, 0.1), np.arange(minx, maxx, 0.1), 
             color = "red", linewidth=3)
    # upper line
    plt.plot(np.arange(minx, maxx - diff, 0.1), np.arange(minx + diff, maxx, 0.1), 
             color = "green", linestyle='dashed', linewidth=3)
    # lower line
    plt.plot(np.arange(minx + diff, maxx, 0.1), np.arange(minx, maxx - diff, 0.1), 
             color = "green", linestyle='dashed', linewidth=3)
    # plot
    plt.scatter(actual, predicted, color='b', marker='o', alpha=0.2)
    # write hit ratio (pct.)
    text_x, text_y_offset = _get_parityplot_text_pos(minx, maxx)
    plt.text(text_x, minx+2*text_y_offset, 'threshold value: {} {}'.format(diff, unit), fontsize=12)
    plt.text(text_x, minx+text_y_offset, 'hit ratio: {:.2f}%'.format(hit_ratio * 100), fontsize=12)
    
    # save the figure.
    dir_path = "../../Output/framework_of_modelling/{}/setting-no-{}/parity_plots/{}".format(RUN_ID, setting_no, modelname)
    dir_exists = os.path.exists(dir_path)
    if not dir_exists:
        os.makedirs(dir_path)
    
    plot_fn = filepath.split("/")[-1].split(".csv")[0]
    plt.savefig(
        "{}/parityplot-{}-{}-{}-{}-setting{}.png".format(
            dir_path, target, modelname, plot_fn, RUN_ID, setting_no),
        bbox_inches='tight',
        dpi=150
    )
    
    plt.close()
    
    
# The code blocks below relate to sensitivity plots.

def _get_sensitivity_color_of_circuit(circuit):
    circuit_color = {
        'SUZUKA': 'tab:blue', 
        'AP': 'tab:orange', 
        'AutoPolis': 'tab:orange',         
        'MOTEGI': 'tab:green', 
        'SUGO': 'tab:red',
        'FUJI': 'tab:purple', 
        'OKAYAMA': 'tab:brown',
        'NA': 'tab:gray'
    }
    
    return circuit_color[circuit]


def _create_range_on_feat(feat_list, df_input, feat_name):
    samples_min = df_input[feat_name].min()
    samples_max = df_input[feat_name].max() 
    half_line_x_width = (samples_max - samples_min) * 0.18 * 0.5
    
    feat_list[0] = feat_list[0] - half_line_x_width
    feat_list[1] = feat_list[1] - half_line_x_width / 2
    feat_list[2] = feat_list[2]
    feat_list[4] = feat_list[4] + half_line_x_width
    feat_list[3] = feat_list[3] + half_line_x_width / 2
    
    return feat_list


def _get_circuit_name(fp):
    fp_trans = fp.lower()
    fp_trans = fp_trans.replace('lap', '+++')
    if "suzuka" in fp_trans:
        return 'SUZUKA'
    elif 'motegi' in fp_trans:
        return 'MOTEGI'
    elif 'sugo' in fp_trans:
        return 'SUGO'
    elif 'fuji' in fp_trans:
        return 'FUJI'
    elif 'okayama' in fp_trans:
        return 'OKAYAMA'
    elif 'ap' in fp_trans:
        return 'AP'
    elif 'autopolis' in fp_trans:
        return 'AP'    
    else:
        return 'NA'
    

def _create_samples_df_for_sensitivity(test_set, features):
    samples_df = pd.DataFrame()
    num_total_points = 100
    num_test_laps = len(test_set)
    num_sample_each_lap = math.floor(num_total_points / num_test_laps)

    for fp,df in test_set.items():
        df = df[features]
        df = df.dropna()
        if len(df) < num_sample_each_lap:
            num_test_laps -= 1
            num_sample_each_lap = math.floor(num_total_points / num_test_laps)
            continue
        df = df.sample(n = num_sample_each_lap)
        #circuit_name = fp.split("/")[-1].split("#")[1].split("_")[-1]
        circuit_name = _get_circuit_name(fp)
        df['circuit'] = circuit_name
        samples_df = samples_df.append(df)
        
    samples_df.reset_index(drop=True, inplace=True)
    
    return samples_df

# need to be modified later
def _sensitivity_plot(RUN_ID, setting_no, target, modelname, feat_df, feat_name):
    
    fig,ax = plt.subplots(1, figsize=(12,7))

    ax.set_title("Sensitivity Plot - {} vs {}(predicted)\n(random samples from all the laps)".
                 format(feat_name, target), size=18)
    ax.set_xlabel(xlabel=feat_name, size=16)
    ax.set_ylabel(ylabel=target+'(predicted)', size=16)
    
    num_lap = int(len(feat_df)/5)
    unique_labels_lines = {}
    
    for i in range(0, num_lap):      
        dff = feat_df.loc[i]
        _label = dff.loc[i, 'circuit'].unique()[0]           
        _color = _get_sensitivity_color_of_circuit(_label)
        line, = ax.plot(dff[feat_name], dff['pred'],linestyle='-', linewidth=2, markevery=[2],
                        marker='o', alpha=0.8, color=_color)
        
        if _label not in unique_labels_lines:
            unique_labels_lines[_label] = line
    
    ax.legend(list(unique_labels_lines.values()), list(unique_labels_lines.keys()))
    plt.grid()
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    
    dir_path = "../../Output/framework_of_modelling/{}/setting-no-{}/sensitivity_plots/{}".format(RUN_ID, setting_no, feat_name)
    dir_exists = os.path.exists(dir_path)
    if not dir_exists:
        os.makedirs(dir_path)
    filepath_ss = '{}/sensitivity-{}-{}-{}-{}-setting{}.png'.format(
        dir_path, target, feat_name, modelname, RUN_ID, setting_no)

    plt.savefig(filepath_ss, bbox_inches='tight', dpi=150)
    plt.close()
    

def _poly_process_scaled_X(X, _degree=2):
    polynomial_features= PolynomialFeatures(degree=_degree)
    X_std_poly = polynomial_features.fit_transform(X)

    return X_std_poly


def make_sensitivity_plots(RUN_ID, setting_no, test_set, target, features, modelname, pred_model, scaler, rbf_mapper=None):
    points = 5 # one line five points
    samples_df = _create_samples_df_for_sensitivity(test_set, features)
    
    # plot sensitivity for each feature.
    for feat in features:
        feat_df = pd.DataFrame(columns=samples_df.columns)
        
        for r in range(len(samples_df)):
            dff = pd.DataFrame(columns=samples_df.columns)
            for count in range(points):
                dff = dff.append(samples_df.loc[r])

            final_feat_list = _create_range_on_feat(list(dff[feat]), samples_df, feat)
            dff[feat] = final_feat_list  
            feat_df = feat_df.append(dff)

        feat_X_std = scaler.transform(feat_df[features])

        
        if modelname.startswith("Poly2"):
            feat_X_std = _poly_process_scaled_X(feat_X_std, _degree=2)
        elif modelname.startswith("Poly3"):
            feat_X_std = _poly_process_scaled_X(feat_X_std, _degree=3)
        elif modelname.startswith("rbf"):
            feat_X_std = rbf_mapper.transform(feat_X_std)
        
        pred = pred_model.predict(feat_X_std)    
        feat_df['pred'] = list(pred)

        _sensitivity_plot(RUN_ID, setting_no, target, modelname, feat_df, feat)   