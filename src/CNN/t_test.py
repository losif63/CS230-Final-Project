'''
Created on Dec 3, 2025

@author: Sebastian Prepelita
'''
import numpy as np

cnn_1d_euclidian = [0.0810 , 0.0838 , 0.0737 , 0.1268 , 0.0716 , 0.0773 , 0.0737 , 0.0670 , 1.1646 , 0.0963 , 0.1084 , 0.0727 , 0.0654 , 0.0762 , 0.1471 , 0.0845 , 0.1194 , 0.0762 , 0.1136 , 0.1147 , 0.1210 , 0.0830 , 0.0955 , 0.1345]
cnn_1d_rotation = [14.21, 12.98, 14.25, 13.14, 15.08, 14.8 , 13.67, 15.11, 23.55, 19.42, 14.58, 13.2 , 12.56, 13.6 , 16.24, 13.43, 14.6 , 16.19, 16.23, 14.45, 16.65, 13.72, 13.65, 14.35]

cnn_2d_gd_euclidian = [0.0810 , 0.0838 , 0.0737 , 0.1268 , 0.0716 , 0.0773 , 0.0737 , 0.0670 , 1.1646 , 0.0963 , 0.1084 , 0.0727 , 0.0654 , 0.0762 , 0.1471 , 0.0845 , 0.1194 , 0.0762 , 0.1136 , 0.1147 , 0.1210 , 0.0830 , 0.0955 , 0.1345]
cnn_2d_gd_rotation = [16.95, 17.44, 16.09, 15.95, 15.82, 16.29, 19.38, 17.98, 16.22, 18.35, 16.89, 15.83, 15.88, 16.6 , 18.45, 15.73, 17.73, 16.29, 16.62, 17.7 , 16.98, 16.08, 17.49, 15.5 , 16.61]

cnn_2d_spectogram_euclidian = [0.099543, 0.108552, 0.151632, 0.140605, 0.156364, 0.115037, 0.09623 , 0.12425 , 0.113289, 0.136211, 0.125124, 0.146132, 0.121552, 0.106085, 0.162635, 0.104048, 0.101534, 0.117341, 0.119108, 0.063995, 0.10636 , 0.120081, 0.139968, 0.097076]
cnn_2d_spectogram_rotation = [14.34, 19.62, 16.33, 16.42, 18.00, 17.48, 15.76, 16.58, 15.64, 21.22, 15.85, 17.6 , 14.75, 20.79, 19.94, 16.25, 16.22, 19.69, 14.51, 16.2 , 29.07, 18.98, 25.18, 21.05]

from scipy import stats

if __name__ == '__main__':
    print("T-test started")
    alpha = 0.05
    alpha_corected = alpha/6
    print(f"alpha_corected = {alpha_corected}")
    print("="*60)
    print("= Position")
    print("="*60)
    print(f"average 1D: {np.round(np.average(cnn_1d_euclidian),2)} ({np.round(np.std(cnn_1d_euclidian,ddof=1),2)})")
    print(f"average 2D GD: {np.round(np.average(cnn_2d_gd_euclidian),2)} ({np.round(np.std(cnn_2d_gd_euclidian,ddof=1),2)})")
    print(f"average 2D spectogram: {np.round(np.average(cnn_2d_spectogram_euclidian),2)} ({np.round(np.std(cnn_2d_spectogram_euclidian,ddof=1),2)})")
    # Perform the independent two-sample t-test (equal_var=True by default)
    t_statistic, p_value = stats.ttest_ind(cnn_1d_euclidian, cnn_2d_gd_euclidian)
    print(f"\t1D vs 2D GD, p-value: {np.round(p_value,4)}")
    t_statistic, p_value = stats.ttest_ind(cnn_1d_euclidian, cnn_2d_spectogram_euclidian, equal_var=False)
    print(f"\t1D vs 2D spectogram, p-value: {np.round(p_value,4)}")
    t_statistic, p_value = stats.ttest_ind(cnn_2d_gd_euclidian, cnn_2d_spectogram_euclidian, equal_var=False)
    print(f"\t2D gd vs 2D spectogram, p-value: {np.round(p_value,4)}")
    print("="*60)
    
    print("="*60)
    print("= Rotation")
    print("="*60)
    print(f"average 1D: {np.round(np.average(cnn_1d_rotation),2)} ({np.round(np.std(cnn_1d_rotation,ddof=1),2)})")
    print(f"average 2D GD: {np.round(np.average(cnn_2d_gd_rotation),2)} ({np.round(np.std(cnn_2d_gd_rotation,ddof=1),2)})")
    print(f"average 2D spectogram: {np.round(np.average(cnn_2d_spectogram_rotation),2)} ({np.round(np.std(cnn_2d_spectogram_rotation,ddof=1),2)})")
    # Perform the independent two-sample t-test (equal_var=True by default)
    t_statistic, p_value = stats.ttest_ind(cnn_1d_rotation, cnn_2d_gd_rotation, equal_var=False)
    print(f"\t1D vs 2D GD, p-value: {np.round(p_value,4)}")
    t_statistic, p_value = stats.ttest_ind(cnn_1d_rotation, cnn_2d_spectogram_rotation, equal_var=False)
    print(f"\t1D vs 2D spectogram, p-value: {np.round(p_value,4)}")
    t_statistic, p_value = stats.ttest_ind(cnn_2d_gd_rotation, cnn_2d_spectogram_rotation, equal_var=False)
    print(f"\t2D gd vs 2D spectogram, p-value: {np.round(p_value,4)}")
    print("="*60)
    
    print("T-test ended")