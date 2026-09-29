# Phase 8 summary

Descriptive only: 5 training seeds, one fixed test split of 21 patients. No significance claims, no clinical claims. CNN baseline vs graph models is not a pure topology comparison (41,921 vs 48,527 parameters).

## Run status

| model | ok |
|---|---|
| cnn_baseline | 5 |
| graph_drosophila | 5 |
| graph_rewired | 5 |
| graph_er | 5 |
| graph_ws | 5 |
| graph_ba | 5 |

## Mean ± SD over seeds (successful runs)

| Model | runs | slice roc auc | patient roc auc | slice balanced accuracy | patient balanced accuracy | slice sensitivity | patient sensitivity | slice specificity | patient specificity | slice f1 | patient f1 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| CNN baseline | 5 | 0.806 ± 0.017 | 0.900 ± 0.009 | 0.704 ± 0.018 | 0.839 ± 0.029 | 0.655 ± 0.136 | 0.754 ± 0.114 | 0.752 ± 0.164 | 0.925 ± 0.112 | 0.737 ± 0.064 | 0.834 ± 0.049 |
| Drosophila w20 | 5 | 0.905 ± 0.029 | 0.927 ± 0.044 | 0.807 ± 0.053 | 0.862 ± 0.044 | 0.631 ± 0.111 | 0.723 ± 0.088 | 0.983 ± 0.012 | 1.000 ± 0.000 | 0.766 ± 0.089 | 0.837 ± 0.059 |
| Rewired (deg.-pres.) | 5 | 0.913 ± 0.019 | 0.942 ± 0.041 | 0.797 ± 0.036 | 0.860 ± 0.074 | 0.717 ± 0.178 | 0.769 ± 0.196 | 0.876 ± 0.176 | 0.950 ± 0.112 | 0.800 ± 0.090 | 0.844 ± 0.110 |
| ER | 5 | 0.930 ± 0.011 | 0.960 ± 0.018 | 0.826 ± 0.027 | 0.853 ± 0.031 | 0.822 ± 0.111 | 0.831 ± 0.126 | 0.830 ± 0.157 | 0.875 ± 0.177 | 0.864 ± 0.036 | 0.869 ± 0.031 |
| WS | 5 | 0.911 ± 0.023 | 0.933 ± 0.039 | 0.745 ± 0.144 | 0.785 ± 0.169 | 0.533 ± 0.320 | 0.569 ± 0.338 | 0.957 ± 0.068 | 1.000 ± 0.000 | 0.783 ± 0.091 | 0.826 ± 0.093 |
| BA | 5 | 0.937 ± 0.036 | 0.963 ± 0.036 | 0.830 ± 0.110 | 0.838 ± 0.177 | 0.736 ± 0.254 | 0.677 ± 0.354 | 0.924 ± 0.044 | 1.000 ± 0.000 | 0.808 ± 0.199 | 0.749 ± 0.346 |

## Individual seeds

| model | seed | graph_seed | status | best_epoch | val_roc_auc_best | slice_roc_auc | patient_roc_auc | slice_balanced_accuracy | patient_balanced_accuracy | slice_sensitivity | slice_specificity | patient_sensitivity | patient_specificity |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| CNN baseline | 0 | n/a | ok | 15 | 0.885 | 0.806 | 0.913 | 0.727 | 0.846 | 0.595 | 0.859 | 0.692 | 1.000 |
| CNN baseline | 1 | n/a | ok | 30 | 0.895 | 0.794 | 0.894 | 0.700 | 0.808 | 0.510 | 0.891 | 0.615 | 1.000 |
| CNN baseline | 2 | n/a | ok | 23 | 0.885 | 0.786 | 0.904 | 0.701 | 0.822 | 0.662 | 0.739 | 0.769 | 0.875 |
| CNN baseline | 3 | n/a | ok | 26 | 0.894 | 0.820 | 0.894 | 0.713 | 0.885 | 0.633 | 0.793 | 0.769 | 1.000 |
| CNN baseline | 4 | n/a | ok | 20 | 0.897 | 0.824 | 0.894 | 0.677 | 0.837 | 0.876 | 0.478 | 0.923 | 0.750 |
| Drosophila w20 | 0 | n/a | ok | 7 | 0.915 | 0.886 | 0.894 | 0.716 | 0.808 | 0.443 | 0.989 | 0.615 | 1.000 |
| Drosophila w20 | 1 | n/a | ok | 23 | 0.958 | 0.902 | 0.894 | 0.827 | 0.885 | 0.676 | 0.978 | 0.769 | 1.000 |
| Drosophila w20 | 2 | n/a | ok | 9 | 0.949 | 0.875 | 0.933 | 0.822 | 0.846 | 0.676 | 0.967 | 0.692 | 1.000 |
| Drosophila w20 | 3 | n/a | ok | 15 | 0.965 | 0.914 | 0.913 | 0.817 | 0.846 | 0.633 | 1.000 | 0.692 | 1.000 |
| Drosophila w20 | 4 | n/a | ok | 29 | 0.988 | 0.950 | 1.000 | 0.853 | 0.923 | 0.729 | 0.978 | 0.846 | 1.000 |
| Rewired (deg.-pres.) | 0 | 0.000 | ok | 28 | 0.963 | 0.930 | 0.971 | 0.833 | 0.962 | 0.862 | 0.804 | 0.923 | 1.000 |
| Rewired (deg.-pres.) | 1 | 1.000 | ok | 10 | 0.951 | 0.895 | 0.923 | 0.834 | 0.885 | 0.690 | 0.978 | 0.769 | 1.000 |
| Rewired (deg.-pres.) | 2 | 2.000 | ok | 8 | 0.952 | 0.890 | 0.942 | 0.790 | 0.808 | 0.581 | 1.000 | 0.615 | 1.000 |
| Rewired (deg.-pres.) | 3 | 3.000 | ok | 14 | 0.959 | 0.915 | 0.885 | 0.760 | 0.769 | 0.519 | 1.000 | 0.538 | 1.000 |
| Rewired (deg.-pres.) | 4 | 4.000 | ok | 9 | 0.958 | 0.932 | 0.990 | 0.766 | 0.875 | 0.933 | 0.598 | 1.000 | 0.750 |
| ER | 0 | 0.000 | ok | 21 | 0.969 | 0.928 | 0.971 | 0.786 | 0.812 | 0.952 | 0.620 | 1.000 | 0.625 |
| ER | 1 | 1.000 | ok | 20 | 0.956 | 0.942 | 0.952 | 0.815 | 0.837 | 0.924 | 0.707 | 0.923 | 0.750 |
| ER | 2 | 2.000 | ok | 26 | 0.975 | 0.939 | 0.962 | 0.854 | 0.885 | 0.795 | 0.913 | 0.769 | 1.000 |
| ER | 3 | 3.000 | ok | 14 | 0.964 | 0.927 | 0.933 | 0.833 | 0.885 | 0.710 | 0.957 | 0.769 | 1.000 |
| ER | 4 | 4.000 | ok | 15 | 0.958 | 0.915 | 0.981 | 0.843 | 0.846 | 0.729 | 0.957 | 0.692 | 1.000 |
| WS | 0 | 0.000 | ok | 22 | 0.960 | 0.890 | 0.923 | 0.731 | 0.769 | 0.495 | 0.967 | 0.538 | 1.000 |
| WS | 1 | 1.000 | ok | 7 | 0.942 | 0.894 | 0.904 | 0.500 | 0.500 | 0.000 | 1.000 | 0.000 | 1.000 |
| WS | 2 | 2.000 | ok | 30 | 0.991 | 0.948 | 0.990 | 0.821 | 0.846 | 0.643 | 1.000 | 0.692 | 1.000 |
| WS | 3 | 3.000 | ok | 24 | 0.983 | 0.919 | 0.894 | 0.846 | 0.885 | 0.714 | 0.978 | 0.769 | 1.000 |
| WS | 4 | 4.000 | ok | 12 | 0.965 | 0.904 | 0.952 | 0.826 | 0.923 | 0.814 | 0.837 | 0.846 | 1.000 |
| BA | 0 | 0.000 | ok | 28 | 0.969 | 0.952 | 0.981 | 0.888 | 0.923 | 0.852 | 0.924 | 0.846 | 1.000 |
| BA | 1 | 1.000 | ok | 26 | 0.961 | 0.878 | 0.904 | 0.639 | 0.538 | 0.300 | 0.978 | 0.077 | 1.000 |
| BA | 2 | 2.000 | ok | 30 | 0.988 | 0.932 | 0.962 | 0.837 | 0.846 | 0.762 | 0.913 | 0.692 | 1.000 |
| BA | 3 | 3.000 | ok | 14 | 0.963 | 0.953 | 0.971 | 0.880 | 0.885 | 0.814 | 0.946 | 0.769 | 1.000 |
| BA | 4 | 4.000 | ok | 29 | 0.969 | 0.972 | 1.000 | 0.906 | 1.000 | 0.952 | 0.859 | 1.000 | 1.000 |

## Paired differences: Drosophila w20 − null (per seed)

### graph_drosophila - graph_rewired

| seed | slice_roc_auc | patient_roc_auc | slice_balanced_accuracy | patient_balanced_accuracy | slice_sensitivity | patient_sensitivity | slice_specificity | patient_specificity | slice_f1 | patient_f1 |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | -0.044 | -0.077 | -0.117 | -0.154 | -0.419 | -0.308 | +0.185 | +0.000 | -0.273 | -0.198 |
| 1 | +0.006 | -0.029 | -0.007 | +0.000 | -0.014 | +0.000 | +0.000 | +0.000 | -0.010 | +0.000 |
| 2 | -0.016 | -0.010 | +0.031 | +0.038 | +0.095 | +0.077 | -0.033 | +0.000 | +0.065 | +0.056 |
| 3 | -0.001 | +0.029 | +0.057 | +0.077 | +0.114 | +0.154 | +0.000 | +0.000 | +0.092 | +0.118 |
| 4 | +0.018 | +0.010 | +0.088 | +0.048 | -0.205 | -0.154 | +0.380 | +0.250 | -0.047 | -0.012 |

### graph_drosophila - graph_er

| seed | slice_roc_auc | patient_roc_auc | slice_balanced_accuracy | patient_balanced_accuracy | slice_sensitivity | patient_sensitivity | slice_specificity | patient_specificity | slice_f1 | patient_f1 |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | -0.043 | -0.077 | -0.070 | -0.005 | -0.510 | -0.385 | +0.370 | +0.375 | -0.287 | -0.135 |
| 1 | -0.040 | -0.058 | +0.012 | +0.048 | -0.248 | -0.154 | +0.272 | +0.250 | -0.098 | -0.019 |
| 2 | -0.065 | -0.029 | -0.032 | -0.038 | -0.119 | -0.077 | +0.054 | +0.000 | -0.068 | -0.051 |
| 3 | -0.013 | -0.019 | -0.016 | -0.038 | -0.076 | -0.077 | +0.043 | +0.000 | -0.045 | -0.051 |
| 4 | +0.035 | +0.019 | +0.011 | +0.077 | +0.000 | +0.154 | +0.022 | +0.000 | +0.005 | +0.098 |

### graph_drosophila - graph_ws

| seed | slice_roc_auc | patient_roc_auc | slice_balanced_accuracy | patient_balanced_accuracy | slice_sensitivity | patient_sensitivity | slice_specificity | patient_specificity | slice_f1 | patient_f1 |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | -0.005 | -0.029 | -0.015 | +0.038 | -0.052 | +0.077 | +0.022 | +0.000 | -0.044 | +0.062 |
| 1 | +0.008 | -0.010 | +0.327 | +0.385 | +0.676 | +0.769 | -0.022 | +0.000 | n/a | n/a |
| 2 | -0.073 | -0.058 | +0.000 | +0.000 | +0.033 | +0.000 | -0.033 | +0.000 | +0.017 | +0.000 |
| 3 | -0.005 | +0.019 | -0.030 | -0.038 | -0.081 | -0.077 | +0.022 | +0.000 | -0.053 | -0.051 |
| 4 | +0.046 | +0.048 | +0.028 | +0.000 | -0.086 | +0.000 | +0.141 | +0.000 | -0.025 | +0.000 |

### graph_drosophila - graph_ba

| seed | slice_roc_auc | patient_roc_auc | slice_balanced_accuracy | patient_balanced_accuracy | slice_sensitivity | patient_sensitivity | slice_specificity | patient_specificity | slice_f1 | patient_f1 |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | -0.066 | -0.087 | -0.172 | -0.115 | -0.410 | -0.231 | +0.065 | +0.000 | -0.292 | -0.155 |
| 1 | +0.023 | -0.010 | +0.188 | +0.346 | +0.376 | +0.692 | +0.000 | +0.000 | +0.344 | +0.727 |
| 2 | -0.058 | -0.029 | -0.016 | +0.000 | -0.086 | +0.000 | +0.054 | +0.000 | -0.047 | +0.000 |
| 3 | -0.039 | -0.058 | -0.063 | -0.038 | -0.181 | -0.077 | +0.054 | +0.000 | -0.111 | -0.051 |
| 4 | -0.022 | +0.000 | -0.052 | -0.077 | -0.224 | -0.154 | +0.120 | +0.000 | -0.107 | -0.083 |

## Summary of paired differences (descriptive; no p-values)

| comparison | metric | n_seeds | mean_difference | sd_difference | seeds_reference_higher | seeds_reference_lower | seeds_equal |
|---|---|---|---|---|---|---|---|
| graph_drosophila - graph_rewired | slice_roc_auc | 5 | -0.007 | +0.024 | 2 | 3 | 0 |
| graph_drosophila - graph_rewired | patient_roc_auc | 5 | -0.015 | +0.041 | 2 | 3 | 0 |
| graph_drosophila - graph_rewired | slice_balanced_accuracy | 5 | +0.010 | +0.079 | 3 | 2 | 0 |
| graph_drosophila - graph_rewired | patient_balanced_accuracy | 5 | +0.002 | +0.091 | 3 | 1 | 1 |
| graph_drosophila - graph_rewired | slice_sensitivity | 5 | -0.086 | +0.225 | 2 | 3 | 0 |
| graph_drosophila - graph_rewired | patient_sensitivity | 5 | -0.046 | +0.185 | 2 | 2 | 1 |
| graph_drosophila - graph_rewired | slice_specificity | 5 | +0.107 | +0.176 | 2 | 1 | 2 |
| graph_drosophila - graph_rewired | patient_specificity | 5 | +0.050 | +0.112 | 1 | 0 | 4 |
| graph_drosophila - graph_rewired | slice_f1 | 5 | -0.035 | +0.145 | 2 | 3 | 0 |
| graph_drosophila - graph_rewired | patient_f1 | 5 | -0.007 | +0.119 | 2 | 2 | 1 |
| graph_drosophila - graph_er | slice_roc_auc | 5 | -0.025 | +0.038 | 1 | 4 | 0 |
| graph_drosophila - graph_er | patient_roc_auc | 5 | -0.033 | +0.037 | 1 | 4 | 0 |
| graph_drosophila - graph_er | slice_balanced_accuracy | 5 | -0.019 | +0.034 | 2 | 3 | 0 |
| graph_drosophila - graph_er | patient_balanced_accuracy | 5 | +0.009 | +0.052 | 2 | 3 | 0 |
| graph_drosophila - graph_er | slice_sensitivity | 5 | -0.190 | +0.200 | 0 | 4 | 1 |
| graph_drosophila - graph_er | patient_sensitivity | 5 | -0.108 | +0.193 | 1 | 4 | 0 |
| graph_drosophila - graph_er | slice_specificity | 5 | +0.152 | +0.158 | 5 | 0 | 0 |
| graph_drosophila - graph_er | patient_specificity | 5 | +0.125 | +0.177 | 2 | 0 | 3 |
| graph_drosophila - graph_er | slice_f1 | 5 | -0.099 | +0.112 | 1 | 4 | 0 |
| graph_drosophila - graph_er | patient_f1 | 5 | -0.032 | +0.084 | 1 | 4 | 0 |
| graph_drosophila - graph_ws | slice_roc_auc | 5 | -0.006 | +0.043 | 2 | 3 | 0 |
| graph_drosophila - graph_ws | patient_roc_auc | 5 | -0.006 | +0.041 | 2 | 3 | 0 |
| graph_drosophila - graph_ws | slice_balanced_accuracy | 5 | +0.062 | +0.150 | 3 | 2 | 0 |
| graph_drosophila - graph_ws | patient_balanced_accuracy | 5 | +0.077 | +0.174 | 2 | 1 | 2 |
| graph_drosophila - graph_ws | slice_sensitivity | 5 | +0.098 | +0.327 | 2 | 3 | 0 |
| graph_drosophila - graph_ws | patient_sensitivity | 5 | +0.154 | +0.348 | 2 | 1 | 2 |
| graph_drosophila - graph_ws | slice_specificity | 5 | +0.026 | +0.069 | 3 | 2 | 0 |
| graph_drosophila - graph_ws | patient_specificity | 5 | +0.000 | +0.000 | 0 | 0 | 5 |
| graph_drosophila - graph_ws | slice_f1 | 4 | -0.026 | +0.031 | 1 | 3 | 0 |
| graph_drosophila - graph_ws | patient_f1 | 4 | +0.003 | +0.046 | 1 | 1 | 2 |
| graph_drosophila - graph_ba | slice_roc_auc | 5 | -0.032 | +0.036 | 1 | 4 | 0 |
| graph_drosophila - graph_ba | patient_roc_auc | 5 | -0.037 | +0.036 | 0 | 4 | 1 |
| graph_drosophila - graph_ba | slice_balanced_accuracy | 5 | -0.023 | +0.132 | 1 | 4 | 0 |
| graph_drosophila - graph_ba | patient_balanced_accuracy | 5 | +0.023 | +0.186 | 1 | 3 | 1 |
| graph_drosophila - graph_ba | slice_sensitivity | 5 | -0.105 | +0.293 | 1 | 4 | 0 |
| graph_drosophila - graph_ba | patient_sensitivity | 5 | +0.046 | +0.371 | 1 | 3 | 1 |
| graph_drosophila - graph_ba | slice_specificity | 5 | +0.059 | +0.043 | 4 | 0 | 1 |
| graph_drosophila - graph_ba | patient_specificity | 5 | +0.000 | +0.000 | 0 | 0 | 5 |
| graph_drosophila - graph_ba | slice_f1 | 5 | -0.042 | +0.235 | 1 | 4 | 0 |
| graph_drosophila - graph_ba | patient_f1 | 5 | +0.087 | +0.362 | 1 | 3 | 1 |
