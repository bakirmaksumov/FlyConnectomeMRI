# Phase 9 summary: visual-system graph

Descriptive only: 5 training seeds, one fixed 21-patient test split. No significance or clinical claims. PRIMARY metric: patient-level ROC AUC. The CNN baseline was NOT retrained (frozen Phase 8 reference). EB vs visual is descriptive, not a controlled comparison.

## Run status (Phase 9 runs)

| model | ok |
|---|---|
| graph_visual_drosophila | 5 |
| graph_visual_rewired | 5 |
| graph_visual_er | 5 |
| graph_visual_ws | 5 |
| graph_visual_ba | 5 |

## Mean ± SD over seeds (successful runs)

| Model | runs | patient roc auc | slice roc auc | patient balanced accuracy | slice balanced accuracy | patient sensitivity | patient specificity | patient f1 | slice f1 | patient precision | patient recall | slice precision | slice recall |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| CNN baseline (Phase 8 ref.) | 5 | 0.900 ± 0.009 | 0.806 ± 0.017 | 0.839 ± 0.029 | 0.704 ± 0.018 | 0.754 ± 0.114 | 0.925 ± 0.112 | 0.834 ± 0.049 | 0.737 ± 0.064 | 0.953 ± 0.067 | 0.754 ± 0.114 | 0.868 ± 0.049 | 0.655 ± 0.136 |
| Visual Drosophila | 5 | 0.944 ± 0.029 | 0.922 ± 0.021 | 0.857 ± 0.039 | 0.825 ± 0.044 | 0.738 ± 0.069 | 0.975 ± 0.056 | 0.841 ± 0.047 | 0.818 ± 0.088 | 0.982 ± 0.041 | 0.738 ± 0.069 | 0.962 ± 0.034 | 0.724 ± 0.134 |
| Visual rewired (deg.-pres.) | 5 | 0.927 ± 0.053 | 0.914 ± 0.034 | 0.838 ± 0.157 | 0.806 ± 0.080 | 0.677 ± 0.314 | 1.000 ± 0.000 | 0.765 ± 0.286 | 0.773 ± 0.157 | 1.000 ± 0.000 | 0.677 ± 0.314 | 0.971 ± 0.030 | 0.670 ± 0.219 |
| Visual ER | 5 | 0.931 ± 0.037 | 0.907 ± 0.027 | 0.769 ± 0.119 | 0.745 ± 0.085 | 0.538 ± 0.237 | 1.000 ± 0.000 | 0.675 ± 0.202 | 0.655 ± 0.162 | 1.000 ± 0.000 | 0.538 ± 0.237 | 0.988 ± 0.022 | 0.514 ± 0.216 |
| Visual WS | 5 | 0.960 ± 0.025 | 0.930 ± 0.023 | 0.872 ± 0.074 | 0.817 ± 0.056 | 0.769 ± 0.154 | 0.975 ± 0.056 | 0.855 ± 0.101 | 0.816 ± 0.119 | 0.983 ± 0.037 | 0.769 ± 0.154 | 0.952 ± 0.046 | 0.739 ± 0.193 |
| Visual BA | 5 | 0.927 ± 0.044 | 0.918 ± 0.022 | 0.762 ± 0.137 | 0.768 ± 0.097 | 0.523 ± 0.274 | 1.000 ± 0.000 | 0.649 ± 0.266 | 0.704 ± 0.191 | 1.000 ± 0.000 | 0.523 ± 0.274 | 0.975 ± 0.035 | 0.585 ± 0.240 |

## Individual seeds (Phase 9 runs)

| model | seed | graph_seed | status | best_epoch | val_roc_auc_best | patient_roc_auc | slice_roc_auc | patient_balanced_accuracy | slice_balanced_accuracy | patient_sensitivity | patient_specificity | diag_graph_layer2_dead_nodes_frac | diag_logit_std |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Visual Drosophila | 0 | n/a | ok | 25 | 0.961 | 0.981 | 0.922 | 0.885 | 0.851 | 0.769 | 1.000 | 0.133 | 3.445 |
| Visual Drosophila | 1 | n/a | ok | 19 | 0.969 | 0.904 | 0.892 | 0.885 | 0.849 | 0.769 | 1.000 | 0.141 | 3.753 |
| Visual Drosophila | 2 | n/a | ok | 26 | 0.976 | 0.962 | 0.942 | 0.885 | 0.850 | 0.769 | 1.000 | 0.133 | 4.519 |
| Visual Drosophila | 3 | n/a | ok | 9 | 0.939 | 0.933 | 0.912 | 0.822 | 0.827 | 0.769 | 0.875 | 0.156 | 3.407 |
| Visual Drosophila | 4 | n/a | ok | 19 | 0.983 | 0.942 | 0.940 | 0.808 | 0.748 | 0.615 | 1.000 | 0.109 | 5.033 |
| Visual rewired (deg.-pres.) | 0 | 0.000 | ok | 9 | 0.928 | 0.933 | 0.905 | 0.885 | 0.842 | 0.769 | 1.000 | 0.172 | 3.660 |
| Visual rewired (deg.-pres.) | 1 | 1.000 | ok | 28 | 0.975 | 0.942 | 0.891 | 0.577 | 0.674 | 0.154 | 1.000 | 0.117 | 4.781 |
| Visual rewired (deg.-pres.) | 2 | 2.000 | ok | 7 | 0.944 | 0.904 | 0.896 | 0.885 | 0.810 | 0.769 | 1.000 | 0.133 | 3.495 |
| Visual rewired (deg.-pres.) | 3 | 3.000 | ok | 28 | 0.973 | 0.856 | 0.904 | 0.846 | 0.818 | 0.692 | 1.000 | 0.180 | 5.747 |
| Visual rewired (deg.-pres.) | 4 | 4.000 | ok | 29 | 0.982 | 1.000 | 0.974 | 1.000 | 0.886 | 1.000 | 1.000 | 0.180 | 5.989 |
| Visual ER | 0 | 0.000 | ok | 25 | 0.961 | 0.942 | 0.914 | 0.769 | 0.717 | 0.538 | 1.000 | 0.086 | 3.689 |
| Visual ER | 1 | 1.000 | ok | 28 | 0.961 | 0.942 | 0.912 | 0.846 | 0.754 | 0.692 | 1.000 | 0.031 | 4.829 |
| Visual ER | 2 | 2.000 | ok | 8 | 0.936 | 0.894 | 0.862 | 0.654 | 0.655 | 0.308 | 1.000 | 0.055 | 2.869 |
| Visual ER | 3 | 3.000 | ok | 29 | 0.983 | 0.894 | 0.914 | 0.654 | 0.717 | 0.308 | 1.000 | 0.016 | 5.808 |
| Visual ER | 4 | 4.000 | ok | 21 | 0.983 | 0.981 | 0.932 | 0.923 | 0.884 | 0.846 | 1.000 | 0.078 | 3.556 |
| Visual WS | 0 | 0.000 | ok | 22 | 0.971 | 0.981 | 0.934 | 0.923 | 0.843 | 0.846 | 1.000 | 0.047 | 4.155 |
| Visual WS | 1 | 1.000 | ok | 19 | 0.964 | 0.952 | 0.917 | 0.846 | 0.807 | 0.692 | 1.000 | 0.047 | 2.902 |
| Visual WS | 2 | 2.000 | ok | 8 | 0.944 | 0.933 | 0.903 | 0.769 | 0.731 | 0.538 | 1.000 | 0.055 | 2.449 |
| Visual WS | 3 | 3.000 | ok | 28 | 0.972 | 0.942 | 0.930 | 0.861 | 0.822 | 0.846 | 0.875 | 0.055 | 6.468 |
| Visual WS | 4 | 4.000 | ok | 27 | 0.982 | 0.990 | 0.964 | 0.962 | 0.884 | 0.923 | 1.000 | 0.031 | 4.906 |
| Visual BA | 0 | 0.000 | ok | 28 | 0.978 | 0.904 | 0.935 | 0.846 | 0.805 | 0.692 | 1.000 | 0.062 | 6.319 |
| Visual BA | 1 | 1.000 | ok | 28 | 0.969 | 0.923 | 0.929 | 0.577 | 0.655 | 0.154 | 1.000 | 0.078 | 4.158 |
| Visual BA | 2 | 2.000 | ok | 13 | 0.955 | 0.923 | 0.906 | 0.885 | 0.827 | 0.769 | 1.000 | 0.070 | 5.631 |
| Visual BA | 3 | 3.000 | ok | 24 | 0.989 | 0.885 | 0.884 | 0.654 | 0.679 | 0.308 | 1.000 | 0.086 | 3.941 |
| Visual BA | 4 | 4.000 | ok | 27 | 0.979 | 1.000 | 0.934 | 0.846 | 0.877 | 0.692 | 1.000 | 0.055 | 4.755 |

## Paired differences: Visual Drosophila − control (per seed)

### graph_visual_drosophila - graph_visual_rewired

| seed | patient_roc_auc | slice_roc_auc | patient_balanced_accuracy | slice_balanced_accuracy | patient_sensitivity | patient_specificity | patient_f1 | slice_f1 |
|---|---|---|---|---|---|---|---|---|
| 0 | +0.048 | +0.017 | +0.000 | +0.009 | +0.000 | +0.000 | +0.000 | +0.030 |
| 1 | -0.038 | +0.001 | +0.308 | +0.175 | +0.615 | +0.000 | +0.603 | +0.331 |
| 2 | +0.058 | +0.046 | +0.000 | +0.040 | +0.000 | +0.000 | +0.000 | +0.056 |
| 3 | +0.077 | +0.009 | -0.024 | +0.009 | +0.077 | -0.125 | +0.015 | +0.085 |
| 4 | -0.058 | -0.034 | -0.192 | -0.139 | -0.385 | +0.000 | -0.238 | -0.277 |

### graph_visual_drosophila - graph_visual_er

| seed | patient_roc_auc | slice_roc_auc | patient_balanced_accuracy | slice_balanced_accuracy | patient_sensitivity | patient_specificity | patient_f1 | slice_f1 |
|---|---|---|---|---|---|---|---|---|
| 0 | +0.038 | +0.008 | +0.115 | +0.134 | +0.231 | +0.000 | +0.170 | +0.264 |
| 1 | -0.038 | -0.021 | +0.038 | +0.095 | +0.077 | +0.000 | +0.051 | +0.166 |
| 2 | +0.067 | +0.080 | +0.231 | +0.196 | +0.462 | +0.000 | +0.399 | +0.367 |
| 3 | +0.038 | -0.001 | +0.168 | +0.110 | +0.462 | -0.125 | +0.363 | +0.269 |
| 4 | -0.038 | +0.008 | -0.115 | -0.136 | -0.231 | +0.000 | -0.155 | -0.248 |

### graph_visual_drosophila - graph_visual_ws

| seed | patient_roc_auc | slice_roc_auc | patient_balanced_accuracy | slice_balanced_accuracy | patient_sensitivity | patient_specificity | patient_f1 | slice_f1 |
|---|---|---|---|---|---|---|---|---|
| 0 | +0.000 | -0.012 | -0.038 | +0.008 | -0.077 | +0.000 | -0.047 | -0.012 |
| 1 | -0.048 | -0.025 | +0.038 | +0.042 | +0.077 | +0.000 | +0.051 | +0.086 |
| 2 | +0.029 | +0.039 | +0.115 | +0.119 | +0.231 | +0.000 | +0.170 | +0.207 |
| 3 | -0.010 | -0.017 | -0.038 | +0.005 | -0.077 | +0.000 | -0.047 | -0.011 |
| 4 | -0.048 | -0.024 | -0.154 | -0.136 | -0.308 | +0.000 | -0.198 | -0.260 |

### graph_visual_drosophila - graph_visual_ba

| seed | patient_roc_auc | slice_roc_auc | patient_balanced_accuracy | slice_balanced_accuracy | patient_sensitivity | patient_specificity | patient_f1 | slice_f1 |
|---|---|---|---|---|---|---|---|---|
| 0 | +0.077 | -0.013 | +0.038 | +0.046 | +0.077 | +0.000 | +0.051 | +0.092 |
| 1 | -0.019 | -0.037 | +0.308 | +0.194 | +0.615 | +0.000 | +0.603 | +0.374 |
| 2 | +0.038 | +0.036 | +0.000 | +0.023 | +0.000 | +0.000 | +0.000 | -0.031 |
| 3 | +0.048 | +0.028 | +0.168 | +0.148 | +0.462 | -0.125 | +0.363 | +0.347 |
| 4 | -0.058 | +0.006 | -0.038 | -0.129 | -0.077 | +0.000 | -0.056 | -0.211 |

## Summary of paired differences (descriptive; no p-values)

| comparison | metric | n_seeds | mean_difference | sd_difference | seeds_reference_higher | seeds_reference_lower | seeds_equal |
|---|---|---|---|---|---|---|---|
| graph_visual_drosophila - graph_visual_rewired | patient_roc_auc | 5 | +0.017 | +0.061 | 3 | 2 | 0 |
| graph_visual_drosophila - graph_visual_rewired | slice_roc_auc | 5 | +0.008 | +0.029 | 4 | 1 | 0 |
| graph_visual_drosophila - graph_visual_rewired | patient_balanced_accuracy | 5 | +0.018 | +0.181 | 1 | 2 | 2 |
| graph_visual_drosophila - graph_visual_rewired | slice_balanced_accuracy | 5 | +0.019 | +0.112 | 4 | 1 | 0 |
| graph_visual_drosophila - graph_visual_rewired | patient_sensitivity | 5 | +0.062 | +0.358 | 2 | 1 | 2 |
| graph_visual_drosophila - graph_visual_rewired | patient_specificity | 5 | -0.025 | +0.056 | 0 | 1 | 4 |
| graph_visual_drosophila - graph_visual_rewired | patient_f1 | 5 | +0.076 | +0.313 | 2 | 1 | 2 |
| graph_visual_drosophila - graph_visual_rewired | slice_f1 | 5 | +0.045 | +0.216 | 4 | 1 | 0 |
| graph_visual_drosophila - graph_visual_rewired | patient_precision | 5 | -0.018 | +0.041 | 0 | 1 | 4 |
| graph_visual_drosophila - graph_visual_rewired | patient_recall | 5 | +0.062 | +0.358 | 2 | 1 | 2 |
| graph_visual_drosophila - graph_visual_rewired | slice_precision | 5 | -0.009 | +0.056 | 2 | 3 | 0 |
| graph_visual_drosophila - graph_visual_rewired | slice_recall | 5 | +0.053 | +0.319 | 4 | 1 | 0 |
| graph_visual_drosophila - graph_visual_er | patient_roc_auc | 5 | +0.013 | +0.049 | 3 | 2 | 0 |
| graph_visual_drosophila - graph_visual_er | slice_roc_auc | 5 | +0.015 | +0.038 | 3 | 2 | 0 |
| graph_visual_drosophila - graph_visual_er | patient_balanced_accuracy | 5 | +0.087 | +0.134 | 4 | 1 | 0 |
| graph_visual_drosophila - graph_visual_er | slice_balanced_accuracy | 5 | +0.080 | +0.127 | 4 | 1 | 0 |
| graph_visual_drosophila - graph_visual_er | patient_sensitivity | 5 | +0.200 | +0.291 | 4 | 1 | 0 |
| graph_visual_drosophila - graph_visual_er | patient_specificity | 5 | -0.025 | +0.056 | 0 | 1 | 4 |
| graph_visual_drosophila - graph_visual_er | patient_f1 | 5 | +0.166 | +0.229 | 4 | 1 | 0 |
| graph_visual_drosophila - graph_visual_er | slice_f1 | 5 | +0.163 | +0.241 | 4 | 1 | 0 |
| graph_visual_drosophila - graph_visual_er | patient_precision | 5 | -0.018 | +0.041 | 0 | 1 | 4 |
| graph_visual_drosophila - graph_visual_er | patient_recall | 5 | +0.200 | +0.291 | 4 | 1 | 0 |
| graph_visual_drosophila - graph_visual_er | slice_precision | 5 | -0.026 | +0.051 | 1 | 4 | 0 |
| graph_visual_drosophila - graph_visual_er | slice_recall | 5 | +0.210 | +0.338 | 4 | 1 | 0 |
| graph_visual_drosophila - graph_visual_ws | patient_roc_auc | 5 | -0.015 | +0.033 | 1 | 3 | 1 |
| graph_visual_drosophila - graph_visual_ws | slice_roc_auc | 5 | -0.008 | +0.027 | 1 | 4 | 0 |
| graph_visual_drosophila - graph_visual_ws | patient_balanced_accuracy | 5 | -0.015 | +0.100 | 2 | 3 | 0 |
| graph_visual_drosophila - graph_visual_ws | slice_balanced_accuracy | 5 | +0.008 | +0.093 | 4 | 1 | 0 |
| graph_visual_drosophila - graph_visual_ws | patient_sensitivity | 5 | -0.031 | +0.201 | 2 | 3 | 0 |
| graph_visual_drosophila - graph_visual_ws | patient_specificity | 5 | +0.000 | +0.000 | 0 | 0 | 5 |
| graph_visual_drosophila - graph_visual_ws | patient_f1 | 5 | -0.014 | +0.136 | 2 | 3 | 0 |
| graph_visual_drosophila - graph_visual_ws | slice_f1 | 5 | +0.002 | +0.172 | 2 | 3 | 0 |
| graph_visual_drosophila - graph_visual_ws | patient_precision | 5 | -0.002 | +0.003 | 0 | 1 | 4 |
| graph_visual_drosophila - graph_visual_ws | patient_recall | 5 | -0.031 | +0.201 | 2 | 3 | 0 |
| graph_visual_drosophila - graph_visual_ws | slice_precision | 5 | +0.010 | +0.037 | 3 | 2 | 0 |
| graph_visual_drosophila - graph_visual_ws | slice_recall | 5 | -0.015 | +0.258 | 2 | 3 | 0 |
| graph_visual_drosophila - graph_visual_ba | patient_roc_auc | 5 | +0.017 | +0.055 | 3 | 2 | 0 |
| graph_visual_drosophila - graph_visual_ba | slice_roc_auc | 5 | +0.004 | +0.030 | 3 | 2 | 0 |
| graph_visual_drosophila - graph_visual_ba | patient_balanced_accuracy | 5 | +0.095 | +0.142 | 3 | 1 | 1 |
| graph_visual_drosophila - graph_visual_ba | slice_balanced_accuracy | 5 | +0.056 | +0.125 | 4 | 1 | 0 |
| graph_visual_drosophila - graph_visual_ba | patient_sensitivity | 5 | +0.215 | +0.305 | 3 | 1 | 1 |
| graph_visual_drosophila - graph_visual_ba | patient_specificity | 5 | -0.025 | +0.056 | 0 | 1 | 4 |
| graph_visual_drosophila - graph_visual_ba | patient_f1 | 5 | +0.192 | +0.281 | 3 | 1 | 1 |
| graph_visual_drosophila - graph_visual_ba | slice_f1 | 5 | +0.115 | +0.249 | 3 | 2 | 0 |
| graph_visual_drosophila - graph_visual_ba | patient_precision | 5 | -0.018 | +0.041 | 0 | 1 | 4 |
| graph_visual_drosophila - graph_visual_ba | patient_recall | 5 | +0.215 | +0.305 | 3 | 1 | 1 |
| graph_visual_drosophila - graph_visual_ba | slice_precision | 5 | -0.013 | +0.058 | 2 | 3 | 0 |
| graph_visual_drosophila - graph_visual_ba | slice_recall | 5 | +0.139 | +0.335 | 3 | 2 | 0 |

## EB-128 w20 (Phase 8) vs Visual-128 raw (Phase 9): descriptive only

| family | metric | eb_w20_mean | eb_w20_sd | visual_raw_mean | visual_raw_sd |
|---|---|---|---|---|---|
| Drosophila | patient_roc_auc | 0.927 | 0.044 | 0.944 | 0.029 |
| Drosophila | slice_roc_auc | 0.905 | 0.029 | 0.922 | 0.021 |
| Drosophila | patient_balanced_accuracy | 0.862 | 0.044 | 0.857 | 0.039 |
| Drosophila | slice_balanced_accuracy | 0.807 | 0.053 | 0.825 | 0.044 |
| Rewired | patient_roc_auc | 0.942 | 0.041 | 0.927 | 0.053 |
| Rewired | slice_roc_auc | 0.913 | 0.019 | 0.914 | 0.034 |
| Rewired | patient_balanced_accuracy | 0.860 | 0.074 | 0.838 | 0.157 |
| Rewired | slice_balanced_accuracy | 0.797 | 0.036 | 0.806 | 0.080 |
| ER | patient_roc_auc | 0.960 | 0.018 | 0.931 | 0.037 |
| ER | slice_roc_auc | 0.930 | 0.011 | 0.907 | 0.027 |
| ER | patient_balanced_accuracy | 0.853 | 0.031 | 0.769 | 0.119 |
| ER | slice_balanced_accuracy | 0.826 | 0.027 | 0.745 | 0.085 |
| WS | patient_roc_auc | 0.933 | 0.039 | 0.960 | 0.025 |
| WS | slice_roc_auc | 0.911 | 0.023 | 0.930 | 0.023 |
| WS | patient_balanced_accuracy | 0.785 | 0.169 | 0.872 | 0.074 |
| WS | slice_balanced_accuracy | 0.745 | 0.144 | 0.817 | 0.056 |
| BA | patient_roc_auc | 0.963 | 0.036 | 0.927 | 0.044 |
| BA | slice_roc_auc | 0.937 | 0.036 | 0.918 | 0.022 |
| BA | patient_balanced_accuracy | 0.838 | 0.177 | 0.762 | 0.137 |
| BA | slice_balanced_accuracy | 0.830 | 0.110 | 0.768 | 0.097 |
