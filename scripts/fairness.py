import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

def demographic_parity(y_true, y_pred, sensitive):
    # sensitive: array of gender values (e.g., 'Male', 'Female')
    mask_female = (sensitive == 'Female')
    mask_male = (sensitive == 'Male')
    p_female = np.mean(y_pred[mask_female] == 1)
    p_male = np.mean(y_pred[mask_male] == 1)
    return p_female, p_male, abs(p_female - p_male)

def equal_opportunity(y_true, y_pred, sensitive):
    # True positive rate for each group
    mask_female = (sensitive == 'Female')
    mask_male = (sensitive == 'Male')
    tpr_female = np.sum((y_pred == 1) & (y_true == 1) & mask_female) / max(np.sum((y_true == 1) & mask_female), 1)
    tpr_male = np.sum((y_pred == 1) & (y_true == 1) & mask_male) / max(np.sum((y_true == 1) & mask_male), 1)
    return tpr_female, tpr_male, abs(tpr_female - tpr_male)
