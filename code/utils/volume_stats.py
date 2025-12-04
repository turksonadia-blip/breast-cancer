"""
Contains various functions for computing statistics over 3D volumes
"""
import numpy as np
import copy 

# from sklearn.metrics import jaccard_score

def Dice3d(a, b):
    """
    This will compute the Dice Similarity coefficient for two 3-dimensional volumes
    Volumes are expected to be of the same size. We are expecting binary masks -
    0's are treated as background and anything else is counted as data

    Arguments:
        a {Numpy array} -- 3D array with first volume
        b {Numpy array} -- 3D array with second volume

    Returns:
        float
    """
    
    # a - prediction
    # b - ground truth
    
    
    if len(a.shape) != 3 or len(b.shape) != 3:
        raise Exception(f"Expecting 3 dimensional inputs, got {a.shape} and {b.shape}")

    if a.shape != b.shape:
        raise Exception(f"Expecting inputs of the same shape, got {a.shape} and {b.shape}")

    # "labels": { 
    #   "0": "background", 
    #   "1": "Anterior", 
    #   "2": "Posterior"
    # }, 

    # dice_l1+l2 anterior+posterior
    pred, gt = copy.deepcopy(a), copy.deepcopy(b)
    pred[pred!=0]=1; gt[gt!=0]=1
    intersection = np.sum(pred*gt) # np.sum(a[a==b]) 
    volumes = np.sum(pred) + np.sum(gt)
    if volumes == 0:
        return -1
    dice_l1l2 = 2.*float(intersection) / float(volumes)

    # dice_l1 anterior
    pred, gt = copy.deepcopy(a), copy.deepcopy(b)
    pred[pred==2]=0; gt[gt==2]=0
    intersection = np.sum(pred*gt) # np.sum(a[a==b]) 
    volumes = np.sum(pred) + np.sum(gt)
    if volumes == 0:
        return -1
    dice_l1 = 2.*float(intersection) / float(volumes)
    
    # dice_l2 posterior
    pred, gt = copy.deepcopy(a), copy.deepcopy(b)
    pred[pred==1]=0; gt[gt==1]=0
    pred[pred==2]=1; gt[gt==2]=1
    intersection = np.sum(pred*gt) # np.sum(a[a==b]) 
    volumes = np.sum(pred) + np.sum(gt)
    if volumes == 0:
        return -1
    dice_l2 = 2.*float(intersection) / float(volumes)
    
    # dice_l0 background
    pred, gt = copy.deepcopy(a), copy.deepcopy(b)
    pred[pred!=0]=1; gt[gt!=0]=1

    # swap 0 and 1 for a and b for dice background
    where_0 = np.where(pred == 0)
    where_1 = np.where(pred == 1)
    pred[where_0] = 1
    pred[where_1] = 0
    where_0 = np.where(gt == 0)
    where_1 = np.where(gt == 1)
    gt[where_0] = 1
    gt[where_1] = 0

    intersection = np.sum(pred*gt) # np.sum(a[a==b]) 
    volumes = np.sum(pred) + np.sum(gt)
    if volumes == 0:
        return -1
    dice_b = 2.*float(intersection) / float(volumes)
 
    return dice_l1l2, dice_l1, dice_l2, dice_b

def Sensitivity(a,b):
    # a - prediction
    # b - ground truth
    # Sensitivity is to evaluate how accurate the positive cases are detected/predicted
    
    if len(a.shape) != 3 or len(b.shape) != 3:
        raise Exception(f"Expecting 3 dimensional inputs, got {a.shape} and {b.shape}")
    if a.shape != b.shape:
        raise Exception(f"Expecting inputs of the same shape, got {a.shape} and {b.shape}")
  
    # Sens = TP/(TP+FN)
    # TP+TN: total number of true positive cases

    # sens_all
    pred, gt = copy.deepcopy(a), copy.deepcopy(b)
    pred[pred!=0]=1; gt[gt!=0]=1
    tp = np.sum(gt[gt==pred])
    fn = np.sum(gt[gt!=pred])
    if fn+tp == 0:
        return -1
    sens_all = (tp)/(fn+tp)

    # sens_anterior
    pred, gt = copy.deepcopy(a), copy.deepcopy(b)
    pred[pred==2]=0; gt[gt==2]=0
    tp = np.sum(gt[gt==pred])
    fn = np.sum(gt[gt!=pred])
    if fn+tp == 0:
        return -1
    sens_anterior = (tp)/(fn+tp)

    # sens_posterior
    pred, gt = copy.deepcopy(a), copy.deepcopy(b)
    pred[pred==1]=0; gt[gt==1]=0
    pred[pred==2]=1; gt[gt==2]=1
    tp = np.sum(gt[gt==pred])
    fn = np.sum(gt[gt!=pred])
    if fn+tp == 0:
        return -1
    sens_posterior = (tp)/(fn+tp)

    return sens_all, sens_anterior, sens_posterior

def Specificity(a,b):
    # a - prediction
    # b - ground truth
    # Specificity is to evaluate how accurate the negative cases are detected/predicted
    
    if len(a.shape) != 3 or len(b.shape) != 3:
        raise Exception(f"Expecting 3 dimensional inputs, got {a.shape} and {b.shape}")
    if a.shape != b.shape:
        raise Exception(f"Expecting inputs of the same shape, got {a.shape} and {b.shape}")
  
    # Spec = TN/(TN+FP)
    # TN+FP: total number of true negative cases
    
    # spec_all
    pred, gt = copy.deepcopy(a), copy.deepcopy(b)
    pred[pred!=0]=1; gt[gt!=0]=1

    # swap 0 and 1 for a and b for specificity calculation
    where_0 = np.where(pred == 0)
    where_1 = np.where(pred == 1)
    pred[where_0] = 1
    pred[where_1] = 0
    where_0 = np.where(gt == 0)
    where_1 = np.where(gt == 1)
    gt[where_0] = 1
    gt[where_1] = 0
    tn = np.sum(gt[gt==pred])
    fp = np.sum(gt[gt!=pred])
    if tn+fp == 0:
        return -1
    spec_all = (tn)/(tn+fp)

    # spec_anterior
    pred, gt = copy.deepcopy(a), copy.deepcopy(b)
    pred[pred==2]=0; gt[gt==2]=0
    
    # swap 0 and 1 for a and b for specificity calculation
    where_0 = np.where(pred == 0)
    where_1 = np.where(pred == 1)
    pred[where_0] = 1
    pred[where_1] = 0
    where_0 = np.where(gt == 0)
    where_1 = np.where(gt == 1)
    gt[where_0] = 1
    gt[where_1] = 0
    tn = np.sum(gt[gt==pred])
    fp = np.sum(gt[gt!=pred])
    if tn+fp == 0:
        return -1
    spec_anterior = (tn)/(tn+fp)

    # spec_posterior
    pred, gt = copy.deepcopy(a), copy.deepcopy(b)
    pred[pred==1]=0; gt[gt==1]=0
    pred[pred==2]=1; gt[gt==2]=1
    
    # swap 0 and 1 for a and b for specificity calculation
    where_0 = np.where(pred == 0)
    where_1 = np.where(pred == 1)
    pred[where_0] = 1
    pred[where_1] = 0
    where_0 = np.where(gt == 0)
    where_1 = np.where(gt == 1)
    gt[where_0] = 1
    gt[where_1] = 0
    tn = np.sum(gt[gt==pred])
    fp = np.sum(gt[gt!=pred])
    if tn+fp == 0:
        return -1
    spec_posterior = (tn)/(tn+fp)
    # all L1+L2, 1+2; anteriror L1, 1; posterior L2, 2; background 0
    return spec_all, spec_anterior, spec_posterior

def Jaccard3d(a, b):
    # a - prediction
    # b - ground truth
    
    """
    This will compute the Jaccard Similarity coefficient for two 3-dimensional volumes
    Volumes are expected to be of the same size. We are expecting binary masks - 
    0's are treated as background and anything else is counted as data

    Arguments:
        a {Numpy array} -- 3D array with first volume
        b {Numpy array} -- 3D array with second volume

    Returns:
        float
    """

    if len(a.shape) != 3 or len(b.shape) != 3:
        raise Exception(f"Expecting 3 dimensional inputs, got {a.shape} and {b.shape}")

    if a.shape != b.shape:
        raise Exception(f"Expecting inputs of the same shape, got {a.shape} and {b.shape}")

    # TASK: Write implementation of Jaccard similarity coefficient. Please do not use 
    # the Dice3D function from above to do the computation ;)
    # <YOUR CODE GOES HERE>
    # Jaccard3d = Dice3d(a,b)/(2-Dice3d(a,b))

    # jaccard_all
    pred, gt = copy.deepcopy(a), copy.deepcopy(b)
    pred[pred!=0]=1;gt[gt!=0]=1

    intersection = np.sum(pred*gt)
    a_plus_b = np.sum(pred) + np.sum(gt)
    union = a_plus_b - intersection
    jaccard_all = (intersection)/(union)

    # jaccard_anterior
    pred, gt = copy.deepcopy(a), copy.deepcopy(b)
    pred[pred==2]=0; gt[gt==2]=0

    intersection = np.sum(pred*gt)
    a_plus_b = np.sum(pred) + np.sum(gt)
    union = a_plus_b - intersection
    jaccard_anterior = (intersection)/(union)

    # jaccard_posterior
    pred, gt = copy.deepcopy(a), copy.deepcopy(b)
    pred[pred==1]=0; gt[gt==1]=0
    pred[pred==2]=1; gt[gt==2]=1

    intersection = np.sum(pred*gt)
    a_plus_b = np.sum(pred) + np.sum(gt)
    union = a_plus_b - intersection
    jaccard_posterior = (intersection)/(union)

    return jaccard_all, jaccard_anterior, jaccard_posterior

def F1_score(a, b):
    # a - prediction
    # b - ground truth
    # F1 score is Dice score
    """
    This will compute the F1 score for two 3-dimensional volumes
    Volumes are expected to be of the same size. We are expecting binary masks - 
    0's are treated as background and anything else is counted as data

    Arguments:
        a {Numpy array} -- 3D array with first volume
        b {Numpy array} -- 3D array with second volume

    Returns:
        float
    """

    if len(a.shape) != 3 or len(b.shape) != 3:
        raise Exception(f"Expecting 3 dimensional inputs, got {a.shape} and {b.shape}")

    if a.shape != b.shape:
        raise Exception(f"Expecting inputs of the same shape, got {a.shape} and {b.shape}")

    # Sens = TP/(TP+FN)
    # TP+TN: total number of true positive cases
    pred, gt = copy.deepcopy(a), copy.deepcopy(b)
    pred[pred!=0]=1; gt[gt!=0]=1
    
    tp = np.sum(gt[gt==pred])
    fn = np.sum(gt[gt!=pred])

    # Spec = TN/(TN+FP)
    # TN+FP: total number of true negative cases   
    # swap 0 and 1 for a and b for specificity calculation
    
    where_0, where_1 = np.where(pred == 0), np.where(pred == 1)
    pred[where_0], pred[where_1] = 1, 0
    where_0, where_1 = np.where(gt == 0), np.where(gt == 1)
    gt[where_0], gt[where_1] = 1, 0
    
    tn = np.sum(gt[gt==pred])
    fp = np.sum(gt[gt!=pred])
    
    # F1 = (2TP)/(2TP+FP+FN)
    
    return (2*tp)/(2*tp+fp+fn)