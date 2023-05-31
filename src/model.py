import numpy as np
import scipy.io
import scipy.interpolate
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import svm
from sklearn import metrics
 
from sklearn.model_selection import KFold
from sklearn.utils import shuffle
 
 
def show_nthLargestWeights(n, weight_vector):
    weight_magnitudes = abs(weight_vector)
    weight_magnitudes = weight_magnitudes.ravel()
    sorted_weight_magnitudes = np.sort(weight_magnitudes)
    largest_magnitudes = sorted_weight_magnitudes[-n:]
 
    indices = []
    for magnitude in largest_magnitudes:
        index = np.where(weight_magnitudes == magnitude)
        indices.append(index)
    indices = [item for items in indices for item in items]
 
    largest_signed_weights = []
    for index in indices:
        # SIGNED weights
        largest_signed_weights.append(weight_vector[index])
    largest_signed_weights = [item for items in largest_signed_weights for item in items]
 
    # for i in range(0, len(weight_vector)):
    #     print(f"{i}  {weight_vector[i]}")
 
    df = pd.DataFrame({
        'Electrode Index': indices,
        'Signed Weight': largest_signed_weights
    })
 
    fig = plt.figure(figsize=(8, 2))
    ax = fig.add_subplot(111)
 
    ax.table(cellText=df.values,
             rowLabels=df.index,
             colLabels=df.columns,
             loc="center",
             cellLoc="center"
             )
    ax.set_title("Signed Weights of 6 Most Dominant Electrodes (Ascending Order, Overt Movement)")
 
    ax.axis("off")
    plt.show()
 
 
# electrode index on the horizontal axis, weights on the vertical axis
# weight_vector is a 204x1 vector
 
def show_weightsPlot(weight_vector):
    electrode_index = np.arange(start=0, stop=204, step=1)
    plt.stem(electrode_index, weight_vector.ravel())
    plt.title('Electrode Index vs Channel Weight (Overt Movement)')
    plt.xlabel('Electrode Index [0-203]')
    plt.ylabel('Log Scale of Channel Weights')
    plt.yscale('log')
    plt.show()
 
 
# This function shows the location of one channel on head.
#
# INPUTS:
#   chanVal is a vector with the weight of the channels to be plotted.
 
 
def show_chanWeights(chanVal):
    matlab_offset = 1
 
    selNum = np.asarray(range(1, 306))
    cortIX = np.where(np.mod(selNum, 3) != 0)
    selNum = selNum[cortIX]
 
    resolution = 200
 
    # Load sensor location
    # load sensors102.mat
    mat = scipy.io.loadmat('sensors102.mat')
    c102 = mat['c102']
    x = c102[:, 2 - matlab_offset]
    y = c102[:, 3 - matlab_offset]
    xlin = np.linspace(min(x), max(x) + 35, resolution)
    ylin = np.linspace(min(y), max(y), resolution)
    r = 5
 
    MinChanVal = min(chanVal)
    z = np.ones(len(x)) * MinChanVal
 
    selSen = np.ceil(selNum / 3)
 
    maxSen = int(max(selSen))
    for senIX in range(1, maxSen):
        currVal = np.zeros(2)
        for chanIX in range(1, 2):
            chanInd = (senIX - 1) * 3 + chanIX
            tmp = np.where(selNum == chanInd)
            if len(tmp) != 0:
                currVal[chanIX - matlab_offset] = chanVal[tmp]
        z[senIX] = max(currVal)
 
    X, Y = np.meshgrid(xlin, ylin)
    Z = scipy.interpolate.griddata((x, y), z, (X, Y), method='cubic')
    # pcm = plt.pcolor([X, Y], Z)
    plt.pcolor(Z)
    plt.axis('equal')  # ax.axis('equal')
    plt.axis('off')
    plt.colorbar()
    plt.show()
 
 
def getFullShuffledData(h0, h1):
    # 120 x 204
    h0 = np.transpose(h0)
    # 120 x 1
    h0Labels = np.zeros((120, 1))
    # 120 x 204
    h1 = np.transpose(h1)
    # 120 x 1
    h1Labels = np.ones((120, 1))
    # 240 x 204
    fullData = np.vstack((h0, h1))
    # 240 x 1
    fullLabels = np.concatenate((h0Labels, h1Labels))
    # shuffle in unison
    fullData, fullLabels = shuffle(fullData, fullLabels, random_state=0)
    return fullData, fullLabels
 
 
def secondLevelCrossValidation(k, fullData, fullLabels):
    kf = KFold(n_splits=k)
 
    maxAlpha = 0
    maxAccuracy = 0
    for j, (train_index, test_index) in enumerate(kf.split(fullData)):
        testingData = fullData[test_index, :]
        testingDataLabels = fullLabels[test_index]
        trainingData = fullData[train_index, :]
        trainingDataLabels = fullLabels[train_index]
 
        alphaRange = np.logspace(start=-10, stop=10, num=60, endpoint=True)
        for alpha in alphaRange:
            clf = svm.LinearSVC(penalty='l2', loss='squared_hinge', dual=False, C=(1 / alpha))
            clf.fit(X=trainingData, y=trainingDataLabels.ravel())
 
            accuracy = clf.score(X=testingData, y=testingDataLabels.ravel())
            # print(f"    {alpha} : {accuracy} : {maxAccuracy}")
            if accuracy > maxAccuracy:
                maxAccuracy = accuracy
                maxAlpha = alpha
        print(f"    Second Level Fold {j}: {maxAlpha}")
 
    return maxAlpha
 
 
def generateROCCurve(fprData, tprData, accuracyData):
 
    fpr_grid = np.linspace(0.0, 1.0, 1000)
    mean_tpr = np.zeros_like(fpr_grid)
    mean_accuracy = 0
    for i in range(6):
        mean_tpr += np.interp(fpr_grid, np.asarray(fprData[i]), np.asarray(tprData[i]))
        mean_accuracy += accuracyData[i]
 
    mean_tpr = mean_tpr / 6
    mean_accuracy = round((mean_accuracy / 6), 4)
 
    for i in range(6):
        fpr = np.asarray(fprData[i])
        tpr = np.asarray(tprData[i])
 
        # print(f"Fold {i}: {accuracyData[i]} vs Avg: {mean_accuracy}")
 
        plt.figure()
        plt.xlabel('True Positive Rate (TPR)')
        plt.ylabel('False Positive Rate (FPR)')
        plt.title(f"Overt Movement Testing Fold {i} ROC vs Total ROC")
        plt.xlim([0, 1.0])
        plt.ylim([0, 1.0])
 
        plt.plot(fpr, tpr, linewidth=2, color='blue')
        plt.plot(fpr_grid, mean_tpr, linewidth=2, color='orange')
        plt.plot(fpr, fpr, linewidth=1, color='black', linestyle='dashed')
 
        plt.legend(['Testing Fold ROC', 'Total ROC', 'Chance Level @ AUC=0.5'])
 
        plt.grid()
        plt.show()
 
    df = pd.DataFrame({
        'Individual Fold Accuracy': np.asarray(accuracyData),
        'Total Avg Accuracy': mean_accuracy
    })
 
    fig = plt.figure(figsize=(8, 2))
    ax = fig.add_subplot(111)
 
    ax.table(cellText=df.values,
             rowLabels=df.index,
             colLabels=df.columns,
             loc="center",
             cellLoc="center"
             )
    ax.set_title("Overt Movement Individual Fold Accuracy vs Total Average Accuracy")
 
    ax.axis("off")
    plt.show()
 
 
def firstLevelCrossValidation(k, fullData, fullLabels):
    kf = KFold(n_splits=k)
 
    fprData = []
    tprData = []
    accuracyData = []
    alphaData = []
    for i, (train_index, test_index) in enumerate(kf.split(fullData)):
        testingData = fullData[test_index, :]
        testingDataLabels = fullLabels[test_index]
        trainingData = fullData[train_index, :]
        trainingDataLabels = fullLabels[train_index]
 
        alpha = secondLevelCrossValidation(k=5, fullData=trainingData, fullLabels=trainingDataLabels)
        print(f"First Level Fold {i}: {alpha}")
        alphaData.append(alpha)
        clf = svm.LinearSVC(penalty='l2', loss='squared_hinge', dual=False, C=(1 / alpha))
        clf.fit(X=trainingData, y=trainingDataLabels.ravel())
 
        accuracy = clf.score(X=testingData, y=testingDataLabels.ravel())
        accuracyData.append(accuracy)
        decision_function = clf.decision_function(X=testingData)
 
        fpr, tpr, thresholds = metrics.roc_curve(y_true=testingDataLabels.ravel(), y_score=decision_function)
        fprData.append(fpr)
        tprData.append(tpr)
 
    # generateROCCurve(fprData=fprData, tprData=tprData, accuracyData=accuracyData)
 
    return alphaData, accuracyData
 
 
def singleLevelAnalysis(fullData, fullLabels):
    trainingData = fullData[40:, :]  # folds 2-6 training
    trainingDataLabels = fullLabels[40:, :]
 
    # train model on folds 2-6 and then get vector from clf
    alpha = secondLevelCrossValidation(k=5, fullData=trainingData, fullLabels=trainingDataLabels)
    clf = svm.LinearSVC(penalty='l2', loss='squared_hinge', dual=False, C=(1 / alpha))
    clf.fit(X=trainingData, y=trainingDataLabels.ravel())
 
    # 1x204
    weights = clf.coef_
    # 204x1
    weights = np.transpose(weights)
 
    show_chanWeights(abs(weights))
    show_weightsPlot(weight_vector=weights)
    show_nthLargestWeights(n=6, weight_vector=weights)
 
 
def fullLevelAnalysis(fullData, fullLabels):
    # goal: use roc_curve for each 1st level fold, and total ROC
    # need: true labels, decision function
    # need: svc of n_samples, n_features
    return firstLevelCrossValidation(k=6, fullData=fullData, fullLabels=fullLabels)
 
 
def main():
    # 204x120, need to transpose
    img_h0 = pd.read_csv('feaSubEImg_1.csv', header=None).to_numpy()
    img_h1 = pd.read_csv('feaSubEImg_2.csv', header=None).to_numpy()
    ovt_h0 = pd.read_csv('feaSubEOvert_1.csv', header=None).to_numpy()
    ovt_h1 = pd.read_csv('feaSubEOvert_2.csv', header=None).to_numpy()
 
    # 240x204, 240x1
    fullData, fullLabels = getFullShuffledData(h0=img_h0, h1=img_h1)
    # singleLevelAnalysis(fullData=fullData, fullLabels=fullLabels)
    img_alphaData, img_accuracyData = fullLevelAnalysis(fullData=fullData, fullLabels=fullLabels)
 
    fullData2, fullLabels2 = getFullShuffledData(h0=ovt_h0, h1=ovt_h1)
    # singleLevelAnalysis(fullData=fullData, fullLabels=fullLabels)
    ovt_alphaData, ovt_accuracyData = fullLevelAnalysis(fullData=fullData2, fullLabels=fullLabels2)
 
    img_alphaData = np.asarray(img_alphaData)
    img_accuracyData = np.asarray(img_accuracyData)
    ovt_alphaData = np.asarray(ovt_alphaData)
    ovt_accuracyData = np.asarray(ovt_accuracyData)
 
    img_maxAccuracy = np.max(img_accuracyData)
    ovt_maxAccuracy = np.max(ovt_accuracyData)
    img_bestAlpha = img_alphaData[np.where(img_accuracyData == img_maxAccuracy)][0]
    ovt_bestAlpha = ovt_alphaData[np.where(ovt_accuracyData == ovt_maxAccuracy)][0]
 
    # train on img
    img_clf = svm.LinearSVC(penalty='l2', loss='squared_hinge', dual=False, C=(1 / img_bestAlpha))
    img_clf.fit(fullData, fullLabels.ravel())
    # train on ovt
    ovt_clf = svm.LinearSVC(penalty='l2', loss='squared_hinge', dual=False, C=(1 / ovt_bestAlpha))
    ovt_clf.fit(fullData2, fullLabels2.ravel())
 
    # img train tested on ovt
    decision_function = img_clf.decision_function(X=fullData2)
    fpr, tpr, thresholds = metrics.roc_curve(y_true=fullLabels2, y_score=decision_function)
    auc = metrics.roc_auc_score(y_true=fullLabels2, y_score=decision_function)
    # ovt train tested on img
    decision_function2 = ovt_clf.decision_function(X=fullData)
    fpr2, tpr2, thresholds2 = metrics.roc_curve(y_true=fullLabels, y_score=decision_function2)
    auc2 = metrics.roc_auc_score(y_true=fullLabels, y_score=decision_function2)
 
    # ROC FOR IMG TRAIN TESTED ON OVT
    plt.figure()
    plt.xlabel('True Positive Rate (TPR)')
    plt.ylabel('False Positive Rate (FPR)')
    plt.title('ROC for Imaginary Trained SVM Tested on Overt Data')
    plt.xlim([0, 1.0])
    plt.ylim([0, 1.0])
 
    plt.plot(fpr, tpr, linewidth=2, color='blue')
    plt.plot(fpr, fpr, linewidth=1, color='black', linestyle='dashed')
 
    plt.legend([f'Ovt Tested ROC @ AUC={round(auc, 4)}', 'Chance Level @ AUC=0.5'])
 
    plt.grid()
    plt.show()
 
    # ROC FOR OVT TRAIN TESTED ON IMG
    plt.figure()
    plt.xlabel('True Positive Rate (TPR)')
    plt.ylabel('False Positive Rate (FPR)')
    plt.title('ROC for Overt Trained SVM Tested on Imaginary Data')
    plt.xlim([0, 1.0])
    plt.ylim([0, 1.0])
 
    plt.plot(fpr2, tpr2, linewidth=2, color='blue')
    plt.plot(fpr2, fpr2, linewidth=1, color='black', linestyle='dashed')
 
    plt.legend([f'Img Tested ROC @ AUC={round(auc2, 4)}', 'Chance Level @ AUC=0.5'])
 
    plt.grid()
    plt.show()
 
    # REGULARIZATION PARAM COMPARISON -------
 
    df = pd.DataFrame({
        'Imaginary Movement Values': img_alphaData,
        'Overt Movement Values': ovt_alphaData
    })
 
    df['Imaginary Movement Values'] = df['Imaginary Movement Values'].apply(float)
    df['Overt Movement Values'] = df['Overt Movement Values'].apply(float)
 
    fig = plt.figure(figsize=(8, 2))
    ax = fig.add_subplot(111)
 
    ax.table(cellText=df.values,
             rowLabels=df.index,
             colLabels=df.columns,
             loc="center",
             cellLoc="center"
             )
    ax.set_title("Individual Fold Regularization Parameter Value Comparison")
 
    ax.axis("off")
    plt.show()
 
 
main()
