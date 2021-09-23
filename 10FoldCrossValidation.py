''' created on  12 May 2021
author: Oanh Thi La
purposes:
1. split data for training into 10 fold cross validation
2. split data for training follow the rule of 2 samples for train, 1 samples for validation
'''


import numpy as np
from numpy import load, save, concatenate
from sklearn.model_selection import KFold
import pandas as pd
import tkinter as tkinter
from tkinter import filedialog
from tkinter.filedialog import askopenfilename, askdirectory
from sklearn import preprocessing

# method 1: SPLIT TRAINING DATA INTO 10 Folds using K-fold cross validation

def load_trainingdata():
    x_training_path = askopenfilename(title='Choose x_training files', filetypes=[("NPY", ".npy")])
    xtraining = np.load(x_training_path)
    n_samples = xtraining.shape[0]
    TOA_xtraining = np.zeros([n_samples, 8, 3, 3])
    for i in range(0, n_samples):
        for j in range(0, 8):
            TOA_xtraining[i][j] = xtraining[i][j]
    TOA_xtraining_reshape = TOA_xtraining.reshape(TOA_xtraining.shape[0], 3, 3, 8, 1).astype('float32')

    angles_xtraining = np.zeros([n_samples, 3, 3, 3])
    for m in range(0, n_samples):
        for k in range(0, 3):
            angles_xtraining[m][k] = xtraining[m][k + 8]
    angles_xtraining_reshape = angles_xtraining.reshape(angles_xtraining.shape[0], 3 * 3 * 3).astype('float32')
## normalize angles data from (-1,1) to scale (0,1)
    pdread_angles = pd.DataFrame(angles_xtraining_reshape)
    scaler = preprocessing.MinMaxScaler()
    colums = pdread_angles.columns
    transform = scaler.fit_transform(pdread_angles)
    angles_xtraining_normalize = pd.DataFrame(transform, columns=colums)
    angles_xtraining_normalize.head()
    angles_xtraining_final = angles_xtraining_normalize.to_numpy()

    AOT_xtraining = np.zeros([n_samples, 1, 3, 3])
    for n in range(0, n_samples):
        for h in range(0, 1):
            AOT_xtraining[n][h] = xtraining[n][h + 11]
    AOT_xtraining_reshape = AOT_xtraining.reshape(AOT_xtraining.shape[0], 3 * 3 * 1).astype('float32')

    y_training_iCOR_path = askopenfilename(title='Choose y_training files', filetypes=[("NPY", ".npy")])
    ytraining_iCOR = np.load(y_training_iCOR_path)
    return TOA_xtraining_reshape, angles_xtraining_final, AOT_xtraining_reshape, ytraining_iCOR


TOA_xtraining_reshape, angles_xtraining_final, AOT_xtraining_reshape, ytraining_iCOR = load_trainingdata()
path_save = tkinter.filedialog.asksaveasfilename(title=u'Save to npy file', filetypes=[("NPY", ".npy")])
save(path_save, angles_xtraining_final) #save to all data for training

## split training data into k-fold here
def kFold(TOA_xtraining_reshape, angles_xtraining_final, AOT_xtraining_reshape, ytraining_iCOR):
    kf = KFold(n_splits=10, shuffle=True,
               random_state=1)  # n_splits is the number of folds (int number)  in array want to split
    kf.get_n_splits([TOA_xtraining_reshape, angles_xtraining_final, AOT_xtraining_reshape],
                    ytraining_iCOR)  # return the number of splitting iterations in the cross-validator
    return kf


kf = kFold(TOA_xtraining_reshape, angles_xtraining_final, AOT_xtraining_reshape, ytraining_iCOR)

foldnumber = 0
for train_index, vali_index in kf.split(TOA_xtraining_reshape, ytraining_iCOR):
    TOA_train, TOA_valid = TOA_xtraining_reshape[train_index], TOA_xtraining_reshape[vali_index]
    angles_train, angles_valid = angles_xtraining_final[train_index], angles_xtraining_final[vali_index]
    AOT_train, AOT_valid = AOT_xtraining_reshape[train_index], AOT_xtraining_reshape[vali_index]
    print("XTRAIN:", train_index, "XVALID:", vali_index)
    y_train, y_valid = ytraining_iCOR[train_index], ytraining_iCOR[vali_index]
    print("YTRAIN:", train_index, "YVALID:", vali_index)
    # SAVE
    foldnumber = foldnumber + 1
    path_save = tkinter.filedialog.askdirectory()
    save(path_save + '_TOA_train' + ".npy", TOA_train)
    save(path_save + '_TOA_vali' + ".npy", TOA_valid)

    save(path_save + '_angles_train' + ".npy", angles_train)
    save(path_save + '_angles_vali' + ".npy", angles_valid)

    save(path_save + '_AOT_train' + ".npy", AOT_train)
    save(path_save + '_AOT_vali' + ".npy", AOT_valid)

    save(path_save + '_y_train_iCOR' + ".npy", y_train)
    save(path_save + '_y_vali_iCOR' + ".npy", y_valid)




## method2: SPLIT TRAINING DATA FOLLOW rule of 2 train 1 validation.
## SELECTING X_VALIDATION FROM TOA_xtraining_reshape, angles_xtraining_final, AOT_xtraining_reshape
def select_TOA_x_validsamle(TOA_xtraining_reshape):
    npatch = TOA_xtraining_reshape.shape[0]
    n_validsample = int(npatch/3)
    TOA_x_vali_samples = np.zeros([n_validsample, 3, 3, 8, 1])
    for i in range(0, n_validsample):
        TOA_x_vali_samples[i] = TOA_xtraining_reshape[3*i]

    return TOA_x_vali_samples


TOA_x_vali_samples = select_TOA_x_validsamle(TOA_xtraining_reshape)

path_save = tkinter.filedialog.askdirectory()
save(path_save + 'TOA_XVali.npy', TOA_x_vali_samples)

def select_angles_x_validsamle(angles_xtraining_final):
    npatch = angles_xtraining_final.shape[0]
    n_validsample = int(npatch/3)
    angles_x_vali_samples = np.zeros([n_validsample, 27])
    for i in range(0, n_validsample):
        angles_x_vali_samples[i] = angles_xtraining_final[3*i]

    return angles_x_vali_samples


angles_x_vali_samples = select_angles_x_validsamle(angles_xtraining_final)

path_save = tkinter.filedialog.askdirectory()
save(path_save + 'angles_XVali.npy', angles_x_vali_samples)

def select_AOT_x_validsamle(AOT_xtraining_reshape):
    npatch = AOT_xtraining_reshape.shape[0]
    n_validsample = int(npatch/3)
    AOT_x_vali_samples = np.zeros([n_validsample, 9])
    for i in range(0, n_validsample):
       AOT_x_vali_samples[i] = AOT_xtraining_reshape[3*i]

    return AOT_x_vali_samples


AOT_x_vali_samples = select_AOT_x_validsamle(AOT_xtraining_reshape)

path_save = tkinter.filedialog.askdirectory()
save(path_save + 'AOT_XVali.npy', AOT_x_vali_samples)


## SELECTING X_TRAIN FROM TOA_xtraining_reshape, angles_xtraining_final, AOT_xtraining_reshape
def select_TOA_x_trainsample(TOA_xtraining_reshape):
    npatch = TOA_xtraining_reshape.shape[0]
    n_valid = int(npatch/3)
    n_trainsample = npatch - n_valid
    TOA_x_train = np.zeros([n_trainsample, 3, 3, 8, 1])
    list = []
    for i in range(0, len(TOA_xtraining_reshape), 3): # i run in range 0 to 63534 with step is 3
       list.append(i) # list = (0, 3, 6, 9...)
       # or use this for loop below
    # for i in range(n_valid): # i run from 0 to 21178
    #    list.append(i*3) # list = (0, 3, 6, 9...42356)
       TOA_x_train = np.delete(TOA_xtraining_reshape, list, 0) # delete all the rows in list

    return TOA_x_train


TOA_x_train = select_TOA_x_trainsample(TOA_xtraining_reshape)
path_save = tkinter.filedialog.askdirectory()
save(path_save + 'TOA_XTrain.npy', TOA_x_train)

def select_angles_x_trainsample(angles_xtraining_final):
    npatch = angles_xtraining_final.shape[0]
    n_valid = int(npatch/3)
    n_trainsample = npatch - n_valid
    angles_x_train = np.zeros([n_trainsample, 27])
    list = []
    for i in range(0, len(angles_xtraining_final), 3): # i run in range 0 to 63534 with step is 3
       list.append(i) # list = (0, 3, 6, 9...)
       # or use this for loop below
    # for i in range(n_valid): # i run from 0 to 21178
    #    list.append(i*3) # list = (0, 3, 6, 9...42356)
       angles_x_train = np.delete(angles_xtraining_final, list, 0) # delete all the rows in list

    return angles_x_train


angles_x_train = select_angles_x_trainsample(angles_xtraining_final)
path_save = tkinter.filedialog.askdirectory()
save(path_save + 'angles_XTrain.npy', angles_x_train)

def select_AOT_x_trainsample(AOT_xtraining_reshape):
    npatch = AOT_xtraining_reshape.shape[0]
    n_valid = int(npatch/3)
    n_trainsample = npatch - n_valid
    AOT_x_train = np.zeros([n_trainsample, 9])
    list = []
    for i in range(0, len(AOT_xtraining_reshape), 3): # i run in range 0 to 63534 with step is 3
       list.append(i) # list = (0, 3, 6, 9...)
       # or use this for loop below
    # for i in range(n_valid): # i run from 0 to 21178
    #    list.append(i*3) # list = (0, 3, 6, 9...42356)
       AOT_x_train = np.delete(AOT_xtraining_reshape, list, 0) # delete all the rows in list

    return AOT_x_train


AOT_x_train = select_AOT_x_trainsample(AOT_xtraining_reshape)
path_save = tkinter.filedialog.askdirectory()
save(path_save + 'AOT_XTrain.npy', AOT_x_train)

## SELECTING Y_VALIDATION FROM ytraining_iCOR
def select_y_validsamle(ytraining_iCOR):
    [npatch, sizepatch] = ytraining_iCOR.shape
    n_validsample = int(npatch/3)
    y_vali_samples = np.zeros([n_validsample, 5])
    for j in range(0, n_validsample):
        y_vali_samples[j] = ytraining_iCOR[3*j]

    return  y_vali_samples


y_vali_samples = select_y_validsamle(ytraining_iCOR)
path_save = tkinter.filedialog.askdirectory()
save(path_save + 'iCOR_Y_vali.npy', y_vali_samples)


## SELECTING Y_TRAIN BY DELETING Y_VALI IN iCOR_Rrs_nonsta_dataset1D
def select_y_trainsample(ytraining_iCOR):
    npatch = ytraining_iCOR.shape[0]
    n_valid = (int(npatch/3))
    n_trainsample = npatch - n_valid
    y_train = np.zeros([n_trainsample, 5])
    list = []
    for i in range(0, len(ytraining_iCOR), 3): # i run in range 0 to 63534 with step is 3
       list.append(i) # list = (0, 3, 6, 9...4845)
       # or use this for loop below
    # for i in range(n_valid): # i run from 0 to 1615
    #    list.append(i*3) # list = (0, 3, 6, 9...4845)
       y_train = np.delete(ytraining_iCOR, list, 0) # delete all the rows in list

    return y_train

y_train = select_y_trainsample(ytraining_iCOR)
path_save = tkinter.filedialog.askdirectory()
save(path_save + 'iCOR_YTrain.npy', y_train)



# # for PHASE 2: only station data of
## for x_test (only station TOA patches)
TOA_test_paths = filedialog.askopenfilenames(title='Choose TOA testing files', filetypes=[("NPY", ".npy")])
TOA_BaBe_20180810 = np.load(TOA_test_paths[00])
TOA_BaBe_20170504 = np.load(TOA_test_paths[1])
TOA_BaMau_20160601 = np.load(TOA_test_paths[2])
TOA_BayMau_20160601 = np.load(TOA_test_paths[3])
TOA_LinhDam_20170401= np.load(TOA_test_paths[4])
TOA_LinhDam_20160601= np.load(TOA_test_paths[5])
TOA_NghiaTan_20160601= np.load(TOA_test_paths[6])
TOA_ThuLe_20190626 = np.load(TOA_test_paths[7])
TOA_VanQuan_20190626 = np.load(TOA_test_paths[8])
TOA_Westlake_20160601 = np.load(TOA_test_paths[9])
TOA_Westlake_20190813 = np.load(TOA_test_paths[10])
TOA_XaDan_20160601= np.load(TOA_test_paths[11])

xtesting = np.concatenate((TOA_BaBe_20180810, TOA_BaBe_20170504, TOA_BaMau_20160601, TOA_BayMau_20160601,
                           TOA_LinhDam_20170401, TOA_LinhDam_20160601, TOA_NghiaTan_20160601, TOA_ThuLe_20190626,
                                      TOA_VanQuan_20190626, TOA_Westlake_20160601, TOA_Westlake_20190813, TOA_XaDan_20160601), axis=0) .astype('float32')

n_testsamples = xtesting.shape[0]
TOA_xtesting = np.zeros([n_testsamples, 8, 3, 3])
for i in range(0, n_testsamples):
    for j in range(0, 8):
        TOA_xtesting[i][j] = xtesting[i][j]
TOA_xtesting_reshape = TOA_xtesting.reshape(TOA_xtesting.shape[0], 3, 3, 8, 1).astype('float32')
path_save = tkinter.filedialog.asksaveasfilename(title=u'Save to npy file', filetypes=[("NPY", ".npy")])
save(path_save, TOA_xtesting_reshape)

angles_xtesting = np.zeros([n_testsamples, 3, 3, 3])
for m in range(0, n_testsamples):
    for k in range(0, 3):
        angles_xtesting[m][k] = xtesting[m][k + 8]
angles_xtesting_reshape = angles_xtesting.reshape(angles_xtesting.shape[0], 3 * 3 * 3).astype('float32')
## normalize angles data from (-1,1) to scale (0,1)
pdread_angles_test = pd.DataFrame(angles_xtesting_reshape)
scaler2 = preprocessing.MinMaxScaler()
colums2 = pdread_angles_test.columns
transform2 = scaler2.fit_transform(pdread_angles_test)
angles_xtesting_normalize = pd.DataFrame(transform2, columns=colums2)
angles_xtesting_normalize.head()
angles_xtesting_final = angles_xtesting_normalize.to_numpy()
path_save = tkinter.filedialog.asksaveasfilename(title=u'Save to npy file', filetypes=[("NPY", ".npy")])
save(path_save, angles_xtesting_final)

AOT_xtesting = np.zeros([n_testsamples, 1, 3, 3])
for n in range(0, n_testsamples):
    for h in range(0, 1):
        AOT_xtesting[n][h] = xtesting[n][h + 11]
AOT_xtesting_reshape = AOT_xtesting.reshape(AOT_xtesting.shape[0], 3 * 3 * 1).astype('float32')
path_save = tkinter.filedialog.asksaveasfilename(title=u'Save to npy file', filetypes=[("NPY", ".npy")])
save(path_save, AOT_xtesting_reshape)


## for y test (iCOR)
iCOR_test_paths = filedialog.askopenfilenames(title='Choose iCOR testing files', filetypes=[("NPY", ".npy")])
iCOR_Westlake_20190813 = np.load(iCOR_test_paths[00])
iCOR_BaBe_20170504 = np.load(iCOR_test_paths[1])
iCOR_BaBe_20180810 = np.load(iCOR_test_paths[2])
iCOR_BaMau_20160601 = np.load(iCOR_test_paths[3])
iCOR_BayMau_20160601 = np.load(iCOR_test_paths[4])
iCOR_LinhDam_20160601= np.load(iCOR_test_paths[5])
iCOR_LinhDam_20170401 = np.load(iCOR_test_paths[6])
iCOR_NghiaTan_20160601 = np.load(iCOR_test_paths[7])
iCOR_ThuLe_20190626  = np.load(iCOR_test_paths[8])
iCOR_VanQuan_20190626 = np.load(iCOR_test_paths[9])
iCOR_Westlake_20160601 = np.load(iCOR_test_paths[10])
iCOR_XaDan_20160601= np.load(iCOR_test_paths[11])

iCOR_testing_dataset = np.concatenate((iCOR_BaBe_20180810, iCOR_BaBe_20170504, iCOR_BaMau_20160601, iCOR_BayMau_20160601,
                                       iCOR_LinhDam_20170401, iCOR_LinhDam_20160601, iCOR_NghiaTan_20160601,
                                       iCOR_ThuLe_20190626, iCOR_VanQuan_20190626, iCOR_Westlake_20160601,
                                       iCOR_Westlake_20190813, iCOR_XaDan_20160601), axis=0) .astype('float32')

iCOR_Rrs_testing_AVERAGE = np.zeros([iCOR_testing_dataset.shape[0], 5, 1])
for i in range(iCOR_testing_dataset.shape[0]):
    for j in range(5):
        iCOR_Rrs_testing_AVERAGE[i, j] = np.average(iCOR_testing_dataset[i][j])
iCOR_Rrs_test_Avr_reshape = iCOR_Rrs_testing_AVERAGE.reshape(iCOR_Rrs_testing_AVERAGE.shape[0], 5).astype('float32')

path_save = tkinter.filedialog.asksaveasfilename(title=u'Save to npy file', filetypes=[("NPY", ".npy")])
save(path_save, iCOR_Rrs_test_Avr_reshape)
##save to iCOR_Rrs_ytesting to csv file
path_save = tkinter.filedialog.asksaveasfilename(title=u'Save to excel file', filetypes=[("Excel", ".csv")])
np.savetxt(path_save, iCOR_Rrs_test_Avr_reshape, delimiter=",")


# for y_test (insitu samples)

insitu_testing_path = filedialog.askopenfilename(title=u'Open insitu for testing file',filetypes=[("Excel files", ".xlsx .xls")])
insitu_testing = pd.read_excel(insitu_testing_path)
insitu_testingdata = insitu_testing[['B1', 'B2', 'B3', 'B4', 'B5']].values

path_save = tkinter.filedialog.asksaveasfilename(title=u'Save to npy file', filetypes=[("NPY", ".npy")])
save(path_save, insitu_testingdata)