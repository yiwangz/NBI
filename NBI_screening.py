import numpy as np
import os
import pandas as pd
import scipy as sp
from sklearn import svm
from sklearn.model_selection import GridSearchCV
import multiprocessing as mp
from scipy.stats import norm
from sklearn import linear_model
import NBIfunc as NBIfunc
import operator
import random

import warnings
warnings.filterwarnings(action="ignore", module="scipy", message="^internal gelsd")

########################################################################################################################
########################################################################################################################
def NBI_screening(seed, scenario, samplesize):

    ####################################################################################################################
    num_cv = 5
    num_perm = 500
    p = 0.5
    q = 0.1
    n_bio = 12
    bioNames = ["X" + str(ii) for ii in range(n_bio)]
    rho1 = 0.5
    rho2 = 0.2
    n = samplesize
    n_valid = int(n*0.8)
    n_indpt = 1000


    lambda_vec = [1/16,1/8,1/4,1/2,1,2,4,8]
    gamma_vec = [1/16,1/8,1/4,1/2,1,2,4,8]
    tuned_paras = [{'C':lambda_vec}]
    param_grid = dict(gamma=gamma_vec, C=lambda_vec)

    ####################################################################################################################
    ## set the seed
    np.random.seed(19891010+seed)

    ####################################################################################################################
    ## cross validation index ##########################################################################################
    cvIndex = NBIfunc.CVshuffle(n)
    cvIndex.CVindex(num_cv)
    cvIndex1 = cvIndex.CVindex1
    cvIndex2 = cvIndex.CVindex2

    ####################################################################################################################
    ## original data generation ########################################################################################
    dataOrig = NBIfunc.simData(n_bio, n, rho1, rho2)
    dataOrig.dataGen(decisionF=scenario)
    biomarker = dataOrig.biomarker
    A = dataOrig.A
    trueLabel = dataOrig.trueLabel
    B = dataOrig.B
    min_B = dataOrig.min_B

    ####################################################################################################################
    ## independent data generation ########################################################################################
    dataIndpt = NBIfunc.simData(n_bio, n_indpt, rho1, rho2)
    dataIndpt.dataGen(decisionF=scenario)
    biomarkerindpt = dataIndpt.biomarker
    Aindpt = dataIndpt.A
    trueLabelindpt = dataIndpt.trueLabel
    Bindpt = dataIndpt.B
    min_Bindpt = dataIndpt.min_B

    namesIndpt = ['Benefit','True_label','Assignment']+bioNames
    data_indpt = np.column_stack((Bindpt, trueLabelindpt, Aindpt, biomarkerindpt))
    data_indpt = pd.DataFrame(data_indpt, columns=namesIndpt)

    data_indpt_origB = data_indpt['Benefit']
    if min_Bindpt < 0:
        data_indpt_origB = data_indpt_origB - abs(min_Bindpt) - 0.001

    ####################################################################################################################
    ## matrix to store results #########################################################################################
    nbi_test = np.zeros((num_cv, n_bio))

    ####################################################################################################################
    ## start cross validation ##########################################################################################
    for cv in range(num_cv):
        Xexist = [0,1]  ## start with existing variables X1 and X2
        Xnew = list(range(2,n_bio))  ## start labeling as the position of biomarker
        n_Xnew = len(Xnew)
        n_select = 1

        while n_select > 0 and n_Xnew > 0:

            ############################################################################################################
            ## null data generation ####################################################################################
            XnullGen = NBIfunc.Xnull(biomarker, Xexist, Xnew, n, num_perm)
            XnullGen.Xnull_gen()
            Xnull = XnullGen.Xnull

            XnullNames = ["Xnull_" + str(ii) for ii in range(n_Xnew*num_perm)]

            varNames = ['Benefit','True_label','Assignment']
            names = varNames+bioNames+XnullNames

            data = np.column_stack((B,trueLabel,A,biomarker,Xnull))
            data = pd.DataFrame(data, columns=names)

            data_train = data.iloc[np.concatenate((cvIndex1[cv],cvIndex2[cv]))]
            data_validation = data.drop(np.concatenate((cvIndex1[cv],cvIndex2[cv])))

            ############################################################################################################
            ## nbi calculation for Xnew ################################################################################
            predLabel_mat = np.zeros((n_valid, n_Xnew+1))

            owlModel = GridSearchCV(svm.SVC(kernel='linear', C=1.0), tuned_paras, cv=5, fit_params={'sample_weight': data_train['Benefit']/p})

            for f in range(n_Xnew+1):

                if f == 0:
                    if len(Xexist) == 0:
                        predLabel = []
                        Alist = [-1,1]
                        for i in range(n_valid):
                            predLabel.append(random.sample(Alist,1)[0])
                    else:
                        owlModel.fit(data_train.iloc[:,[elem+3 for elem in Xexist]], data_train['Assignment'])
                        predLabel = owlModel.best_estimator_.predict(data_validation.iloc[:,[elem+3 for elem in Xexist]])
                else:
                    if len(Xexist) == 0:
                        owlModel.fit(data_train.iloc[:,[elem+3 for elem in [Xnew[f-1]]]], data_train['Assignment'])
                        predLabel = owlModel.best_estimator_.predict(data_validation.iloc[:,[elem+3 for elem in [Xnew[f-1]]]])
                    else:
                        owlModel.fit(data_train.iloc[:,[elem+3 for elem in Xexist+[Xnew[f-1]]]], data_train['Assignment'])
                        predLabel = owlModel.best_estimator_.predict(data_validation.iloc[:,[elem+3 for elem in Xexist+[Xnew[f-1]]]])

                predLabel_mat[:,f] = predLabel

            nbi_result = NBIfunc.NBI(predLabel_mat, data_validation)
            nbi_result.NBIcal(cut_num=5)
            relative_nbi = nbi_result.relative_nbi

            ############################################################################################################
            ## start permutation test ##################################################################################
            predLabelNull_mat = np.zeros((n_valid, n_Xnew*num_perm+1))
            predLabelNull_mat[:,0] = predLabel_mat[:,0]

            owlModel_null = GridSearchCV(svm.SVC(kernel='linear', C=1.0), tuned_paras, cv=5, fit_params={'sample_weight': data_train['Benefit']/p})  ## we need to add weights here

            for ff in range(n_Xnew*num_perm):
                if len(Xexist) == 0:
                    owlModel.fit(data_train.iloc[:,[elem+3 for elem in [n_bio+ff]]], data_train['Assignment'])
                    predLabelNull = owlModel.best_estimator_.predict(data_validation.iloc[:,[elem+3 for elem in [n_bio+ff]]])
                else:
                    owlModel_null.fit(data_train.iloc[:,[elem+3 for elem in Xexist+[n_bio+ff]]], data_train['Assignment'])
                    predLabelNull = owlModel_null.best_estimator_.predict(data_validation.iloc[:,[elem+3 for elem in Xexist+[n_bio+ff]]])

                predLabelNull_mat[:,ff+1] = predLabelNull

            emp_nbi_result = NBIfunc.NBI(predLabelNull_mat, data_validation)
            emp_nbi_result.NBIcal(cut_num=5)
            emp_nbi = emp_nbi_result.relative_nbi

            ############################################################################################################
            ## get the p values and BH test results ####################################################################
            p_vec = np.zeros(n_Xnew)

            for k in range(n_Xnew):
                p_vec[k] = len([item for sublist in np.where(emp_nbi[(k*num_perm):((k+1)*num_perm)] > relative_nbi[k]) for item in sublist])/num_perm

            p_sorted = sorted(p_vec)

            bh_test = np.array([(p_sorted[kk] <= (kk+1)*q/n_Xnew) for kk in range(n_Xnew)])
            bh_index = [item for sublist in np.where(bh_test == 1) for item in sublist]

            ############################################################################################################
            ## get the selected variables ##############################################################################
            if len(bh_index) != 0 :
                selectedMarker = np.array([(p_vec[kk] == p_sorted[0]) for kk in range(n_Xnew)])
                selectedMarker = [item for sublist in np.where(selectedMarker == 1) for item in sublist]

                if len(selectedMarker) == 1:
                    Xselect = [Xnew[selectedMarker[0]]]
                    n_select = 1
                else:
                    relative_nbi_selected = []
                    for jj in range(len(selectedMarker)):
                        relative_nbi_selected.append(relative_nbi[selectedMarker[jj]])

                    index_maxNBI, maxNBI = max(enumerate(relative_nbi_selected), key=operator.itemgetter(1))

                    Xselect = [Xnew[selectedMarker[index_maxNBI]]]
                    n_select = 1
            else:
                Xselect = []
                n_select = 0

            Xexist = Xexist+Xselect  ##update the existing variable
            Xnew = [elem for elem in Xnew if elem not in Xselect]  ##update the new variable
            n_Xnew = len(Xnew)  ##update the number of new variables

        ################################################################################################################
        ## record results ##############################################################################################
        Xselect_index = [elem for elem in list(range(n_bio)) if list(range(n_bio))[elem] in Xexist ]
        nbi_test[cv,Xselect_index] = 1

    d = dict()
    d['nbi_test'] = nbi_test
    return(d)

if __name__ == '__main__':

    scenario_char = "linear"
    samplesize = 1000
    samplesize_char = str(samplesize)

    ncpus = 1
    pool = mp.Pool(processes=ncpus)
    len_replicate = ncpus
    slurm_index = int(os.environ["SLURM_ARRAY_TASK_ID"])
    slurm_index_str = str(slurm_index)
    results = pool.starmap(NBI_screening, [(slurm_index, scenario_char, samplesize)])

    nbi_test = np.row_stack(results[i]['nbi_test'] for i in range(len_replicate))
    np.savetxt("NBI_screening_"+scenario_char+"_"+samplesize_char+"_nbi_test_"+slurm_index_str+".txt", nbi_test, delimiter=',')
