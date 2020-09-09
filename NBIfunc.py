import numpy as np
import os
import pandas as pd
import scipy as sp
from sklearn import svm
from sklearn.model_selection import GridSearchCV
import multiprocessing as mp
from scipy.stats import norm
from sklearn import linear_model

########################################################################################################################
## partition a list
def CVpart(lst, n):
    division = len(lst) / float(n)
    return [lst[int(round(division*i)):int(round(division*(i+1)))] for i in range(n)]

########################################################################################################################
## generate CV indices
class CVshuffle:
    def __init__(self, n):
        self.n = n

    def CVindex(self, num_cv):
        index1 = np.arange(int(self.n/2))
        index2 = np.arange(int(self.n/2),self.n)
        np.random.shuffle(index1)
        np.random.shuffle(index2)
        CVindex1 = CVpart(index1, num_cv)
        CVindex2 = CVpart(index2, num_cv)

        self.CVindex1 = CVindex1
        self.CVindex2 = CVindex2

########################################################################################################################
## calculate NBI values for the new variables
class NBI:
    def __init__(self, predLabel_mat, data_validation):
        self.predLabel_mat = predLabel_mat
        self.data_validation = data_validation

    def NBIcal(self, cut_num=5):
        R_all_class = np.column_stack((self.data_validation['Benefit'],self.data_validation['Assignment'],self.predLabel_mat))
        R_all_class = pd.DataFrame(R_all_class)

        vector = range(self.predLabel_mat.shape[1]-1)

        gain_index = pd.Series(vector).apply(lambda x: [i for i in range(self.data_validation.shape[0])
                                                       if ((int(R_all_class.iloc[i,1])==1 and int(R_all_class.iloc[i,2])==-1 and int(R_all_class.iloc[i,x+3])==1) or
                                                           (int(R_all_class.iloc[i,1])==-1 and int(R_all_class.iloc[i,2])==1 and int(R_all_class.iloc[i,x+3])==-1))])

        loss_index = pd.Series(vector).apply(lambda x: [i for i in range(self.data_validation.shape[0])
                                                       if ((int(R_all_class.iloc[i,1])==1 and int(R_all_class.iloc[i,2])==1 and int(R_all_class.iloc[i,x+3])==-1) or
                                                           (int(R_all_class.iloc[i,1])==-1 and int(R_all_class.iloc[i,2])==-1 and int(R_all_class.iloc[i,x+3])==1))])

        raw_nbi = np.zeros(len(vector))
        relative_nbi = np.zeros(len(vector))

        for i in range(len(vector)):
            if (len(gain_index[i]) >= cut_num and len(loss_index[i]) >= cut_num):
                raw_nbi[i] = np.mean(R_all_class.iloc[gain_index[i],0])-np.mean(R_all_class.iloc[loss_index[i],0])
                t,p = sp.stats.ttest_ind(R_all_class.iloc[gain_index[i],0], R_all_class.iloc[loss_index[i],0], equal_var=False)
                relative_nbi[i] = str(t)
            else:
                raw_nbi[i] = 0
                relative_nbi[i] = 0

        self.raw_nbi = raw_nbi
        self.relative_nbi = relative_nbi


########################################################################################################################
## generate the null variables by residual permutation
class Xnull:
    def __init__(self, biomarker, Xexist, Xnew, n, num_perm):
        self.biomarker = biomarker
        self.Xexist = Xexist
        self.Xnew = Xnew
        self.n = n
        self.num_perm = num_perm

    def Xnull_gen(self):
        Xnull_mat = np.zeros((self.n, self.num_perm*len(self.Xnew)))

        for i in range(len(self.Xnew)):

            if len(self.Xexist) != 0:
                model_Xnew = linear_model.LinearRegression()
                model_Xnew.fit(self.biomarker[:,self.Xexist],self.biomarker[:,self.Xnew[i]])
                Xnew_predict = model_Xnew.predict(self.biomarker[:,self.Xexist])
            else:
                Xnew_predict = np.zeros(self.n)

            residual = self.biomarker[:,self.Xnew[i]]-Xnew_predict

            for j in range(self.num_perm):
                residual_perm = np.zeros(self.n)
                residual_perm[:int(self.n/2)] = np.random.choice(residual[:int(self.n/2)], int(self.n/2), replace=False)
                residual_perm[int(self.n/2):] = np.random.choice(residual[int(self.n/2):], int(self.n/2), replace=False)
                Xnull_mat[:,i*self.num_perm+j] = Xnew_predict+residual_perm

        self.Xnull = Xnull_mat


########################################################################################################################
## generate simulation data
class simData:
    def __init__(self, n_bio, n, rho1, rho2):
        self.n_bio = n_bio
        self.n = n
        self.rho1 = rho1
        self.rho2 = rho2

    def dataGen(self, decisionF = 'linear'):
        mean = np.zeros(self.n_bio)

        cov1 = [[1,0], [0,1]]
        cov2 = [[1,0,self.rho1,0], [0,1,0,self.rho1], [self.rho1,0,1,0], [0,self.rho1,0,1]]
        cov3 = np.zeros((self.n_bio-6,self.n_bio-6))
        cov3[:,:] = self.rho2
        for i in range(self.n_bio-6):
            cov3[i,i] = 1.0

        cov = np.zeros((self.n_bio, self.n_bio))
        cov[:2,:2] = cov1
        cov[2:6,2:6] = cov2
        cov[6:self.n_bio,6:self.n_bio] = cov3

        biomarker_norm = np.random.multivariate_normal(mean, cov, self.n)

        biomarker = np.zeros((self.n,self.n_bio))
        for i in range(self.n_bio):
            biomarker[:,i] = norm.cdf(biomarker_norm[:,i])

        A = np.repeat((1,-1), repeats=(self.n/2,self.n/2))

        mu = np.zeros(self.n)
        if decisionF == 'linear':
            for i in range(self.n):
                mu[i] = 0.5+biomarker[i,0]+A[i]*(1+biomarker[i,0]+biomarker[i,1]-1.8*biomarker[i,2]-2.2*biomarker[i,3])
        elif decisionF == 'binary':
            for i in range(self.n):
                mu[i] = 0.5+biomarker[i,0]+12*A[i]*((biomarker[i,0]>0.12 and biomarker[i,1]<0.88 and biomarker[i,2]>0.2 and biomarker[i,3]<0.8)-0.5)
        elif decisionF == 'nonlinear':
            for i in range(self.n):
                mu[i] = 0.5+biomarker[i,0]+2*A[i]*(max(0,biomarker[i,0]-0.9)-max(0,biomarker[i,1]-0.78)+max(0,biomarker[i,2]-0.1)-max(0,biomarker[i,3]-0.22))
        else:
            print('Decision function not supported.')

        trueLabel = np.zeros(self.n)
        if decisionF == 'linear':
            for i in range(self.n):
                if 1+biomarker[i,0]+biomarker[i,1]-1.8*biomarker[i,2]-2.2*biomarker[i,3] > 0:
                    trueLabel[i] = 1
                else:
                    trueLabel[i] = -1
        elif decisionF == 'binary':
            for i in range(self.n):
                if (biomarker[i,0]>0.12 and biomarker[i,1]<0.88 and biomarker[i,2]>0.2 and biomarker[i,3]<0.8)-0.5 > 0:
                    trueLabel[i] = 1
                else:
                    trueLabel[i] = -1
        elif decisionF == 'nonlinear':
            for i in range(self.n):
                if max(0,biomarker[i,0]-0.9)-max(0,biomarker[i,1]-0.78)+max(0,biomarker[i,2]-0.1)-max(0,biomarker[i,3]-0.22) > 0:
                    trueLabel[i] = 1
                else:
                    trueLabel[i] = -1

        B = np.random.normal(loc=mu, scale=1, size=self.n)

        min_B = min(B)

        if min_B < 0:
            B = B+abs(min_B)+0.001

        self.biomarker = biomarker
        self.A = A
        self.trueLabel = trueLabel
        self.B = B
        self.min_B = min_B



















