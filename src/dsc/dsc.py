#!/usr/bin/env python2.7
# -*- coding: utf-8 -*-

# --- Program internal modules -------------------------------------------
from __future__ import division
import numpy as np
import pandas as pd
import time
import librosa
import pickle
from sklearn import grid_search
from sklearn.decomposition import SparseCoder,DictionaryLearning
from sklearn import cluster
from lightning.regression import CDRegressor
import matplotlib.pyplot as plt
# --- Locally installed modules -----------------------------------------
from reader import Reader
# ------------------------------------------------------------------------


class DSC():
    def __init__(self,train_set,train_sum,gradient_step_size,epsilon,regularization_parameter,steps,n_components,m,T,k):
        """
        Either form is acceptable, but the two should not be mixed. Choose one
        convention to document the __init__ method and be consistent with it.

        Args:
          gradient_step_size (float): gradiant rate for the convergence step for DD (4b).
          epsilon (float) : the gradient stepsize of the pre-training (2b).
          regularization_parameter (float) : weight of penalty function.
          steps (int) : interations to be performed for the convergence part
          param2 (list of str): Description of `param2`. Multiple
            lines are supported.
          param3 (int, optional): Description of `param3`, defaults to 0.

        """
        self.train_set = train_set
        self.train_sum = train_sum
        self.alpha = gradient_step_size
        self.epsilon = epsilon
        self.rp = regularization_parameter
        self.steps = steps
        self.n = n_components
        self.m = m
        self.T = T
        self.k = k

        '''
        Instances that can be used for plotting
        '''
        self.acc_nnsc = None
        self.err_nnsc = None
        self.acc_ddsc = None
        self.err_ddsc = None
        self.a_nnsc = None
        self.b_nnsc = None
        self.a_ddsc = None
        self.b_ddsc = None


################################################################
# will initiualize the matrices A,B with positive values and scale
# columns of B s.t b(j) = 1
    def _initialization(self):

        a = np.random.random((self.n,self.m))
        b = np.random.random((self.T,self.n))

        # scale columns s.t. b_i^(j) = 1
        b /= sum(b)
        return a,b

    # add random positive number to A,B
    # scale columns of b(j)

#################################################################
    def accuracy(self,x,x_sum,B,A):
        '''
        Everything needs to be in lists of ndarrays
        of the components
        '''
        B_cat = np.hstack(B)
        A_cat = np.vstack(A)

        A_prime = self.F(x_sum.values,B_cat,A=A_cat)
        A_last = np.split(A_prime,self.k,axis=0)
        x_predict = self.predict(A_last,B)
        acc_numerator = (map(lambda i: (np.minimum( (B[i].dot(A_last[i])).sum() ,
                        (sum(x[i].sum())))) ,
                        range(len(B))))
        acc_denominator = sum(x_predict).sum()
        acc = sum(acc_numerator) / acc_denominator
        acc_numerator = (map(lambda i: (np.minimum( (B[i].dot(A_last[i])).sum() ,
                        (sum(x[i].sum())))) ,
                        range(len(B))))
        acc_denominator = x_sum.values.sum()
        acc_star = sum(acc_numerator) / acc_denominator
        return acc, acc_star

    def get_accuracy_plot(self):
        return self.acc_nnsc, self.acc_ddsc

    def get_error_plot(self):
        return self.err_nnsc, self.err_ddsc

    def get_a(self):
        return self.a_nnsc, self.a_ddsc

    def get_b(self):
        return self.b_nnsc, self.b_ddsc

    def error(self,x,x_sum,B,A):
        '''
        Error for the whole disaggregation part within list, sum the list to get
        the resulting disaggregation
        Parameters : must have x_train as x
        '''
        B_cat = np.hstack(B)
        A_cat = np.vstack(A)

        error = (map(lambda i: ((1.0/2.0)*np.linalg.norm( (x[i]
                       - B[i].dot(A[i]))**2)),range(len(B))))
        A_last_error = self.F(x_sum.values,B_cat,A_cat)
        A_last_error_list = np.split(A_last_error,self.k,axis=0)
        error_star = (map(lambda i: ((1.0/2.0)*np.linalg.norm( (x[i]
                       - B[i].dot(A_last_error_list[i]))**2)),range(len(B))))
        return error, error_star

    def pre_training(self,x):
        # TODO : implement s.t. conditions and frobenius norm to the options
        tic = time.time()
        #the NON NEGATIVE SPARSE CODING
        A_list,B_list = self.nnsc(x)

        tac = time.time()
        t = tac - tic
        print 'time of computations for Dictionary Learning with m: %s and T: %s took: %f' %(self.m,self.T,t)
        return A_list,B_list
################################################################
    # using only the positive values
    @staticmethod
    def _pos_constraint(a):
        indices = np.where(a < 0.0)
        a[indices] = 0.0
        return a
#################################################################
    def nnsc(self,appliances):
        '''
        Method as in NNSC from nonnegative sparse coding finland.
        from P.Hoyer
        TODO:
        implement the coordinate descent algorithm, as of now we are using         gradient descent (not as efficient)
        Also create multiple ndarrays that we take the argmin for.
        '''
        epsilon = 0.01
        acc_nnsc = []
        err_nnsc = []
        a_nnsc = []
        b_nnsc = []
        # used for F
        x_train_sum = self.train_set.values()
        A_list = []
        B_list = []
        for x in appliances:
            A,B = self._initialization()
            Ap = A
            Bp = B
            Ap1 = Ap
            Bp1 = Bp
            t = 0
            change = 1
            while t <= self.steps and self.epsilon <= change:
                # 2a
                Bp = Bp - self.alpha*np.dot((np.dot(Bp,Ap) - x),Ap.T)
                # 2b
                Bp = self._pos_constraint(Bp)
                # 2c
                Bp /= sum(Bp)
                # element wise division
                dot2 = np.divide(np.dot(Bp.T,x),(np.dot(np.dot(Bp.T,Bp),Ap) + self.rp))
                # 2d
                Ap = np.multiply(Ap,dot2)


                change = np.linalg.norm(Ap - Ap1)
                change2 = np.linalg.norm(Bp - Bp1)
                Ap1 = Ap
                Bp1 = Bp
                t += 1
                print "NNSC change is %s for iter %s, and B change is %s" %(change,t,change2)

            print "Gone through one appliance"
            A_list.append(Ap)
            B_list.append(Bp)

        # for thesis
        acc_iter = self.accuracy(x_train_sum,self.train_sum,B_list,A_list)
        err_iter = self.error(x_train_sum,self.train_sum,B_list,A_list)
        acc_nnsc.append(acc_iter)
        err_nnsc.append(err_iter)
        # append norm of matrices
        a_nnsc.append(np.linalg.norm(sum(A_list)))
        b_nnsc.append(np.linalg.norm(sum(B_list)))

        self.acc_nnsc = acc_nnsc
        self.err_nnsc = err_nnsc
        self.a_nnsc = a_nnsc
        self.b_nnsc = b_nnsc
        return A_list,B_list
#################################################################
    def F(self,x,B,x_train=None,A=None,rp_tep=False,rp_gl=False):
        '''
        input is lists of the elements
        output list of elements
        '''
        # 4b
        B = np.asarray(B)
        A = np.asarray(A)
       coder = SparseCoder(dictionary=B.T,
                            transform_alpha=self.rp, transform_algorithm='lasso_cd')
        comps, acts = librosa.decompose.decompose(x,transformer=coder)
        acts = self._pos_constraint(acts)

        return acts
#################################################################
    def DD(self,x,B,A):
        '''
        Taking the parameters as x_train_use and discriminate over the
        entire region
        '''
        # 3.
        A_star = np.vstack(A)
        B_cat = np.hstack(B)
        change = 1
        t = 0
        acc_ddsc = []
        err_ddsc = []
        a_ddsc = []
        b_ddsc = []
        x_train_sum = self.train_set.values()
        while t <= self.steps and self.epsilon <= change:
            B_cat_p = B_cat
            # 4a
            acts = self.F(x,B_cat,A=A_star)
            # 4b
            B_cat = (B_cat-self.alpha*((x-B_cat.dot(acts))
                     .dot(acts.T) - (x-B_cat.dot(A_star)).dot(A_star.T)))
            # 4c
            # scale columns s.t. b_i^(j) = 1
            B_cat = self._pos_constraint(B_cat)
            B_cat /= sum(B_cat)

            # convergence check
            acts_split = np.split(acts,self.k,axis=0)
            B_split = np.split(B_cat,self.k,axis=1)
            acc_iter = self.accuracy(x_train_sum,self.train_sum,B,acts_split)
            acc_iter = self.accuracy(x_train_sum,self.train_sum,B_split,A)
            err_iter = self.error(x_train_sum,self.train_sum,B,acts_split)
            acc_ddsc.append(acc_iter)
            err_ddsc.append(err_iter)
            a_ddsc.append(np.linalg.norm(acts))
            b_ddsc.append(np.linalg.norm(B_cat))

            change = np.linalg.norm(B_cat - B_cat_p)
            t += 1
            print "DD change is %f and step is %d" %(change,t)

        self.acc_ddsc = acc_ddsc
        self.err_ddsc = err_ddsc
        self.a_ddsc = a_ddsc
        self.b_ddsc = b_ddsc
        return B_cat

#################################################################
    def predict(self,A,B):
        x = map(lambda x,y: x.dot(y),B,A)
        return x
#################################################################
def main():
    '''
    Program to train disaggregation using Sparse Coding
    '''
    #read dataset
    data = 0
    dataset = ['cleanpecanhour2014','weekendpecanhour2014','weekdayspecanhour2014']
    reader = Reader(dataset[data])
    d = pickle.load( open( "do.p", "rb" ) )

    timeframes = [7,14,30]
    timeframes = [x*24 for x in timeframes]
    timeframes = [14,30]
    portion = 0.5
    n = 20
    for timeframe in timeframes:
        x_train, x_test = reader.split(d,portion,timeframe)

        # use in whole house disaggregation step
        x_train_use = x_train.pop('use',None)
        x_test_use = x_test.pop('use',None)
        x_train_localhour = x_train.pop('localhour',None)
        x_test_localhour = x_test.pop('localhour',None)

        # algorithm starts
        # parameters
        train_set = x_train
        test_set = x_test
        train_sum = sum(x_train.values())
        k = len(x_train.keys())
        T,m = x_train[x_train.keys()[0]].shape
        rp = 0.0005
        epsilon = 0.001
        alpha = 0.0001
        steps = 100 # steps must be higher than k
        # get data
        n_components = n

        # Discriminative Sparse Coding pre_training
        dsc = DSC(train_set,train_sum,alpha,epsilon,rp,steps,n_components,m,T,k)
        print "started the pre-training"
        A_list,B_list = dsc.pre_training(x_train.values())
        print "done pre_training"
        # Discriminative Disaggregation training
        B_cat = dsc.DD(x_train_use.values,B_list,A_list)
        print "done DD"
        # Given test examples x_test
        A_prime = dsc.F(x_test_use.values,B_cat,A=np.vstack(A_list))
        A_last = np.split(A_prime,k,axis=0)
        x_predict = dsc.predict(A_last,B_list)
        x_predict_sum = sum(x_predict)
        print "the shape of the first predicted appliances is :%s" %(x_predict[0].shape,)
        # energy disaggregation accuracy
        acc = dsc.accuracy(x_train.values(),train_sum,B_list,A_last)
        # energy disaggregation error
        error, error_star = dsc.error(x_train.values(),train_sum,B_list,A_list)
        print "error: %s, error_star: %s" % (sum(error),sum(error_star))
        acc_nnddsc, acc_ddddsc = dsc.get_accuracy_plot()
        err_nnddsc, err_ddddsc = dsc.get_error_plot()
        # plotting acc/err
        a_nndsc, a_dsc = dsc.get_a()
        b_nndsc, b_dsc = dsc.get_b()

if __name__ == '__main__':
    main()
