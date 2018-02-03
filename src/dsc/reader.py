#!/usr/bin/env python

import os
import sys
import pandas as pd
from collections import defaultdict
from StringIO import StringIO
#################################################################

class Data():
    def __init__(self,dataset):
        self.dataset = dataset
        #self.data_frame = None


class Reader():

    def __init__(self,dataset):
        self.dataset = dataset

    def dataSetToFile(self):

        def find(name, path):
            for root, dirs, files in os.walk(path):
                print root,dirs,files
                if name in files:
                    return os.path.join(root, name)
                else:
                    print 'Could not find the dataset %s' %self.dataset
                    sys.exit("Error message")

        self.dataset += '.csv'

        return find(self.dataset, '../../../res/datasets')

    #################################################################
    # to preProcess inputData into how many appliances

    # training for each individual source in the household
    # regularization parameter penalty,
    # gradient step size alpha
    # convergence rate, conv
    def dataParser(self):#, timeScale, numberAppliances):
        datasets = {'pecanminute2014','pecanhour2014','testhouses2012','energimynd','cleanpecanhour2014','weekendpecanhour2014','weekdayspecanhour2014'}

        appliances = defaultdict(dict)
        if type(self.dataset) != str:
#        if type(dataSet) != str:
            print 'Dataset needs to be in string format'
            sys.exit("Error message")

        if self.dataset in datasets:
            readfile = self.dataSetToFile()
            f = open(readfile,'rb')
            if readfile == 'testhouses2012.csv':
                df = pd.read_csv(f,index_col=[0],sep='|',usecols=[0,1,2,3,4,21,22,26,31,41],header=0)
            elif readfile == 'cleanpecanhour2014.csv':
                df = pd.read_csv(f,index_col=[0],sep=';',header=0,na_values=['g'])
            else:
                df = pd.read_csv(f,index_col=[0],sep=';',header=0,na_values=['g'])
        else:
            sys.exit("Could not read the data")
#        return houses, appliances, timeScale, gradientStep        
        return df

    #################################################################
    def format_data(self,df,other=True):
        '''
        Parameters: df dataframe of the apppliacnes
        Return: X^T x m
        '''
        def add_other(df):
            '''
            assuming there is only localhour and use that does not need to
            be treated
            '''
            non_appliances = []
            non_appliances.append(df.columns.tolist().index('localhour'))
            non_appliances.append(df.columns.tolist().index('use'))
            list_appliances = [i for j, i in enumerate(df.columns.tolist()) if j not in non_appliances]
            appliances_sum = df[list_appliances].sum(axis=1)
            return df['use'].subtract(appliances_sum)

        if other == True:
            df['other'] = add_other(df)

        unique = pd.unique(df.index.values.ravel())
        # find the houses with the whole year data
        full_year = map(lambda x: df[df.index==x].shape[0],unique.tolist())
        indices = [i for i, x in enumerate(full_year) if x == 8760]
        best_indices = map(lambda x: unique[x],indices)
        
        d = {}
        for appliance in df.columns.tolist():
            started = 0
            if best_indices == []:
                for i in unique:
                    
                    if started == 0:
                        
                        d[str(appliance)] = df[[str(appliance)]][df[str(appliance)].index == i]
                        started = 1
                        dfindex = d[str(appliance)].index
                    else:
        
                        d[str(appliance)][str(i)] = pd.Series(df[str(appliance)][df[str(appliance)].index == i].values,index=dfindex)
                
           
                d[str(appliance)]=d[str(appliance)].rename(columns = {str(appliance):str(dfindex[0])})
                d[str(appliance)].reset_index(drop=True, inplace=True)
            else:
                for i in best_indices:
                    
                    if started == 0:
                        
                        d[str(appliance)] = df[[str(appliance)]][df[str(appliance)].index == i]
                        started = 1
                        dfindex = d[str(appliance)].index
                    else:
        
                        d[str(appliance)][str(i)] = pd.Series(df[str(appliance)][df[str(appliance)].index == i].values,index=dfindex)
                
                d[str(appliance)]=d[str(appliance)].rename(columns = {str(appliance):str(dfindex[0])})
                d[str(appliance)].reset_index(drop=True, inplace=True)
        return d
        
    
    def split(self,d,portion,timeframe, portion_houses=None, option=None):
        '''
        Parameters: d = dictionary, portion 0.5 - 0.9, timeframe 1-8760
        
        Return: x_train,x_test dictionarys containing dataframes of all the appliances within the timeframe.
        '''
       
        if portion > 0.9:
                print 'holy shit thats a high value of portion, %s' %(portion)
        x_train = {}
        x_test = {}
        timeframe = range(timeframe)
        key = d.keys()[0]
        columns = d[key].columns.tolist()
        train_list  = timeframe[int(len(timeframe) * 0.0):int(len(timeframe) * portion)]
        test_list = timeframe[int(len(timeframe) * portion):int(len(timeframe) * 1.0)]
        '''
        start_day_2014 = 3 # thursday
        if option == 'week':
            for key in d.keys():
                x_train[key] = d[key].loc
        '''
        if portion_houses != None:
            houses  = columns[int(len(columns) * 0.0): int(len(columns) * portion_houses)]
            
        for key in d.keys():
        
            if portion_houses != None:
                x_train[key] = d[key].loc[train_list,houses]
                x_test[key] = d[key].loc[test_list,houses]
            else:
                x_train[key] = d[key].loc[train_list,:]
                x_test[key] = d[key].loc[test_list,:]
                
        return x_train,x_test
