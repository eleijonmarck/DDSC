#!/usr/bin/env python2.7
# -*- coding: utf-8 -*-

# --- Program internal modules -------------------------------------------
from __future__ import division
from dsc import DSC
import numpy as np
import pandas as pd
import click
import pickle

############ plotting
import matplotlib.pyplot as plt
plt.style.use('ggplot')
## plotting basis functions
import matplotlib.cm as cm
# plotly
import plotly.plotly as py
from plotly.graph_objs import *
###############################
# plots directory
figure_directory= '../../../doc/thesis/figures/'
# change so that all figures have font size defulat as 22
plt.rcParams.update({'font.size': 18})
# figure size always the same
plt.figure(figsize=(16,12))

# --- Locally installed modules -----------------------------------------
from reader import Reader

class Plotter(object):
    def __init__(self,n,t,acc,data):
        self.n = n
        self.T = t
        self.acc = acc
        self.data = data

    def appliances(self, x_train, x_test, x_test_use, x_predict):
        # row and column sharing
        f, ((ax1, ax2, ax3, ax4, ax5, ax6)) = plt.subplots(6, 1, sharex='col', sharey='row', figsize=(16,18))
        ## piechart
        f2, ((axes1, axes2)) = plt.subplots(2, 1, sharex='col', sharey='row', figsize=(16,18))
        pie_chart_true = []
        pie_chart_pred = []

        x = range(x_train[x_train.keys()[0]].shape[0])
        plt.rcParams.update({'font.size': 15})

        y = np.asarray(x_test_use)[0:,0]
        ax1.plot(x, y, color='b',label='Actual energy')
        y=[-1]*x_train[x_train.keys()[0]].shape[0]
        ax1.plot(x, y, color='r', label='Predicted')
        #ax1.set_ylim([0,2])
        ax1.set_ylabel('Whole Home')
        ax1.legend()
        ##
        y = np.asarray(x_test[x_test.keys()[0]])[0:,0]
        pie_chart_true.append(y.sum())
        ax2.plot(x, y, color='b')
        y = np.asarray(x_predict[0])[0:,0]
        ax2.plot(x , y, color='r')
        #ax2.get_yaxis().set_visible(False)
        ax2.set_ylabel('Refrigerator')
        pie_chart_pred.append(y.sum())
        ##
        y = np.asarray(x_test[x_test.keys()[1]])[0:,0]
        pie_chart_true.append(y.sum())
        ax3.plot(x, y, color='b')
        y = np.asarray(x_predict[1])[0:,0]
        ax3.plot(x,y, color='r')
        #ax3.get_yaxis().set_visible(False)
        ax3.set_ylabel('Dishwasher')
        pie_chart_pred.append(y.sum())

        ##
        y = np.asarray(x_test[x_test.keys()[2]])[0:,0]
        pie_chart_true.append(y.sum())
        ax4.plot(x, y, color='b')
        y = np.asarray(x_predict[2])[0:,0]
        ax4.plot(x,y, color='r')
        #ax4.get_yaxis().set_visible(False)
        ax4.set_ylabel('Furnace')
        pie_chart_pred.append(y.sum())

        ##
        y = np.asarray(x_test[x_test.keys()[3]])[0:,0]
        pie_chart_true.append(y.sum())
        ax5.plot(x, y, color='b')
        y = np.asarray(x_predict[3])[0:,0]
        ax5.plot(x,y, color='r')
        #ax5.get_yaxis().set_visible(False)
        ax5.set_ylabel('Air')
        pie_chart_pred.append(y.sum())
        ##
        y = np.asarray(x_test[x_test.keys()[4]])[0:,0]
        pie_chart_true.append(y.sum())
        ax6.plot(x, y, color='b')
        y = np.asarray(x_predict[4])[0:,0]
        ax6.plot(x,y, color='r')
        #ax6.get_yaxis().set_visible(False)
        ax6.set_ylabel('Others')
        ax6.set_xlabel('Hours')
        pie_chart_pred.append(y.sum())

        if self.data == 0:
            f.savefig(figure_directory+'normal_appliances_'+str(self.n)+'_'+str(self.T) + '.png')
        elif self.data == 1:
            f.savefig(figure_directory+'end_appliances_'+str(self.n)+'_'+str(self.T) + '.png')
        elif self.data == 2:
            f.savefig(figure_directory+'days_appliances_'+str(self.n)+'_'+str(self.T) + '.png')

        ## pie-charts
        labels = x_test.keys()
        self.pie_chart(axes1,pie_chart_true,labels)
        axes1.set_title('True usage')
        self.pie_chart(axes2,pie_chart_pred,labels)
        axes2.set_title('Predicted usage')
        axes2.text(0.95, 0.01, 'Accuracy of ' + str(round(self.acc[0],1)),
        verticalalignment='center', horizontalalignment='right',
        transform=axes2.transAxes,
        color='black', fontsize=15)

        if self.data == 0:
            f2.savefig(figure_directory+'normal_pie_chart_'+str(self.n)+'_'+str(self.T) + '.png')
        elif self.data == 1:
            f2.savefig(figure_directory+'end_pie_chart_'+str(self.n)+'_'+str(self.T) + '.png')
        elif self.data == 2:
            f2.savefig(figure_directory+'days_pie_chart_'+str(self.n)+'_'+str(self.T) + '.png')


    def pie_chart(self, subplot, pie_chart, labels):
        # The slices will be ordered and plotted counter-clockwise.
        ## --- Plotting the true-piechart
        pie_chart_sum = sum(pie_chart)
        pie_chart = map(lambda x: x/pie_chart_sum,pie_chart)
        cmap = plt.cm.prism
        colors = cmap(np.linspace(0., 1., len(pie_chart)))
        colors = ['yellowgreen', 'gold', 'lightskyblue', 'lightcoral']
        pie_wedge_collection = subplot.pie(pie_chart, colors=colors, labels=labels, labeldistance=1.05);

        for pie_wedge in pie_wedge_collection[0]:
            pie_wedge.set_edgecolor('white')
        # Set aspect ratio to be equal so that pie is drawn as a circle.



@click.command()
@click.option('--t', default=14,help='Timeframe; note that half will be test')
@click.option('--ph',default=None, help='Portion of houses to investigate')
def main(t,ph,n):
    '''
    Program to train disaggregation using Sparse Coding
    '''
#   read dataset
    datasets = [0,1,2]
    for data in datasets:
        dataset = ['cleanpecanhour2014','weekendpecanhour2014','weekdayspecanhour2014']
        reader = Reader(dataset[data])
        ##returning a datafile pandasObject
        df = reader.dataParser()

        print "parsed the data"
        # returns a dictionary of all of the appliances
        d = reader.format_data(df,other=True)
        print "formated data"
        portion = 0.5
        factor_n_t = 0.1 # heuristically determined

        timeframes = [14,30,60]
        timeframes = [x*24 for x in timeframes]
        alphas = [0.00001, 0.00001, 0.000001]
        portion = 0.5
        # Good values (t,n,alpha)
        # (14,40, alpha = 0.0001)
        # (336,800, alpha = 0.00001)
        # (720,,1400, alpha = )
        for timeframe, alpha in zip(timeframes,alphas):
            n = int(factor_n_t*timeframe)
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
            steps = 10 # steps must be higher than k
            n_components = n

            # Sparse Coding pre_training
            dsc = DSC(train_set,train_sum,alpha,epsilon,rp,steps,n_components,m,T,k)
            print "started the pre-training"
            A_list,B_list = dsc.pre_training(x_train.values())
            print "done pre_training"
            # Didscriminative Disaggregation training
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
            acc_nndsc, acc_dddsc = dsc.get_accuracy_plot()
            err_nndsc, err_dddsc = dsc.get_error_plot()
            # plotting acc/err
            a_nndsc, a_ddsc = dsc.get_a()
            b_nndsc, b_ddsc = dsc.get_b()

            hours = timeframe/2
            plot_it = Plotter(n,hours,acc,data)

            plot_it.appliances(x_train, x_test, x_test_use, x_predict)

if __name__ == '__main__':
    main()
