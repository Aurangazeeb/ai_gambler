from sklearn.preprocessing import OrdinalEncoder
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score
import time
import os

class Player():
    
    def __init__(self, n_games, shop_obj, col_series = True, \
                 n_features = 1, is_multivariate = False, is_multi_additive = True):
        
        '''
        The player has characteristic ngames, shop object while nfeatures is optional.
        Since single time series is not parallely stacked nfeatures is always = 1
        
        '''
                 
#                  colormapper, nummapper, patencoder, shopname, driver, current_server):
        self.n_games = n_games
        self.playshop = shop_obj
        self.col_series = col_series
        self.n_features = n_features
        
        self.is_multivariate = is_multivariate
        self.is_multi_additive = is_multi_additive
        
        self.playername = 'P_' + str(n_games)
        self.is_classifier_fitted = False #model reset
        self.is_seqmodel_fitted = False #model reset
        self.is_educated = False
        self.model_history = pd.DataFrame() #error log
        self.acc_log = pd.DataFrame(index = [0]) #accuracy log
        self.acc_log_folder = self.playshop.name + '_acc_logs'
        self.acc_log_prefix = self.acc_log_folder + '/acc_log_'
        self.color_encoder = OrdinalEncoder(dtype = int) #quick fetch
        
    def educate(self, seqmodel, clsmodel, **seqmodel_vals):
        self.is_educated = True
        self.seqmodel =seqmodel
        self.classifier = clsmodel
        self.seqmodel_vals = seqmodel_vals
        
    class NotEducatedError(Exception):
        pass
        
    def learn(self, gamedata, num_classes_cls, yhat_choice = 'default'):
        if self.is_educated:
            self.series = gamedata
            self.prep_input(self.series, \
                            self.n_games, \
                            self.seqmodel, \
                            self.classifier, \
                            n_features = self.n_features, \
                            multivar =  self.is_multivariate, \
                            isadditive =  self.is_multi_additive, \
                            **self.seqmodel_vals)
            self.fit_classify(self.Xtensor, self.ytensor, num_classes_cls, yhat_choice, self.col_series, test_cls = True)
        else:
            raise self.NotEducatedError('Educate the player first !')
            
    def __str__(self):
        return self.playername + ' Player'
                
            
        
    def predict_next(self, make_log = True, play_reverse = False, wait4next_bttn = 1, fetch_till = 15, sleeptime = 1, fetch_from = '2:50'):
#         if driver is not None:
#             self.driver = driver
        series = self.fetch_latest(col_series = self.col_series, \
                                   play_reverse = play_reverse, \
                                   wait = wait4next_bttn, \
                                   fetch_till = fetch_till, \
                                   sleeptime = sleeptime, \
                                   fetch_from = fetch_from)
        prepped_series = self._prep_newdata(series)
        seqpred = self.seqmodel.predict(prepped_series)
        seqpred = seqpred.reshape(-1,1)
        finalpred = self.classifier.predict(seqpred)[0]
        if self.col_series:
            finalpred = self.series2colors(finalpred)[0][0]
        self.finalpred = finalpred
        print()
        print('*' * 70)
        print()
        if self.col_series:
            finalpred = self.finalpred.upper()
            print(f'By watching {self.n_games} games, I think the next color is {finalpred} ...!')
        else:
            print(f'By watching {self.n_games} games, I think the next number \
            is {finalpred} and color is {self.playshop.nummap[finalpred].upper()}...!')
        print()
        print('*' * 70)
        if make_log:
            self._make_acc_log(self.col_series)
        
    def predict_series(self, colors):
        if len(colors) / self.n_games == 1:
            colorseries = self.colors2series(colors)
            colortensor = self._prep_newdata(colorseries)
            seqpred = self.seqmodel.predict(colortensor)
            seqpred = seqpred.reshape(-1,1)
            stackpred = self.classifier.predict(seqpred)
            color_pred = self.series2colors(stackpred)
            print(f'By watching {self.n_games} games, I think the next color is {color_pred[0].upper()} ...!')
        else:
            print(f"I'm watching {self.n_games}... Color list length does not match it ...!")
        
        

## predict realtime

#     def _fetch_pages(self, numrows, numentry = 10, sleeptime = 1):
#         if numrows <= numentry:
#             numpages = 2
#         elif numrows % numentry == 0:
#             numpages = numrows // numentry
#         else:
#             numpages = numrows // numentry + 1
#         time_needed = self.playshop._time_taken(numpages, sleeptime)
#         if self.playshop._time_left() < (time_needed + 5):
#             raise Exception('Need more time to fetch... Try reducing datapoints...')
#         timenow = self.playshop._time2str(self.playshop._time_left())
#         print(f'Started fetching {numpages} pages .. at {timenow}')
#         pages = pd.DataFrame(self.playshop._pull_pages(numpages, sleeptime))
#         if self.playshop._quality(pages) == 'nice':
#             return pages
#         else:
#             raise Exception('Consider refetching.. Data has bad quality...!')
        


    def fetch_latest(self, n_data = None, col_series = True, play_reverse = False, \
                     wait = 2, fetch_till = 15, fetch_from = '2:50',sleeptime = 1):
        print(f'You are on server : {self.playshop.servers[self.playshop.current_server - 1]} !')
        self.playshop.reload(wait)
        if n_data is None:
            n_data = self.n_games
#         mintime = fetch_till
#         maxtime = self.playshop._str2time(fetch_from)
        
#         timerem = self.playshop._time_left()
#         #fetch gamedf silently
#         if (timerem > mintime) & (timerem < maxtime):
#             self.latest_df = self.playshop._fetch_pages(n_data, numentry = 10, sleeptime = sleeptime)
#             if self.latest_df.price[0] == 0:
#                 self.latest_df = self.latest_df.tail(-1)
#             self.latest_df['color'] = [self.playshop.nummap[num] for num in self.latest_df['number']]
#             colorarray = self.latest_df['color'].values 
#             self.latest_df['colorseries'] = self.color_encoder.fit_transform(colorarray.reshape(-1,1))
#             print()
#             print('Latest data stored in latest_df attribute...!')
#         else:
#             timerem = self.playshop._time2str(timerem)
#             raise Exception(f'Time range outside the limits for fetching.. Time Now : {timerem: >5}s')
        
        self.latestdf = self.playshop.get_gamedf_v2(n_data, entry_time = self.playshop.now(), \
                                                 from_server = self.playshop.current_server, \
                                                 wait4next_bttn = wait, \
                                                 fetch_till = fetch_till, \
                                                 fetch_from = fetch_from, sleeptime = sleeptime)
    
        #select n_data from fetched df
        if col_series:
            gameseries = self.latestdf['colorseries'].values[:n_data]
            if play_reverse:
                gameseries = np.array(list(reversed(gameseries)))
        else:
            gameseries = self.latestdf['number'].values[:n_data]
            if play_reverse:
                gameseries = np.array(list(reversed(gameseries)))
        return gameseries
    
    def _prep_newdata(self, newdata):
        '''
        Should be only used for making next prediction
        
        '''
        newdata = newdata.reshape(-1, self.n_games, self.n_features)
        newtensor = tf.constant(newdata)
        return newtensor
    
    
            
            ###### Learning
            
            

        # split a univariate sequence
    def _split_sequence(self, sequence, n_steps):
        X, y = list(), list()
        for i in range(len(sequence)):
            # find the end of this pattern
            end_ix = i + n_steps
            # check if we are beyond the sequence
            if end_ix > len(sequence)-1:
                break
            # gather input and output parts of the pattern
            seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
            X.append(seq_x)
            y.append(seq_y)
        return np.array(X), np.array(y)

    def _split_sequences(self, sequences, n_steps):
        X, y = list(), list()
        for i in range(len(sequences)):
            # find the end of this pattern
            end_ix = i + n_steps
            # check if we are beyond the dataset
            if end_ix > len(sequences):
                break
            # gather input and output parts of the pattern
            seq_x, seq_y = sequences[i:end_ix, :-1], sequences[end_ix-1, -1]
            X.append(seq_x)
            y.append(seq_y)
        return np.array(X), np.array(y)

            
        ##make sequence model
                
    def init_tensors(self, \
                      n_epochs = 500, \
                      val_split = 0.2, \
                      verbose = 0):
        
        '''
        Initialize tensors for sequence model
        
        '''

        self.n_epochs = tf.constant(n_epochs, dtype = tf.int64)
        self.val_split = tf.constant(val_split)
        self.verbose = tf.constant(verbose)
        print('Tensors initialized !')
        
#     def make_dataset(self, series, n_steps, n_features):
#         '''
#         Returns Xtensor and ytensor attributes
        
#         '''
#         X, y = self._split_sequence(series, n_steps)
#         X_reshaped = X.reshape(X.shape[0], X.shape[1], n_features)
#         Xtensor, ytensor = tf.constant(X_reshaped), tf.constant(y)
#         return Xtensor, ytensor


    def make_dataset(self, series, n_steps, n_features, multivar = False, isadditive = True):
        '''
        Returns Xtensor and ytensor attributes
        
        '''
        if not multivar:
            X, y = self._split_sequence(series, n_steps)
        else:
            X, y = self._split_sequences(n_steps, series, isadditive)
        X_reshaped = X.reshape(X.shape[0], X.shape[1], n_features)
        Xtensor, ytensor = tf.constant(X_reshaped), tf.constant(y)
        return Xtensor, ytensor
        
    def _track_error(self, modelname, history, save = True, print_acc = False):
        buffer_dict = {}
        buffer_dict['model_name'] = modelname
        for key, value in history.items():
            buffer_dict['final_' + key] = value[-1]
        if save:
            row_df = pd.DataFrame(buffer_dict, index = [0])
            self.model_history = pd.concat([model_history, row_df])
            self.model_history.reset_index(drop =True, inplace = True)
        print()
        print('-' * 70)
        print(f'History saved for {modelname}')
        print('-' * 70)
        if not print_acc:
            print('\nFinal Train Loss : {:>10.3f} \
                \n\nFinal  Val Loss : {:>10.3f}' \
                .format(buffer_dict['final_loss'], \
                    buffer_dict['final_val_loss']))
        else :
            print('\nCurrent Mean Acc : {:>10.3f} \
                \n\nCurrent Mean Val Acc : {:>10.3f}' \
                .format(buffer_dict['final_accuracy'], \
                    buffer_dict['final_val_accuracy']))
        print()
           
        
        #seqmodel helpers
        
    def make_colorseries_preds_df(self, pred_model, Xtensor, ytensor):
        predsdf = pd.DataFrame()
        preds = pred_model.predict(Xtensor)
        series = ytensor.numpy()
        predsdf['y'] = ytensor
        
        #series2colors => list
        predsdf['y_color'] = self.series2colors(series)
        
        predsdf['yhat'] = preds
        predsdf['yhat_color'] = self.series2colors(preds)
        predsdf['yhat_bankers'] = np.round(predsdf['yhat'])
        predsdf['yhat_int'] = predsdf['yhat'].map(int)
        return predsdf

    def make_numseries_preds_df(self, pred_model, Xtensor, ytensor):
        predsdf = pd.DataFrame()
        
        preds = pred_model.predict(Xtensor)
        predsdf['y'] = ytensor
        predsdf['y_color'] = predsdf['y'].map(self.playshop.nummap)
        predsdf['yhat'] = preds
        predsdf['yhat_color'] = predsdf['yhat'].map(self.playshop.nummap)
        predsdf['yhat_bankers'] = np.round(predsdf['yhat'])
        predsdf['yhat_int'] = predsdf['yhat'].map(int)
        return predsdf
        
        

    def _showacc(self, splitdataset, return_acc = False):
        '''
        Can return train acc and test acc tuple
        '''
        preds = self.classifier.predict(splitdataset[0]), self.classifier.predict(splitdataset[1])
        acc = accuracy_score(splitdataset[2], preds[0]) * 100 , accuracy_score(splitdataset[3], preds[1]) * 100
        if return_acc:
            return acc
        print(f'\nTrain Acc : {acc[0]: 30.3f} %\n\n Test Acc : {acc[1]: 30.3f} %\n')
        
    
    def test_stack(self, X, y, final_estimator, *init_estimators, splitratio = 0.2):
        '''
        Generates a fitted Classifier by fitting on xtrain and ytrain
        
        '''
        datavals = self._prep_data2testclassify(X, y, splitratio)
        classifier = self._set_classifier_stack(final_estimator, *init_estimators)
        xtrain, ytrain = datavals[0], datavals[2]
        if xtrain.ndim == 1:
            xtrain = xtrain.values.reshape(-1,1)
        classifier.fit(xtrain, ytrain)
        self._showacc(datavals, return_acc = False)
        return classifier

    
        
        ############### Prediction wrapper
        
        
        
        # wrapper method
    def prep_input(self, series, n_steps, seqmodel, clsmodel, n_features, multivar, isadditive, **seqmodel_vals):
        
        
        #store n_steps for fetching realtime data
        self.seqmodel_fitted = False
        self.classifier_fitted = False

        #Make dataset
        self.Xtensor, self.ytensor = self.make_dataset(series, n_steps, n_features, multivar, isadditive)
#         self.Xtensor, self.ytensor = self.make_dataset(series, n_steps, n_features)
        print()
        print('Generated Xtensor and ytensor !')

        '''
        init_vals = (n_lstm_units, n_epochs, n_features, val_split)
        
        '''
#         self.n_steps = n_steps #tensor
#         self.n_features = n_features
#         self.n_lstm_units = 
        self.init_tensors(**seqmodel_vals)
        
        #get the final classifier stack
    
    
    def fit_sequence(self, Xtensor, ytensor, col_series = True, save_error = False):
        '''
        Generates the color predsdf. Supports number series
        '''
        self.is_seqmodel_fitted = True
        # get sequence model
        self.train_history = self.seqmodel.fit(Xtensor, ytensor, \
                                               epochs = self.n_epochs, \
                                               verbose = self.verbose, \
                                               validation_split = self.val_split)
        self._track_error(self.playshop.name, self.train_history.history, save_error)
#         self.seqmodel(self.playshop.name + '_model')
        print('\nSequence modeling complete ...!')
        print()
        
        #get predsdf
        if col_series :
            self.seqpreds_df = self.make_colorseries_preds_df(self.seqmodel, Xtensor, ytensor)
        else:
            self.seqpreds_df = self.make_numseries_preds_df(self.seqmodel, Xtensor, ytensor)
        print('\nSequence model predictions saved in seqpreds_df...!')
        print()
        
        
        
    def _prep_data2testclassify(self, X, y, splitratio, return_vals = True):
#         X, y = gamedata['yhat'], gamedata['colorseries']
        xtrain, xtest, ytrain, ytest = train_test_split(X, y, test_size = splitratio, random_state = 0)
        if xtrain.ndim == 1:
            xtrain, xtest = xtrain.values.reshape(-1,1), xtest.values.reshape(-1,1)
#             ytrain, ytest = ytrain.reshape(-1,1), ytest.reshape(-1,1)
        datavals = (xtrain, xtest, ytrain, ytest)
        if return_vals:
            return datavals
        
            
    def test_fit_classifier(self, X, y, num_classes):
    
    
        '''
        Supports number series

        '''
#         #prepare the predsdf for classification
#         if col_series:
#             X, y = self.seqpreds_df['yhat'], self.seqpreds_df['y']
#             print('X is yhat and y is y itself ...!')
#         else:
#             X, y =self.seqpreds_df['yhat_num'], self.seqpreds_df['y_num']
#             print('X is yhat_num and y is y_num')
            
        # returns xtrain and xtest reshaped
#         datavals = self._prep_data2testclassify(X, y, num_classes)
        train_acc, test_acc = [], []
        stratKfold = StratifiedKFold(n_splits = num_classes)
        for train_indx, test_indx in stratKfold.split(X, y):
            xtrain, xtest = X[train_indx], X[test_indx]
            ytrain, ytest = y[train_indx], y[test_indx]
        
            if xtrain.values.ndim == 1:
                xtrain = xtrain.values.reshape(-1,1)
                xtest = xtest.values.reshape(-1,1)
            
            self.classifier.fit(xtrain, ytrain)
            preds = self.classifier.predict(xtrain), self.classifier.predict(xtest)
            train_acc.append(accuracy_score(ytrain, preds[0]))
            test_acc.append(accuracy_score(ytest, preds[1]))
        
        print(f'\nClassifier test fit performance on training set (CV = {num_classes}) :')
        print()
        print('-' * 70)
        print(f'Mean Train Acc on : {np.mean(train_acc): >5.3f}') 
        print()
        print(f'Mean Test Acc on : {np.mean(test_acc): >5.3f}')
        print('-' * 70)
        
        
        
        
        '''
        Final color classification
        '''
        
        
        
    ### final classification stage
    
    
        
    def _make_final_preds_df(self, finalpreds, labels, col_series = True):
        
        '''
        Final color classification helper
        '''
        predsdf = pd.DataFrame()
        predsdf['y'] = labels
        if col_series:
#             predsdf['yhat_cls'] = finalpreds
            predsdf['yhat'] = self.series2colors(finalpreds)
            
        else:
            predsdf['yhat'] = finalpreds
            predsdf['yhat_color'] = [self.playshop.nummap[pred] for pred in finalpreds] 
            
        return predsdf
    
        
    def classify(self, X, labels, col_series = True):
        '''
        Used to predict with finalized classifer by
        using fit_classifier several times.
        
        Returns the final preds df
        '''
        if(X.ndim == 1):
            X = X.values.reshape(-1,1)
        finalpreds = self.classifier.predict(X)
        predsdf = self._make_final_preds_df(finalpreds, labels, col_series)
        return predsdf
    
    
    def _make_acc_log(self, col_series = True):
#         bufferdf = pd.DataFrame(index = [0])
        day, month, date, timestr, year = time.ctime().split()
        self.acc_log['time'] = '-'.join([year, month, date, day, timestr])
        self.acc_log['final_acc'] = round(self.finalacc, 2)
        self.acc_log['n_datapoints'] = len(self.series)
        self.acc_log['n_games'] = self.n_games
        self.acc_log['n_epochs'] = self.n_epochs.numpy()
        self.acc_log['val_split'] = round(self.val_split.numpy(), 2)
#         self.acc_log['seqmodel'] = self.seqmodel
        self.acc_log['classifier'] = self.classifier
        if col_series:
            self.acc_log['color_preds'] = self.finalpred
            self.acc_log['num_preds'] = 'na'
        else:
            self.acc_log['num_preds'] = self.finalpred
            self.acc_log['color_preds'] = self.playshop.nummap[self.finalpred]
        act_color = input('Input actual color :')
        self.acc_log['act_color'] = act_color
        act_number = input('Input actual number :')
        self.acc_log['act_number'] = act_number
                                        
        savetime_format = '__'.join([year, month, date, day])
        savefile_name = self.acc_log_prefix + savetime_format + '.csv'
        if os.path.exists(self.acc_log_folder):
            if os.path.exists(savefile_name):
                saved_acc_log = pd.read_csv(savefile_name)
                self.acc_log = pd.concat([self.acc_log, saved_acc_log], sort = True) #old bottom
        else:
            os.mkdir(self.acc_log_folder)
        self.acc_log.to_csv(savefile_name, index = False)
        print()
        print(f'Accuracy log is stored in {savefile_name} and acc_log attribute !')
        print()
    
    def final_accuracy(self, X, y):
        if X.ndim == 1:
            X = X.values.reshape(-1,1)
        finalpred = self.classifier.predict(X)
        self.finalacc = accuracy_score(y, finalpred) * 100
        print(f'Final classifier accuracy is : {self.finalacc:10.3f} %')
        print()
        
        
    
    def fit_classifier(self, X, y, label_col, col_series = True):
        '''
        X - should be pandas series
        '''
        self.is_classifier_fitted = True
        if X.ndim == 1:
            X = X.values.reshape(-1,1)
        self.classifier.fit(X, y)
        print()
        print('Classifier model is fit on whole data ... !')
        self.finalpreds_df = self.classify(X, label_col, col_series)
        print()
        print('-' * 70)
        print('Prediction saved in finalpreds_df...!')
        print('-' * 70)
        print()
        self.final_accuracy(X, y)
        
        
    
        
        
        #### Combined fit and predict
    
    '''
    This method can immediately follow prep_input
    '''
    def fit_classify(self, Xtensor, ytensor, num_classes, response_var = 'default', col_series = True, test_cls = True):
        '''
        This method can immediately follow prep_input
        
        '''
        if (not self.is_classifier_fitted) & (not self.is_seqmodel_fitted):
            #generates seqpredsdf and sets up classifier object
            if not self.is_seqmodel_fitted: 
                self.fit_sequence(Xtensor, ytensor, col_series)
            y = self.seqpreds_df['y']
            if response_var == 'default':
                X = self.seqpreds_df['yhat']
            elif response_var == 'bankers':
                X = self.seqpreds_df['yhat_bankers']
            elif response_var == 'int':
                X = self.seqpreds_df['yhat_int']
            if test_cls:
                self.test_fit_classifier(X, y, num_classes)
            colorlabels = self.seqpreds_df['y_color']
            self.fit_classifier(X, y, colorlabels, col_series)
            print('Prediction model is ready... (seqmodel + classifier) !')
            print()
        else:
            print('Model is already prepared...!')
        
    
    def _prep_series(self, series, n_steps, n_features):
        if series.size // n_steps == 1 :
            prepped_series = series.reshape(-1, n_steps, n_features)
        else:
            X, y = self._split_sequence(series, n_steps)
            prepped_series = series.reshape(X.shape[0], X.shape[1], n_features)
        series_tensor = tf.constant(prepped_series)
        return series_tensor
    
    ################################
    
    def series2colors(self, series):
        series = series.reshape(-1,1)
        colors = self.playshop.color_encoder.inverse_transform(series)
        return colors
        
    def colors2series(self, colors):
        '''
        Input colors as list
        '''
        colors = np.array(colors).reshape(-1,1)
        series = self.playshop.color_encoder.transform(colors)
        return series
    
    def reset_models(self):
        self.is_classifier_fitted = False
        self.is_seqmodel_fitted = False
        