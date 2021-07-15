import warnings
warnings.filterwarnings('ignore')
# for model
import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import regularizers, layers
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.optimizers import schedules, SGD, Adam, Nadam
from tensorflow.keras.layers import Dense, Dropout, GRU, GaussianNoise, TimeDistributed
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.initializers import Constant, Orthogonal
#from tensorflow.keras.utils import plot_model
from IPython import display, get_ipython
get_ipython().run_line_magic('matplotlib', 'inline')

# for data
import pickle
import pandas as pd
from glob import glob
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import csv
import utils
from sklearn.preprocessing import MinMaxScaler, RobustScaler

# load .pkl form data_path and split it into 4 subset: x_train, y_train, x_test, y_test
def loading_data(data_path, split): # passive(model_learning_stage)
    print("Begin loading_data at {}".format(datetime.now()))
    with open(data_path, 'rb') as file:
        data = pickle.load(file)
    loaded_data = np.array(data)
    # print("> loaded_data.shape = {}".format(loaded_data.shape))    
    # split data into data_type
    split_position = int(len(loaded_data)*(1-split))
    x_test, y_test = loaded_data[split_position:, 0, :], loaded_data[split_position:, 1, :]
    x_train, y_train = loaded_data[:split_position, 0, :], loaded_data[:split_position, 1, :]
    return x_train, y_train, x_test, y_test

# randomly take a bar from test_data
def generate_prompt(prompt_path, ans_path, shape):
    print('Begin generate_prompt at: {}'.format(datetime.now()))
    print('> prompt_path: {}'.format(prompt_path))
    print('> ans_path: {}'.format(ans_path))

    # load data on the path
    # load bar1
    with open(prompt_path, 'rb') as file:
        all_prompt = pickle.load(file)
        # all_prompt's size: song_num * 2 * bar_size * 4
        # bar_size: 25 (all_prompt[prompt_index][0]) or 45 (all_prompt[prompt_index][1])
    # load ans (the entire song)
    with open(ans_path, 'rb') as file:
        all_ans = pickle.load(file)
        # all_ans's size: song_num * bar_num * 45 * 4
    
    # randomly pick a song from the test data
    prompt_index = np.random.randint(len(all_prompt))
    print('> prompt_index (song index) =', prompt_index)

    temp_prompt = all_prompt[prompt_index][0][:shape[-2]]
    prompt_shape = (-1, shape[-2], shape[-1])
    prompt = np.reshape(temp_prompt, prompt_shape)
    print('> prompt.shape (bar num of prompt song) =', prompt.shape)

    bar_num = prompt.shape[0]
    print('> bar_num =', bar_num)
    
    return bar_num, prompt, all_ans[prompt_index]


class music_generation(object):
    ########################################
    # initialize
    ########################################
    def __init__(self, artist_name, epochs, model_details):
        # Setting parameters
        self.artist_name = artist_name

        # 1) data shape
        self.data_path = 'data/all_pairs_{}.pkl'.format(artist_name)
        self.train_val_split = 0.2 # 0.2 means "use 20% of the data for validation"
        self.train_test_split = 0.2 # 0.3 means "use 30% of the data for test"
        # 2) model structure
        self.model_input_data_shape = (45, 4)
        self.model_output_data_shape = (45, 4)
        self.GRU_num, self.time_dis, self.bn, self.L2, self.input_note_num, self.seq_len, self.optimizer = model_details
        self.dropout_rate = 0.08
        self.hidden_size = 16
        # 3) compile model
        self.loss = 'mean_squared_error'
        if self.optimizer=='nadam': self.optimizer = Nadam()
        elif self.optimizer=='nadam0.01': self.optimizer = Nadam(learning_rate=0.01)
        else: self.optimizer = Adam(learning_rate=0.01)
        # 4) trainging
        self.batch_size = 256
        self.epochs = epochs
        
        self.record = "{}_{}_GRU{}".format(datetime.now().strftime("%m%d%H%M"), self.artist_name, 
                                            #   self.hidden_size, self.dropout_rate, 
                                              self.GRU_num)
        if self.L2: self.record = self.record+'_L2'
        if self.bn: self.record = self.record+'_BatchNorm'
        if self.time_dis: self.record = self.record+'_timedis'

        self.checkpoint_path = "ckpt/"+self.record+".h5" # _ckpt-epoch={epoch:02d}-loss={loss:.2f}
        self.checkpoint_dir = os.path.dirname(self.checkpoint_path)
        self.ckp = None
        self.training_history_path = 'train/{}'.format(self.record)

        # 5) genration
        self.data_generation_path = "data_generation/"+self.record
        self.result_path = 'result/'+self.record
        self.predictions, self.predictions_raw, self.prompt = 0, 0, 0

        # # Working
        # # load training & testing data
        # self.x_train, self.y_train, self.x_test, self.y_test = loading_data(data_path=self.data_path, split=self.train_test_split)
        # self.model_input_data_shape = (self.x_train.shape[-2], self.x_train.shape[-1])
        # print('> self.model_input_data_shape', self.model_input_data_shape)

        # Working_new
        # load training & testing data
        print("Begin loading_data (music_model) at {}".format(datetime.now()))

        with open(self.data_path, 'rb') as file:
            data = pickle.load(file)
        x = []
        y = []
        for training_data in data:
            x.append(training_data[0][:self.input_note_num])
            y.append(training_data[1][:int(self.seq_len/4)])
        x = np.reshape(x, (-1, self.input_note_num, 4))
        y = np.reshape(y, (-1, self.seq_len))
        # print('> x.shape & y.shape', x.shape, y.shape)

        # split data into training & testing data
        split_position = int(len(data)*(1-self.train_test_split))
        self.x_test, self.y_test = x[split_position:], y[split_position:]
        self.x_train, self.y_train = x[:split_position], y[:split_position]

        self.model_input_data_shape = x.shape[1:]
        self.model_output_data_shape = (int(y.shape[-1]/self.model_input_data_shape[-1]), self.model_input_data_shape[-1])
        self.seq_len = y.shape[-1]
        print('> self.model_input_data_shape', self.model_input_data_shape)
        print('> self.model_output_data_shape', self.model_output_data_shape)
        print('> self.seq_len', self.seq_len)


    # Train the model, and also do validation at the end of each epoch
    def model_learning_stage(self, ckp=None):
        print("Begin model_learning_stage (music_model) at", datetime.now())

        # load model by model_type
        model = self.loading_model(GRU_num=self.GRU_num, time_dis=self.time_dis, bn=self.bn, L2=self.L2)

        # transfer learning
        if ckp!=None: model.load_weights(ckp)

        # start training
        callbacks_list = [
            # keras will save the one and only .h5 ckp file which has min val_loss
            # verbose: once stopped, the callback will print the epoch number
            ModelCheckpoint(filepath=self.checkpoint_path, save_best_only=True,
                            verbose=1, monitor='val_loss', save_weights_only=True),
            # keras will stop of loss hasn'y improve for 50 epochs
            # verbose: once stopped, the callback will print the epoch number
            # patience: number of epochs on which we would like to see no improvement
            EarlyStopping(monitor='loss', verbose=1, patience=50)]
        history = model.fit(self.x_train, self.y_train,
                            epochs = self.epochs,
                            callbacks = callbacks_list,
                            batch_size = self.batch_size,
                            validation_split = self.train_val_split)
        # print("Finish model_learning_stage at", datetime.now())
        
        # output the training history (train/{}.csv)
        with open('{}.csv'.format(self.training_history_path), 'w', newline='') as csvfile:
            fieldnames = history.history.keys()
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerow(history.history)
            
        # visualize loss during training (train/{}.jpg)
        plot2 = plt.figure(2)
        plt.title('Hidden Size: {} | Dropout Rate: {}\nLoss: {}'.format(self.hidden_size, self.dropout_rate, self.loss))
        plt.plot(history.history['loss'], label='train_new_music_model')
        plt.plot(history.history['val_loss'], label='validation_new_music_model')
        plt.legend()
        plt.savefig('{}_new_music_model.jpg'.format(self.training_history_path))
        
        # test_stage
        print("Begin model_inferencing_stage (music_model) at", datetime.now())
        results = model.evaluate(self.x_test, self.y_test)
        print("> test loss, test acc:", results)
        # print("Finish model_inferencing_stage at", datetime.now())

    # Building (add layers & compile) the Network
    def loading_model(self, GRU_num, time_dis, bn, L2): # passive(compiling_model)
        # model setting
        print("Begin building model structure (music_model) at", datetime.now())
        
        model = Sequential(name='sequential_{}_{}_{}'.format(self.hidden_size, self.epochs, self.dropout_rate))
        # Input layer: 4 dimension in each timesteps & 45 timeseeps
        model.add(keras.Input(shape=self.model_input_data_shape)) 
        for i in range(GRU_num):
            # GRU layer: return hidden_size output notes on each timesteps
            # every but the first GRU layer
            if i!= 0: 
                model.add(GaussianNoise(0.075))
            # the last GRU layer
            if i==GRU_num-1: 
                model.add(GRU(self.hidden_size, 
                            # return_sequences=True, 
                            kernel_regularizer=regularizers.l2(0.01)
                            ))
            # the middle GRU layer
            else:
                model.add(GRU(self.hidden_size, 
                            return_sequences=True, 
                            kernel_regularizer=regularizers.l2(0.01)
                            ))
            if bn: model.add(layers.BatchNormalization())
        if time_dis:
            model.add(TimeDistributed(Dense(units=self.seq_len, bias_initializer=Constant(value=8))))
        else:
            model.add(Dense(units=self.seq_len, bias_initializer=Constant(value=8)))


        # # debug: output shape of each layer
        # model = keras.Model(inputs = model.input, 
        #                     outputs = [layer.output for layer in model.layers])
        # features = model.predict(self.x_train)
        # for feature in features:
        #     print(feature.shape)

        # compile the model
        model.compile(
            loss=self.loss,
            optimizer=self.optimizer,
            metrics=['accuracy'])

        model.summary()

        return model

    def modify_shape(self, temp_predictions, shape):
        if shape=='output': shape = self.model_output_data_shape
        elif shape=='input': shape = self.model_input_data_shape

        shorter = shape[-2]-len(temp_predictions)
        # fill np.zeros when prediction is shorter than the assigned note num in shape
        if shorter>0: 
            temp_predictions = np.append(temp_predictions, np.zeros((shorter, shape[-1])))
            # reshape again (because checking by size 1*4)
            temp_predictions = np.reshape(temp_predictions, (-1, shape[-1]))
        # cut temp_predictions when prediction is longer than or equals to the assigned note num in shape
        else: 
            temp_predictions = temp_predictions[:shape[-2]]

        # for utils.predictions_to_words()
        temp_predictions = np.reshape(temp_predictions, (1, shape[0], shape[1]))
        return temp_predictions
    
    # modify the prediction to MIDI readable format
    def modify_prediction(self, temp_predictions):
        # Reshape (because checking by size 1*4) and round temp_predictions
        temp_predictions = np.reshape(temp_predictions, (-1, self.model_output_data_shape[-1]))
        temp_predictions = np.round(temp_predictions, 0)
        # # Reshape to output size
        # shorter = self.model_output_data_shape[-2]-len(temp_predictions)
        # # prediction is shorter than output size
        # if shorter>0: 
        #     # fill temp_predictions
        #     temp_predictions = np.append(temp_predictions, np.zeros((shorter, self.model_output_data_shape[-1])))
        #     # reshape again (because checking by size 1*4)
        #     temp_predictions = np.reshape(temp_predictions, (-1, self.model_output_data_shape[-1]))
        # # prediction is longer than output size
        # else: 
        #     temp_predictions = temp_predictions[:self.model_output_data_shape[-2]]

        del_list = []
        # 1.
        # value out of range: Position (1 - 16) | Velocity (1 - 32) | Note On (1 - 128) | Durationn (1 - 64) | None (0)
        for i in range(0, len(temp_predictions)):
            # 3. unreasonable repeat note
            if i==0: repeat = False
            else: repeat = ((temp_predictions[i][0] == temp_predictions[i-1]).all()
                            and(temp_predictions[i][2] == temp_predictions[i-1][2]).all())

            # <0 or >max -> delete the row and add [0, 0, 0, 0] at the end
            trash = ((temp_predictions[i][0]>16) 
                     or (temp_predictions[i][1]>32) 
                     or (temp_predictions[i][2]>128) 
                     or (temp_predictions[i][3]>64) 
                     or ((temp_predictions[i]<=0).any())
                     or repeat)
            if trash: del_list.append(i)
        # delete those value out of range
        temp_predictions = np.delete(temp_predictions, del_list, axis=0)
        # 2.
        # not in order
        for i in range(1, len(temp_predictions)):
            not_in_order = (temp_predictions[i-1][0] > temp_predictions[i][0])
            if not_in_order:
                # delete temp_predictions[i:] = save only temp_predictions[:i]
                temp_predictions = temp_predictions[:i]
                break

        # # add empty events so that the size will be 180*1
        # empty_num = self.model_output_data_shape[0]-len(temp_predictions)
        # temp_predictions = np.append(temp_predictions, np.zeros((empty_num, self.model_input_data_shape[-1])))
        # # reshape temp_predictions
        # # for utils.predictions_to_words()
        # temp_checked_predictions = np.reshape(temp_predictions, self.model_output_data_shape)

        # size: 1*45*4
        return temp_predictions 

    # generate_data with or without prompt; return predictions, predictions_raw, prompt_y
    def generate_data(self, sequence_num, prompt, ans, ckp_name=None, prompt_type='bar'):
        print("Begin generate_data at {}".format(datetime.now()))

        predictions_raw = []
        predictions = []

        # Load the checkpoint
        if ckp_name == None:
            # set the model checkpoint for inferencing state at the latest model checkpoint
            self.ckp = self.checkpoint_path
            print("> the latest checkpoints: ", self.ckp)
        else:
            self.ckp = 'ckpt/'+ckp_name+'.h5' 
            print("> the assigned checkpoints: ", self.ckp)

        # Load model
        model = self.loading_model(GRU_num=self.GRU_num, time_dis=self.time_dis, bn=self.bn, L2=self.L2)
        # loading_model(GRU_num=self.GRU_num)

        # Load weight
        model.load_weights(self.ckp)

        # Load prompt
        self.prompt = ans
        # save bar1 & bar2: raw and checked
        for i in range(2):
            # load prompt
            temp_predictions = prompt[i]
            # save raw prompt
            predictions_raw.append(temp_predictions)
            # modify raw prompt
            temp_checked_predictions = self.modify_prediction(temp_predictions)
            # reshape modified prompt to output shape
            out_predictions = self.modify_shape(temp_predictions=temp_checked_predictions, shape=self.model_output_data_shape)
            # save modified prompt
            predictions.append(out_predictions)


        if prompt_type=='bar':
            # generate output sequence (size: 1*45*4) one by one
            for i in range(0, sequence_num-2, 1):
                # generate the output sequence by the previous output sequence                
                temp_checked_predictions = np.reshape(temp_checked_predictions[:self.model_input_data_shape[0]], 
                                           (1, self.model_input_data_shape[0], self.model_input_data_shape[1]))
                temp_predictions = model.predict(x = temp_checked_predictions) # batch_size=self.batch_size 
                temp_checked_predictions = self.modify_prediction(temp_predictions)
                
                # save the generated sequence: raw and checked
                predictions_raw.append(temp_predictions)
                predictions.append(temp_checked_predictions)
        else: # prompt_type=='song'
            # assign the bar num of generated song
            sequence_num = len(ans)
            # generate prediction
            for i in range(2, sequence_num, 1):
                # reshape modified prompt to input shape
                in_predictions = self.modify_shape(temp_predictions=temp_checked_predictions, shape=self.model_input_data_shape)
                # predict the next raw prediction
                temp_predictions = model.predict(x = in_predictions) # batch_size=self.batch_size 
                # save raw preidiction
                predictions_raw.append(temp_predictions)
                # modify raw preidiction
                temp_checked_predictions = self.modify_prediction(temp_predictions)
                # reshape modified prediction to output shape
                out_predictions = self.modify_shape(temp_predictions=temp_checked_predictions, shape=self.model_output_data_shape)
                # save modified prediction
                predictions.append(out_predictions)

                # load & modify the next prompt
                temp_checked_predictions = self.modify_prediction(ans[i:i+1])
            
            sequence_num = 'song_'+str(sequence_num)

        predictions = np.array(predictions)
        self.predictions, self.predictions_raw = predictions, predictions_raw


        # save the output sequence .csv & .pkl
        # df = pd.DataFrame({'Generated music (before modify)': predictions_raw,
        #                    'Generated music': predictions,
        #                    'Original music': prompt_loaded,
        #                    })
        # df.to_csv(self.data_generation_path+'_rg_g_p.csv')

        file = [self.ckp, predictions_raw, predictions, ans]
        pickle.dump(file, open(self.data_generation_path+'_'+str(sequence_num)+'_ckp_rg_g_p.pkl', 'wb'))

        # words to midi .mid
        output_words = utils.predictions_to_words(predictions=predictions)
        utils.words_to_midi(words=output_words, 
                            midi_name=self.result_path+'_'+str(sequence_num)+'.mid')
        print("Finish generate_data (sequence_num={}, predictions.shape={})".format(sequence_num, predictions.shape))




# Model for generating the 2nd output sequence (2nd bar) from the 1st bar
class bar2_generation(object):
    ########################################
    # initialize
    ########################################
    def __init__(self, artist_name, epochs, model_details):
        # Setting parameters
        self.artist_name = artist_name

        # 1) data shape
        self.data_path = 'data/all_pairs_bar1_{}.pkl'.format(artist_name)
        self.train_val_split = 0.2 # 0.2 means "use 20% of the data for validation"
        self.train_test_split = 0.2 # 0.3 means "use 30% of the data for test"
        # 2) model structure
        self.seq_len = 180
        self.model_input_data_shape = (25, 4)
        self.GRU_num, self.time_dis, self.bn, self.L2 = model_details
        self.dropout_rate = 0.1
        self.hidden_size = 16
        # 3) compile model
        self.loss = 'mean_squared_error'
        self.optimizer = Adam(learning_rate=0.01)
        # 4) trainging
        self.batch_size = 32
        self.epochs = epochs
        # 5) modify prediction
        self.model_output_data_shape = (45, 4)

        self.record = "{}_{}_GRU{}".format(datetime.now().strftime("%m%d%H%M"), self.artist_name, 
                                            #   self.epochs, self.hidden_size, self.dropout_rate, 
                                              self.GRU_num)
        if self.L2: self.record = self.record+'_L2'
        if self.bn: self.record = self.record+'_BatchNorm'
        if self.time_dis: self.record = self.record+'_timedis'

        self.checkpoint_path = "ckpt/"+self.record+"_B2G.h5" # _ckpt-epoch={epoch:02d}-loss={loss:.2f}
        self.checkpoint_dir = os.path.dirname(self.checkpoint_path)
        self.ckp = None
        self.training_history_path = 'train/{}_B2G'.format(self.record)

        # 5) genration
        self.prompt_path = 'data/songs_all_pair_{}.pkl'.format(artist_name)
        self.data_generation_path = "data_generation/"+self.record+"_B2G"
        self.result_path = 'result/'+self.record+"_B2G"
        self.predictions, self.predictions_raw, self.prompt, self.sequence_num = 0, 0, 0, 0
        
        # Working
        # load training & testing data
        print("Begin loading_data (bar2_model) at {}".format(datetime.now()))

        with open(self.data_path, 'rb') as file:
            data = pickle.load(file)
        x = []
        y = []
        for training_data in data:
            x.append(training_data[0][:20])
            y.append(training_data[1])
        x = np.reshape(x, (-1, 20, 4))
        y = np.reshape(y, (-1, 180))
        # print('> x.shape & y.shape', x.shape, y.shape)

        # split data into training & testing data
        split_position = int(len(data)*(1-self.train_test_split))
        self.x_test, self.y_test = x[split_position:], y[split_position:]
        self.x_train, self.y_train = x[:split_position], y[:split_position]

        self.model_input_data_shape = x.shape[1:]
        self.model_output_data_shape = (int(y.shape[-1]/self.model_input_data_shape[-1]), self.model_input_data_shape[-1])
        self.seq_len = y.shape[-1]
        # print('> self.model_input_data_shape', self.model_input_data_shape, 
        #       'self.model_output_data_shape', self.model_output_data_shape, 
        #       'self.seq_len', self.seq_len)

    # Train the model, and also do validation at the end of each epoch 
    def model_learning_stage(self, ckp=None):
        print("Begin model_learning_stage (bar2_model) at", datetime.now())

        # load model by model_type
        model = self.loading_model(GRU_num=self.GRU_num, time_dis=self.time_dis, bn=self.bn, L2=self.L2)

        # transfer learning
        if ckp!=None: model.load_weights(ckp)

        # start training
        callbacks_list = [
            # keras will save the one and only .h5 ckp file which has min val_loss
            # verbose: once stopped, the callback will print the epoch number
            ModelCheckpoint(filepath=self.checkpoint_path, save_best_only=True,
                            verbose=1, monitor='val_loss', save_weights_only=True),
            # keras will stop of loss hasn'y improve for 50 epochs
            # verbose: once stopped, the callback will print the epoch number
            # patience: number of epochs on which we would like to see no improvement
            EarlyStopping(monitor='loss', verbose=1, patience=50)]
        history = model.fit(self.x_train, self.y_train,
                            epochs = self.epochs,
                            callbacks = callbacks_list,
                            batch_size = self.batch_size,
                            validation_split = self.train_val_split)
        # print("Finish model_learning_stage at", datetime.now())
        
        # output the training history (train/{}.csv)
        with open('{}.csv'.format(self.training_history_path), 'w', newline='') as csvfile:
            fieldnames = history.history.keys()
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerow(history.history)
            
        # visualize loss during training (train/{}.jpg)
        plot1 = plt.figure(1)
        plt.title('Hidden Size: {} | Dropout Rate: {}\nLoss: {}'.format(self.hidden_size, self.dropout_rate, self.loss))
        plt.plot(history.history['loss'], label='train_bar2_model')
        plt.plot(history.history['val_loss'], label='validation_bar2_model')
        plt.legend()
        plt.savefig('{}_bar2_model.jpg'.format(self.training_history_path))
        
        # test_stage
        print("Begin model_inferencing_stage (bar2_model) at", datetime.now())
        results = model.evaluate(self.x_test, self.y_test)
        print("> test loss, test acc:", results)
        # print("Finish model_inferencing_stage at", datetime.now())

    # Building (add layers & compile) the Network
    def loading_model(self, GRU_num, time_dis, bn, L2):
        # model setting
        print("Begin building model structure (bar2_model) at", datetime.now())
        
        model = Sequential(name='sequential_{}_{}_{}'.format(self.hidden_size, self.epochs, self.dropout_rate))
        # Input layer: 4 dimension in each timesteps & 45 timeseeps
        model.add(keras.Input(shape=(None, self.model_input_data_shape[-1])))
        for i in range(GRU_num):
            # GRU layer: return hidden_size output notes on each timesteps
            # every but the first GRU layer
            if i!= 0: 
                model.add(GaussianNoise(0.075))
            # the last GRU layer
            if i==GRU_num-1: 
                model.add(GRU(self.hidden_size, 
                            # return_sequences=True, 
                            kernel_regularizer=regularizers.l2(0.01)
                            ))
            # the middle GRU layer
            else:
                model.add(GRU(self.hidden_size, 
                            return_sequences=True, 
                            kernel_regularizer=regularizers.l2(0.01)
                            ))
            if bn: model.add(layers.BatchNormalization())
        model.add(Dense(units=self.seq_len , bias_initializer=Constant(value=8)))

        model.compile(
            loss=self.loss,
            optimizer=self.optimizer,
            metrics=['accuracy'])

        model.summary()

        return model

    # modify the prediction to MIDI readable format
    def modify_prediction(self, temp_predictions, bar1=False):
        # reshape (because checking by size 1*4) and round temp_predictions
        temp_predictions = np.round(np.reshape(temp_predictions, (-1, self.model_input_data_shape[-1])), 0)
        del_list = []
        # 1.
        # value out of range: Position (1 - 16) | Velocity (1 - 32) | Note On (1 - 128) | Durationn (1 - 64) | None (0)
        for i in range(0, len(temp_predictions)):
            # 3. unreasonable repeat note
            if i==0: repeat = False
            else: repeat = ((temp_predictions[i][0] == temp_predictions[i-1]).all()
                            and(temp_predictions[i][2] == temp_predictions[i-1][2]).all())

            # delete the row and add [0, 0, 0, 0] at the end when: <0 or >max or repeat note
            trash = ((temp_predictions[i][0]>16) 
                     or (temp_predictions[i][1]>32) 
                     or (temp_predictions[i][2]>128) 
                     or (temp_predictions[i][3]>64) 
                     or ((temp_predictions[i]<=0).any())
                     or repeat)
            if trash: del_list.append(i)
        # delete those value out of range
        temp_predictions = np.delete(temp_predictions, del_list, axis=0)
        # 2.
        # not in order
        for i in range(1, len(temp_predictions)):
            not_in_order = (temp_predictions[i-1][0] > temp_predictions[i][0])
            if not_in_order:
                # delete temp_predictions[i:] = save only temp_predictions[:i]
                temp_predictions = temp_predictions[:i]
                break

        # add empty events so that the size will be 180*1
        if bar1: 
            shape = self.model_input_data_shape
            length = self.model_input_data_shape[0]
        else: 
            shape = self.model_output_data_shape
            length = self.model_output_data_shape[0]

        empty_num =length-len(temp_predictions)
        temp_predictions = np.append(temp_predictions, np.zeros((empty_num, self.model_input_data_shape[-1])))
        # reshape temp_predictions
        # for utils.predictions_to_words()
        temp_checked_predictions = np.reshape(temp_predictions, shape)

        # size: 1*45*4
        return temp_checked_predictions.tolist()

    # generate_data with or without prompt; return predictions, predictions_raw, prompt_y
    def generate_data(self, sequence_num=None, prompt=None, ans=None, ckp_name=None):
        print("Begin generate_data (bar2_model) at {}".format(datetime.now()))

        predictions_raw = []
        predictions = []

        # Load the checkpoint
        if ckp_name == None:
            # set the model checkpoint for inferencing state at the latest model checkpoint
            self.ckp = self.checkpoint_path
            print("> the latest checkpoints: ", self.ckp)
        else:
            self.ckp = 'ckpt/'+ckp_name+'.h5' 
            print("> the assigned checkpoints: ", self.ckp)

        # Load model
        model = self.loading_model(GRU_num=self.GRU_num, time_dis=self.time_dis, bn=self.bn, L2=self.L2)
        # loading_model(GRU_num=self.GRU_num)

        # Load weight
        model.load_weights(self.ckp)

        # Load prompt
        # assign bar1 and sequence_num
        try:
            new_prompt = (prompt==None)
            self.sequence_num, temp_predictions, self.prompt = generate_prompt(prompt_path=self.data_path, 
                                                                               ans_path=self.prompt_path, 
                                                                               shape=self.model_input_data_shape)
        except:
            new_prompt = False
            temp_predictions = prompt
            self.sequence_num = sequence_num
            self.prompt = ans
        
        # Save bar1: raw and checked
        predictions_raw = temp_predictions.tolist()
        # predictions_raw.append(temp_predictions)
        temp_checked_predictions = self.modify_prediction(temp_predictions, bar1=True)
        predictions.append(temp_checked_predictions)
        # predictions = temp_checked_predictions.tolist()
        
        # generate output sequence
        temp_checked_predictions = np.reshape(temp_checked_predictions, (1, self.model_input_data_shape[0], self.model_input_data_shape[1]))
        temp_predictions = model.predict(x = temp_checked_predictions) # batch_size=self.batch_size 
        temp_checked_predictions = self.modify_prediction(temp_predictions)
            
        # save the generated sequence: raw and checked
        # predictions_raw = predictions_raw + temp_predictions.tolist() 
        predictions_raw.append(temp_predictions)
        # predictions = predictions + temp_checked_predictions.tolist() 
        predictions.append(temp_checked_predictions)

        self.predictions, self.predictions_raw = predictions, predictions_raw

        # save the output sequence .csv & .pkl
        # df = pd.DataFrame({'Generated music (before modify)': predictions_raw,
        #                    'Generated music': predictions,
        #                    'Original music': prompt_loaded,
        #                    })
        # df.to_csv(self.data_generation_path+'_rg_g_p.csv')

        file = [self.ckp, predictions_raw, predictions, self.prompt]
        pickle.dump(file, open(self.data_generation_path+'_'+str(sequence_num)+'_ckp_rg_g_p.pkl', 'wb'))

        # Words to midi .mid
        # reshape predictions from 25*4 & 45*4 to all 45*4
        flatten_predictions = []
        for seq in predictions:
            temp_predictions = np.reshape(seq, (-1, self.model_input_data_shape[-1]))
            temp_predictions = np.round(temp_predictions, 0)
            shorter = self.model_output_data_shape[-2]-len(temp_predictions)
            if shorter>0:
                # fill temp_predictions
                temp_predictions = np.append(temp_predictions, np.zeros((shorter, self.model_input_data_shape[-1])))
                # reshape again
                temp_predictions = np.reshape(temp_predictions, (-1, self.model_input_data_shape[-1]))
            flatten_predictions.append(temp_predictions)
        # predictions to words
        output_words = utils.predictions_to_words(predictions=flatten_predictions)
        # words to midi
        utils.words_to_midi(words=output_words, 
                            midi_name=self.result_path+'.mid')
        print("Finish generate_data (sequence_num={}, predictions={})".format(sequence_num, predictions))