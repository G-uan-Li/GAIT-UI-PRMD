import numpy as np
import random
import datetime
import matplotlib.pyplot as plt
from keras.models import Model
from keras.layers import Input, Conv1D, LSTM, Dense, Dropout, Activation, Flatten, concatenate, UpSampling1D, Lambda
from keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
import os 

class SpatioTemporalModel:
    def __init__(self, timesteps=240, n_dim=90, learning_rate=0.0001):
        self.timesteps = timesteps
        self.n_dim = n_dim
        self.optimizer = Adam(learning_rate=learning_rate)
        self.model = None

    def load_data(self):
        # 替换为实际的数据加载函数
        import dataViconLoad
        Correct_data, Correct_label, Incorrect_data, Incorrect_label = dataViconLoad.load_data()
        self.Correct_data = Correct_data
        self.Correct_label = Correct_label
        self.Incorrect_data = Incorrect_data
        self.Incorrect_label = Incorrect_label
        print(f"Correct data shape: {Correct_data.shape}")
        print(f"Incorrect data shape: {Incorrect_data.shape}")

    def split_data(self):
        trainidx1 = random.sample(range(0, self.Correct_data.shape[0]), int(self.Correct_data.shape[0] * 0.7))
        trainidx2 = random.sample(range(0, self.Incorrect_data.shape[0]), int(self.Incorrect_data.shape[0] * 0.7))
        valididx1 = np.setdiff1d(np.arange(0, self.Correct_data.shape[0]), trainidx1)
        valididx2 = np.setdiff1d(np.arange(0, self.Incorrect_data.shape[0]), trainidx2)

        # Training set: data and labels
        self.train_x = np.concatenate((self.Correct_data[trainidx1, :, :], self.Incorrect_data[trainidx2, :, :]))
        self.train_y = np.concatenate((np.squeeze(self.Correct_label[trainidx1]), np.squeeze(self.Incorrect_label[trainidx2])))

        # Validation set: data and labels
        self.valid_x = np.concatenate((self.Correct_data[valididx1, :, :], self.Incorrect_data[valididx2, :, :]))
        self.valid_y = np.concatenate((np.squeeze(self.Correct_label[valididx1]), np.squeeze(self.Incorrect_label[valididx2])))

    def preprocess_data(self):
        # Reduce the data length by a factor of 2, 4, and 8
        self.train_x_2 = self.train_x[:, ::2, :]
        self.valid_x_2 = self.valid_x[:, ::2, :]
        self.train_x_4 = self.train_x[:, ::4, :]
        self.valid_x_4 = self.valid_x[:, ::4, :]
        self.train_x_8 = self.train_x[:, ::8, :]
        self.valid_x_8 = self.valid_x[:, ::8, :]

        # Reorder the data dimensions to correspond to the five body parts
        self.trainx = self.reorder_data(self.train_x)
        self.validx = self.reorder_data(self.valid_x)
        self.trainx_2 = self.reorder_data(self.train_x_2)
        self.validx_2 = self.reorder_data(self.valid_x_2)
        self.trainx_4 = self.reorder_data(self.train_x_4)
        self.validx_4 = self.reorder_data(self.valid_x_4)
        self.trainx_8 = self.reorder_data(self.train_x_8)
        self.validx_8 = self.reorder_data(self.valid_x_8)

    @staticmethod
    def reorder_data(x):
        X_trunk = np.concatenate((x[:, :, 15:18], x[:, :, 18:21], x[:, :, 24:27], x[:, :, 27:30]), axis=2)
        X_left_arm = np.concatenate((x[:, :, 81:84], x[:, :, 87:90], x[:, :, 93:96], x[:, :, 99:102], x[:, :, 105:108], x[:, :, 111:114]), axis=2)
        X_right_arm = np.concatenate((x[:, :, 84:87], x[:, :, 90:93], x[:, :, 96:99], x[:, :, 102:105], x[:, :, 108:111], x[:, :, 114:117]), axis=2)
        X_left_leg = np.concatenate((x[:, :, 33:36], x[:, :, 39:42], x[:, :, 45:48], x[:, :, 51:54], x[:, :, 57:60], x[:, :, 63:66], x[:, :, 69:72]), axis=2)
        X_right_leg = np.concatenate((x[:, :, 36:39], x[:, :, 42:45], x[:, :, 48:51], x[:, :, 54:57], x[:, :, 60:63], x[:, :, 66:69], x[:, :, 72:75]), axis=2)
        x_segmented = np.concatenate((X_trunk, X_right_arm, X_left_arm, X_right_leg, X_left_leg), axis=-1)
        # print(x_segmented.shape)   (124, 240, 90)
        # exit()
        return x_segmented

    @staticmethod
    def MultiBranchConv1D(input, filters1, kernel_size1, strides1, strides2):
        x1 = Conv1D(filters=filters1, kernel_size=kernel_size1 + 2, strides=strides1, padding='same', activation='relu')(input)
        x1 = Dropout(0.25)(x1)
        x2 = Conv1D(filters=filters1, kernel_size=kernel_size1 + 6, strides=strides1, padding='same', activation='relu')(input)
        x2 = Dropout(0.25)(x2)
        x3 = Conv1D(filters=filters1, kernel_size=kernel_size1 + 12, strides=strides1, padding='same', activation='relu')(input)
        x3 = Dropout(0.25)(x3)
        y1 = concatenate([x1, x2, x3], axis=-1)

        x4 = Conv1D(filters=filters1, kernel_size=kernel_size1, strides=strides2, padding='same', activation='relu')(y1)
        x4 = Dropout(0.25)(x4)
        x5 = Conv1D(filters=filters1, kernel_size=kernel_size1 + 2, strides=strides2, padding='same', activation='relu')(y1)
        x5 = Dropout(0.25)(x5)
        x6 = Conv1D(filters=filters1, kernel_size=kernel_size1 + 4, strides=strides2, padding='same', activation='relu')(y1)
        x6 = Dropout(0.25)(x6)
        x = concatenate([x4, x5, x6], axis=-1)
        return x

    def TempPyramid(self, input_f, input_2, input_4, input_8, n_dims):
        conv1 = self.MultiBranchConv1D(input_f, 64, 3, 2, 2)
        conv2 = self.MultiBranchConv1D(input_2, 64, 3, 2, 1)
        conv3 = self.MultiBranchConv1D(input_4, 64, 3, 1, 1)
        conv4 = self.MultiBranchConv1D(input_8, 64, 3, 1, 1)
        upsample1 = UpSampling1D(size=2)(conv4)

        x = concatenate([conv1, conv2, conv3, upsample1], axis=-1)
        return x

    def build_model(self):
        seq_input = Input(shape=(self.timesteps, self.n_dim), name='full_scale')
        seq_input_trunk = Lambda(lambda x: x[:, :, 0:12])(seq_input)
        seq_input_left_arm = Lambda(lambda x: x[:, :, 12:30])(seq_input)
        seq_input_right_arm = Lambda(lambda x: x[:, :, 30:48])(seq_input)
        seq_input_left_leg = Lambda(lambda x: x[:, :, 48:69])(seq_input)
        seq_input_right_leg = Lambda(lambda x: x[:, :, 69:90])(seq_input)

        seq_input_2 = Input(shape=(int(self.timesteps/2), self.n_dim), name='half_scale')
        seq_input_trunk_2 = Lambda(lambda x: x[:, :, 0:12])(seq_input_2)
        seq_input_left_arm_2 = Lambda(lambda x: x[:, :, 12:30])(seq_input_2)
        seq_input_right_arm_2 = Lambda(lambda x: x[:, :, 30:48])(seq_input_2)
        seq_input_left_leg_2 = Lambda(lambda x: x[:, :, 48:69])(seq_input_2)
        seq_input_right_leg_2 = Lambda(lambda x: x[:, :, 69:90])(seq_input_2)

        seq_input_4 = Input(shape=(int(self.timesteps/4), self.n_dim), name='quarter_scale')
        seq_input_trunk_4 = Lambda(lambda x: x[:, :, 0:12])(seq_input_4)
        seq_input_left_arm_4 = Lambda(lambda x: x[:, :, 12:30])(seq_input_4)
        seq_input_right_arm_4 = Lambda(lambda x: x[:, :, 30:48])(seq_input_4)
        seq_input_left_leg_4 = Lambda(lambda x: x[:, :, 48:69])(seq_input_4)
        seq_input_right_leg_4 = Lambda(lambda x: x[:, :, 69:90])(seq_input_4)

        seq_input_8 = Input(shape=(int(self.timesteps/8), self.n_dim), name='one_eight_scale')
        seq_input_trunk_8 = Lambda(lambda x: x[:, :, 0:12])(seq_input_8)
        seq_input_left_arm_8 = Lambda(lambda x: x[:, :, 12:30])(seq_input_8)
        seq_input_right_arm_8 = Lambda(lambda x: x[:, :, 30:48])(seq_input_8)
        seq_input_left_leg_8 = Lambda(lambda x: x[:, :, 48:69])(seq_input_8)
        seq_input_right_leg_8 = Lambda(lambda x: x[:, :, 69:90])(seq_input_8)

        trunk_feat = self.TempPyramid(seq_input_trunk, seq_input_trunk_2, seq_input_trunk_4, seq_input_trunk_8, 64)
        left_arm_feat = self.TempPyramid(seq_input_left_arm, seq_input_left_arm_2, seq_input_left_arm_4, seq_input_left_arm_8, 64)
        right_arm_feat = self.TempPyramid(seq_input_right_arm, seq_input_right_arm_2, seq_input_right_arm_4, seq_input_right_arm_8, 64)
        left_leg_feat = self.TempPyramid(seq_input_left_leg, seq_input_left_leg_2, seq_input_left_leg_4, seq_input_left_leg_8, 64)
        right_leg_feat = self.TempPyramid(seq_input_right_leg, seq_input_right_leg_2, seq_input_right_leg_4, seq_input_right_leg_8, 64)

        full_feat = concatenate([trunk_feat, left_arm_feat, right_arm_feat, left_leg_feat, right_leg_feat], axis=-1)
        full_feat = LSTM(512)(full_feat)
        full_feat = Dropout(0.5)(full_feat)
        full_feat = Dense(512, activation='relu')(full_feat)
        full_feat = Dropout(0.5)(full_feat)
        full_feat = Dense(512, activation='relu')(full_feat)
        full_feat = Dropout(0.5)(full_feat)
        output = Dense(1, activation='sigmoid')(full_feat)

        self.model = Model(inputs=[seq_input, seq_input_2, seq_input_4, seq_input_8], outputs=[output])
        self.model.compile(optimizer=self.optimizer, loss='mean_squared_error')
        self.model.summary()

    def train_model(self, batch_size=32, epochs=200, patience=20):
        early_stopping = EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True)
        history = self.model.fit(
            [self.trainx, self.trainx_2, self.trainx_4, self.trainx_8],
            self.train_y,
            validation_data=([self.validx, self.validx_2, self.validx_4, self.validx_8], self.valid_y),
            batch_size=batch_size,
            epochs=epochs,
            verbose=1,
            callbacks=[early_stopping]
        )
        return history

    def plot_results(self, history):
        save_folder = '/root/autodl-tmp/result'
        os.makedirs(save_folder, exist_ok=True)
        plt.figure(figsize=(12, 6))
        plt.plot(history.history['loss'], 'b', label='Training Loss')
        plt.plot(history.history['val_loss'], 'r', label='Validation Loss')
        plt.title('Training and Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig(os.path.join(save_folder, 'training_validation_loss07.png'))
        plt.tight_layout()
        
        
        plt.show()

        print("Training loss:", np.min(history.history['loss']))
        print("Validation loss:", np.min(history.history['val_loss']))


        pred_train = self.model.predict([self.trainx, self.trainx_2, self.trainx_4, self.trainx_8])
        pred_test = self.model.predict([self.validx, self.validx_2, self.validx_4, self.validx_8])


        plt.figure(figsize=(14, 8))
        plt.subplot(2, 1, 1)
        plt.plot(pred_train, 's', color='red', label='Prediction', linestyle='None', alpha=0.5, markersize=6)
        plt.plot(self.train_y, 'o', color='green', label='Quality Score', alpha=0.4, markersize=6)
        plt.ylim([-0.1, 1.1])
        plt.title('Training Set Predictions', fontsize=18)
        plt.xlabel('Sequence Number', fontsize=16)
        plt.ylabel('Quality Scale', fontsize=16)
        plt.legend(loc=3, prop={'size':14})

        plt.subplot(2, 1, 2)
        plt.plot(pred_test, 's', color='red', label='Prediction', linestyle='None', alpha=0.5, markersize=6)
        plt.plot(self.valid_y, 'o', color='green', label='Quality Score', alpha=0.4, markersize=6)
        plt.ylim([-0.1, 1.1])
        plt.title('Validation Set Predictions', fontsize=18)
        plt.xlabel('Sequence Number', fontsize=16)
        plt.ylabel('Quality Scale', fontsize=16)
        plt.legend(loc=3, prop={'size':14})

        plt.tight_layout()
        plt.savefig(os.path.join(save_folder, 'predictions.png'))
        plt.show()
        plt.close()
    

# 实例化并使用类
model = SpatioTemporalModel()
model.load_data()
model.split_data()
model.preprocess_data()
model.build_model()
history = model.train_model()
model.plot_results(history)
