import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from keras.layers import Lambda, Conv2D, MaxPooling2D, Dropout, Dense, Flatten
from utils import INPUT_SHAPE, batch_generator
import argparse
import os

np.random.seed(0)


def load_data(args):
    """
    加载培训数据并将其拆分为训练和验证集
    """
    data_df = pd.read_csv(os.path.join(args.data_dir, 'driving_log.csv'))

    X = data_df[['center', 'left', 'right']].values
    y = data_df['steering'].values

    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=args.test_size, random_state=0)

    return X_train, X_valid, y_train, y_valid


def build_model(args):
    """
    NVIDIA模型
    """
    model = Sequential()
    model.add(Lambda(lambda x: x/127.5-1.0, input_shape=INPUT_SHAPE))
    model.add(Conv2D(24, 5, 5, activation='elu', subsample=(2, 2)))
    model.add(Conv2D(36, 5, 5, activation='elu', subsample=(2, 2)))
    model.add(Conv2D(48, 5, 5, activation='elu', subsample=(2, 2)))
    model.add(Conv2D(64, 3, 3, activation='elu'))
    model.add(Conv2D(64, 3, 3, activation='elu'))
    model.add(Dropout(args.keep_prob))          #丢弃层
    model.add(Flatten())                        #平化层
    model.add(Dense(100, activation='elu'))
    model.add(Dense(50, activation='elu'))
    model.add(Dense(10, activation='elu'))
    model.add(Dense(1))
    model.summary()

    return model


def train_model(model, args, X_train, X_valid, y_train, y_valid):
    """
    训练模型
    """
    checkpoint = ModelCheckpoint('dataout/model-{epoch:03d}.h5',
                                 monitor='val_loss',
                                 verbose=0,
                                 save_best_only=args.save_best_only,
                                 mode='auto')
    # learning_rate 学习率
    model.compile(loss='mean_squared_error', optimizer=Adam(lr=args.learning_rate))
    # 开始训练      data_dir 数据目录          batch_size批大小    samples_per_epoch每个时期的样本数

    model.fit_generator(batch_generator(args.data_dir, X_train, y_train, args.batch_size, True),
                        args.samples_per_epoch,
                        args.nb_epoch,
                        max_q_size=1,
                        validation_data=batch_generator(args.data_dir, X_valid, y_valid, args.batch_size, False),
                        nb_val_samples=len(X_valid),
                        callbacks=[checkpoint],
                        verbose=1)


def s2b(s):
    """
   将字符串转换为布尔值
    """
    s = s.lower()
    return s == 'true' or s == 'yes' or s == 'y' or s == '1'


def main():
    """
    加载序列/验证数据集并训练模型
    """
    parser = argparse.ArgumentParser(description='自动驾驶训练')
    parser.add_argument('-d', help='数据目录',        dest='data_dir',          type=str,   default='F:\data')
    parser.add_argument('-t', help='测试大小分数',    dest='test_size',         type=float, default=0.2)
    parser.add_argument('-k', help='丢弃概率',        dest='keep_prob',         type=float, default=0.5)
    parser.add_argument('-n', help='周期数',          dest='nb_epoch',          type=int,   default=10)
    parser.add_argument('-s', help='每个时期的样本数',     dest='samples_per_epoch', type=int,   default=20000)
    parser.add_argument('-b', help='批量大小',              dest='batch_size',        type=int,   default=40)
    parser.add_argument('-o', help='仅保存最佳模型', dest='save_best_only',    type=s2b,   default='true')
    parser.add_argument('-l', help='学习率',         dest='learning_rate',     type=float, default=1.0e-4)

    args = parser.parse_args()

    print('-' * 30)
    print('参数')
    print('-' * 30)
    for key, value in vars(args).items():
        print('{:<20} := {}'.format(key, value))
    print('-' * 30)

    data = load_data(args)
    model = build_model(args)
    train_model(model, args, *data)


if __name__ == '__main__':
    main()

