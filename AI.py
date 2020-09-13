from utils import *
from models.data_base import *

import logging
from time import perf_counter
import talib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor


logger = logging.getLogger(__name__)

tools = GetTools()


class DataStore:
    def __init__(self, candle_info):
        self.data = candle_info.copy()
        self.ta = talib

    def tense_close(self, columns: list, tense):
        index = len(self.data)
        # 全ての要素にNoneを入れる
        none_data = [None for _ in range(index)]
        pd_none_data = pd.DataFrame()
        pd_none_data['none'] = none_data

        if tense == 'before':
            for column in columns:
                self.data['close_b{}'.format(column)] = pd_none_data['none']
                # column分前のcloseを要素に入れる
                self.data['close_b{}'.format(column)][column:] = self.data.loc[:index - column-1,
                                                                               'close'].values
        elif tense == 'after':
            for column in columns:
                self.data['close_a{}'.format(column)] = pd_none_data['none']
                # column分後のcloseを要素に入れる
                self.data['close_a{}'.format(column)][:-column] = self.data.loc[column:,
                                                                                'close'].values
        else:
            logger.error(f'action=tense_close error=tenseを正しく設定していません')

    def clu_sma(self, price, period):
        sma = []
        index = len(price.index)
        for i in range(index - period):
            sma.append(talib.SMA(price[i:period + i], timeperiod=period).values[-1])
        none_index = index - (len(sma))
        for i in range(none_index):
            sma.insert(0, None)
        self.data['sma_{}'.format(period)] = sma


class AiModel:
    def __init__(self):
        self.rf_model = RandomForestRegressor(random_state=0)

    def random_forest(self, train_x, train_y, test_x, test_y):
        # 学習を行う
        self.rf_model.fit(train_x, train_y)
        # 予測を行う
        pred_y = self.rf_model.predict(test_x)
        # 正解値と予測値をプロットする
        plot_data = pd.DataFrame()
        plot_data['test_y'] = test_y
        plot_data['pred_y'] = pred_y
        index = np.arange(0, len(test_x))
        keys = ['test_y', 'pred_y']
        data_plot(keys, index, plot_data)

        # 訓練とテスト値での精度を出力する。
        print(self.rf_model.score(train_x, train_y))
        print(self.rf_model.score(test_x, test_y))

        return pred_y


def set_train_test_data(data, ratio, before, after):
    # dataの型はpandas.DataFrame
    many = len(data)
    x = data.loc[before:many - 1 - after, 'open':'close_b{}'.format(before)]
    y = data.loc[before:many - 1 - after, 'close_a{}'.format(after)]

    train_len = int(len(x) * ratio)
    train_x = x[:train_len]
    train_y = y[:train_len]
    test_x = x[train_len:]
    test_y = y[train_len:]

    return train_x, train_y, test_x, test_y


def data_plot(keys, x, y):
    for key in keys:
        plt.plot(x, y[key])
    plt.show()


# interval毎に5分後の価格を予測する。
def interval_jud(interval, model, data_store, many=201):
    while True:
        now = perf_counter()
        candle_info = CandleInfo.get_table_info(many)
        data_store = DataStore(candle_info)
        before_columns = [i for i in range(1, 201)]
        data_store.tense_close(before_columns, 'before')
        x = data_store.data.loc[200, 'open':'close_b{}'.format(max(before_columns))]
        x = np.array(x).reshape(1, -1)

        pred_y = model.rf_model.predict(x)

        print(f'close:{x[0][3]}')
        print(f'5分後の予測値:{pred_y}')
        interval_time = interval - (perf_counter() - now)

        sleep(interval_time)


if __name__ == '__main__':
    # get_table_infoにどの情報が欲しいのかを引数に入れるコーディングをする
    many = 1000
    candle_info = CandleInfo.get_table_info(many)
    data_store = DataStore(candle_info)

    # class datastoreにbefore分前とafter分後のデータをいれる
    before_columns = [i for i in range(1, 201)]
    after_columns = [5]
    data_store.tense_close(before_columns, 'before')
    data_store.tense_close(after_columns, 'after')

    # train,testデータに分ける
    train_x, train_y, test_x, test_y = set_train_test_data(data=data_store.data, ratio=0.8,
                                                           before=max(before_columns), after=max(after_columns))
    model = AiModel()

    # modelの学習を行い、１度5分後のcloseを予測する。
    pred_y = model.random_forest(train_x, train_y, test_x, test_y)

    # 40秒ごとに5分後の価格を予測する
    interval_jud(40, model, data_store)

