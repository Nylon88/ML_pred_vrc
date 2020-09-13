import ccxt
import logging
import calendar
from datetime import *
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GetTools:
    # 現在の時間をmsで取得し、スタートの時間を設定する
    def get_ms_now(self, limit=500):
        now = datetime.utcnow()
        unix_time = calendar.timegm(now.utctimetuple())

        since = (unix_time - 60 * limit) * 1000  # スタートの設定
        return since

    # functionに引数がないものだけ使用可能
    def interval_exe(self, function, interval):
        try:
            while (1):
                past = time.perf_counter()
                function()
                interval_time = interval - (time.perf_counter() - past)
                time.sleep(interval_time)
                logger.info(f'action= interval() {function}:１ループ終了')
        except Exception as e:
            logger.error(f'action= {function} error={e}')


class Bitmex:
    def __init__(self):
        self.bitmex = self.generate_test()
        self.tools = GetTools()

    def generate_test(self, api_key='SV1EBOnarbdvj0UMo2SF7mpU',
                      secret='HZVLIcSNwIfW_kDXNZW823__KoRoPclue_CA3cSvZNVmMph2'):
        # bitmexオブジェクトの作成
        bitmex = ccxt.bitmex({
            'apiKey': api_key,
            'secret': secret
        })
        bitmex.urls['api'] = bitmex.urls['test']

        return bitmex

    def get_ohlcv(self, limit):
        # ohlcvの取得
        ohlcv = self.bitmex.fetch_ohlcv(symbol='BTC/USD',
                                        timeframe='1m',
                                        since=self.tools.get_ms_now(limit),
                                        limit=limit)
        return ohlcv


if __name__ == '__main__':
    tools = GetTools
    mex = Bitmex()
    ohlcv = mex.get_ohlcv(10)
    print(ohlcv)

