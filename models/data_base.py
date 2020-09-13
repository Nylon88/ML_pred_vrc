from contextlib import contextmanager
import logging
import threading
from time import sleep

from sqlalchemy import create_engine
from sqlalchemy import Column
from sqlalchemy import DateTime
from sqlalchemy import Float
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy.orm import scoped_session
from datetime import *
import sys
from pathlib import Path

import pandas as pd

sys.path.append(str(Path('__file__').parent.parent))
from utils import GetTools


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# エンジンの作成
engine = create_engine('mysql+pymysql://root:@localhost:3306/mysql?charset=utf8')
Session = scoped_session(sessionmaker(bind=engine))
Base = declarative_base()
lock = threading.Lock()


@contextmanager
def session_scope():
    session = Session()
    try:
        lock.acquire()
        yield session
        session.commit()
    except Exception as e:
        logger.error(f'action=session_scope() error={e}')
        session.rollback()
        raise
    finally:
        lock.release()
        print('session finish')
        # 今回はthreadで実行を終了した時closeにする為、コメントアウトしている


# ベースモデルの作成

# 作成したいモデルをベースを継承して作成
class CandleInfo(Base):
    tools = GetTools()
    # テーブルの名前
    __tablename__ = 'candle_info'

    time = Column(DateTime, primary_key=True, nullable=False)
    open = Column(Float)
    high = Column(Float)
    low = Column(Float)
    close = Column(Float)
    volume = Column(Float)

    @classmethod
    def insert(cls, ohlcvs):
        # リスト型を内包したタプル型を辞書型を内包したリスト型へと変換している
        table_date = [{'time': (datetime.fromtimestamp(ohlcv[0] / 1000).strftime('%Y-%m-%d %H:%M:00')),
                       'open': ohlcv[1], 'high': ohlcv[2], 'low': ohlcv[3], 'close': ohlcv[4],
                       'volume': ohlcv[5]} for ohlcv in ohlcvs]
        try:
            with session_scope() as session:
                session.execute(cls.__table__.insert(), table_date)
            logger.info(f'action=insert() update')
            return True
        except IntegrityError as e:
            logger.error(f'action=insert() same time No update')

    @classmethod
    def get_table_info(cls, many):
        time ,open, high, low, close, volume = [], [], [], [], [], []
        while True:
            since = cls.tools.get_ms_now(limit=many+1)
            since = datetime.fromtimestamp(since / 1000)
            with session_scope() as session:
                table_info = session.query(cls).filter(cls.time >= since).all()
            if table_info is []:
                return False
            if many == len(table_info):
                break
            sleep(3)

        for info in table_info:
            time.append(info.time)
            open.append(info.open)
            high.append(info.high)
            low.append(info.low)
            close.append(info.close)
            volume.append(info.volume)
        columns = ['time', 'open', 'high', 'low', 'close', 'volume']
        data = [time, open, high, low, close, volume]
        candle_info = cls.set_df(columns, data)
        return candle_info

    @classmethod
    def set_df(cls, columns, data):
        candle_info = pd.DataFrame()
        for i in range(len(columns)):
            candle_info[columns[i]] = data[i]
        return candle_info



def create_table():
    Base.metadata.create_all(bind=engine)
    logger.info(f'action=init_db() success')


def delete_table():
    Base.metadata.drop_all(bind=engine, tables=[CandleInfo.__table__])


if __name__ == '__main__':
    # delete_table()
    # create_table()
    # candle_info = [['2020-01-02 03:04:06', close]]
    #CandleInfo.insert(candle_info)
    print(CandleInfo.get_table_info(100))

    # print(CandleInfo.get_table_info())


