import numpy as np
import pandas as pd
import itertools
import os
import gc
import logging

logger = logging.getLogger(__name__)

class DataRestructuring:
    BASIC_FORMAT = "%(asctime)s-%(levelname)s-%(message)s"
    chlr = logging.StreamHandler()
    chlr.setFormatter(logging.Formatter(BASIC_FORMAT))
    logger.setLevel('DEBUG')
    logger.addHandler(chlr)
    # 預先定義好的list，包含要預測的shop tag或目前要使用的columns
    SHOP_TAG_LIST = ['2', '6', '10', '12',
                    '13', '15', '18', '19',
                    '21', '22', '25', '26',
                    '36', '37', '39', '48']
    COL_LIST = ['dt', 'chid', 'shop_tag', 'txn_cnt', 'txn_amt', 'domestic_offline_cnt',
        'domestic_online_cnt', 'overseas_offline_cnt', 'overseas_online_cnt',
        'domestic_offline_amt_pct', 'domestic_online_amt_pct',
        'overseas_offline_amt_pct', 'overseas_online_amt_pct', 'card_1_txn_cnt',
        'card_2_txn_cnt', 'card_3_txn_cnt', 'card_4_txn_cnt', 'card_5_txn_cnt',
        'card_6_txn_cnt', 'card_7_txn_cnt', 'card_8_txn_cnt', 'card_9_txn_cnt',
        'card_10_txn_cnt', 'card_11_txn_cnt', 'card_12_txn_cnt',
        'card_13_txn_cnt', 'card_14_txn_cnt', 'card_other_txn_cnt',
        'card_1_txn_amt_pct', 'card_2_txn_amt_pct', 'card_3_txn_amt_pct',
        'card_4_txn_amt_pct', 'card_5_txn_amt_pct', 'card_6_txn_amt_pct',
        'card_7_txn_amt_pct', 'card_8_txn_amt_pct', 'card_9_txn_amt_pct',
        'card_10_txn_amt_pct', 'card_11_txn_amt_pct', 'card_12_txn_amt_pct',
        'card_13_txn_amt_pct', 'card_14_txn_amt_pct', 'card_other_txn_amt_pct',
        'masts', 'educd', 'trdtp', 'naty', 'poscd', 'cuorg', 'slam',
        'gender_code', 'age', 'primary_card']
    def __init__(self, df_train_path, df_test_path, start_index, number):
        self.df_train_path = df_train_path
        if os.path.isfile(self.df_train_path) == False:
            raise Exception(f'The train data file path is wrong: {self.df_train_path}.')
        self.df_test_path = df_test_path
        if os.path.isfile(self.df_test_path) == False:
            raise Exception(f'The test data file path is wrong: {self.df_test_path}.')
        self.start_index = start_index
        self.number = number

    def _read_test_data_limited(self):
        logger.info('Start reading test data.')
        df_test = pd.read_csv(self.df_test_path)
        df_test = df_test[self.start_index: self.start_index+self.number].reset_index(drop=True)
        logger.info(f'Test data shape: {df_test.shape}')
        logger.info('Finish reading test data.')
        self._df_test = df_test

    def _read_train_data_by_chunk(self, chunksize=100000):
        logger.info('Start reading train data.')
        reader = pd.read_csv(self.df_train_path,
                            error_bad_lines=False, # 會自動忽略錯誤row
                            # header=None, # 看資料有無欄位名稱
                            iterator=True,
                            usecols=self.COL_LIST  #限縮要取的資料欄位
                            )
        loop = True
        chunks = []
        while loop:
            try:
                chunk = reader.get_chunk(chunksize)
                chunk = chunk[chunk.chid.isin(self._df_test.chid)]    #只取測試資料的前10,000 chid
                chunk = chunk[chunk.shop_tag.isin(self.SHOP_TAG_LIST)]   #指取要預測的消費類別
                chunks.append(chunk)
            except StopIteration:
                loop = False
                print("Iteration is stopped.")
        df = pd.concat(chunks,ignore_index=True)
        df.shop_tag = df.shop_tag.astype('int8')
        df = self._reduce_mem_usage(df=df, silent=False)
        logger.info('Finish reading train data.')
        return df

    @staticmethod
    def _reduce_mem_usage(df, silent=True, allow_categorical=True, float_dtype="float32"):
        """
        Iterates through all the columns of a dataframe and downcasts the data type
        to reduce memory usage. Can also factorize categorical columns to integer dtype.
        """
        def _downcast_numeric(series, allow_categorical=allow_categorical):
            """
            Downcast a numeric series into either the smallest possible int dtype or a specified float dtype.
            """
            if pd.api.types.is_sparse(series.dtype) is True:
                return series
            elif pd.api.types.is_numeric_dtype(series.dtype) is False:
                if pd.api.types.is_datetime64_any_dtype(series.dtype):
                    return series
                else:
                    if allow_categorical:
                        return series
                    else:
                        codes, uniques = series.factorize()
                        series = pd.Series(data=codes, index=series.index)
                        series = _downcast_numeric(series)
                        return series
            else:
                series = pd.to_numeric(series, downcast="integer")
            if pd.api.types.is_float_dtype(series.dtype):
                series = series.astype(float_dtype)
            return series

        if silent is False:
            start_mem = np.sum(df.memory_usage()) / 1024 ** 2
            print("Memory usage of dataframe is {:.2f} MB".format(start_mem))
        if df.ndim == 1:
            df = _downcast_numeric(df)
        else:
            for col in df.columns:
                df.loc[:, col] = _downcast_numeric(df.loc[:,col])
        if silent is False:
            end_mem = np.sum(df.memory_usage()) / 1024 ** 2
            print("Memory usage after optimization is: {:.2f} MB".format(end_mem))
            print("Decreased by {:.1f}%".format(100 * (start_mem - end_mem) / start_mem))

        return df

    def _create_matrix_full(self, df_train):
        logger.info('Start creating training matrix.')
        indexlist = []
        x = itertools.product(df_train.dt.unique(), self._df_test.chid.unique(), self.SHOP_TAG_LIST)
        indexlist.append(np.array(list(x)))
        matrix = pd.DataFrame(
                data=np.concatenate(indexlist, axis=0),
                columns=["dt", "chid", "shop_tag"])
        matrix.dt = matrix.dt.astype('int8')
        matrix.chid = matrix.chid.astype('int32')
        matrix.shop_tag = matrix.shop_tag.astype('int8')
        matrix = matrix.merge(df_train, on=['dt', 'chid', 'shop_tag'], how='left')
        matrix.fillna(0, inplace=True)
        matrix = self._reduce_mem_usage(df=matrix, silent=False)
        self._matrix = matrix
        gc.collect()
        logger.info('Finish creating training matrix.')

    def execute(self):
        self._read_test_data_limited()
        df_train = self._read_train_data_by_chunk(chunksize=100000)
        self._create_matrix_full(df_train=df_train)
        logger.removeHandler(self.chlr)

    @property
    def test_data(self):
        return self._df_test

    @property
    def matrix(self):
        return self._matrix


if __name__ == '__main__':

    # usage of DataRestructuring
    op = DataRestructuring(df_train_path=r'D:\Gallon\Project\07_E_SUN_2021_Winter_AI\data\tbrain_cc_training_48tags_hash_final.csv',
                        df_test_path=r'D:\Gallon\Project\07_E_SUN_2021_Winter_AI\data\需預測的顧客名單及提交檔案範例.csv',
                        start_index=0,
                        number=1000)
    op.execute()