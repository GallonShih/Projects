import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import os
import gc
import logging

logger = logging.getLogger(__name__)

class FeatureEngineering:
    BASIC_FORMAT = "%(asctime)s-%(levelname)s-%(message)s"
    chlr = logging.StreamHandler()
    chlr.setFormatter(logging.Formatter(BASIC_FORMAT))
    logger.setLevel('DEBUG')
    logger.addHandler(chlr)

    def __init__(self, matrix):
        self.matrix = matrix
        self.oldcols = matrix.columns

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

    def _shrink_mem_new_cols(self, allow_categorical=False):
        # Calls reduce_mem_usage on columns which have not yet been optimized
        if self.oldcols is not None:
            newcols = self.matrix.columns.difference(self.oldcols)
        else:
            newcols = self.matrix.columns
        self.matrix.loc[:,newcols] = self._reduce_mem_usage(self.matrix.loc[:,newcols], allow_categorical=allow_categorical)
        self.oldcols = self.matrix.columns  # This is used to track which columns have already been downcast

    @staticmethod
    def _list_if_not(s, dtype=str):
        # Puts a variable in a list if it is not already a list
        if type(s) not in (dtype, list):
            raise TypeError
        if (s != "") & (type(s) is not list):
            s = [s]
        return s

    def _month_feature_engineering(self):
        logger.info('Start month feature engineering.')
        self.matrix['month'] = (self.matrix.dt % 12) + 1
        self._shrink_mem_new_cols(allow_categorical=False)
        logger.info('Finish month feature engineering.')

    # 每種類別的人均購買次數
    def _shop_tag_mean_txn(self):
        shop_tag_features = self.matrix.groupby(['shop_tag', 'chid'])['txn_cnt'].sum().reset_index()
        shop_tag_features = shop_tag_features.groupby(['shop_tag'])['txn_cnt'].mean().reset_index(name='shop_tag_mean_txn')
        self.matrix = self.matrix.merge(shop_tag_features, on=['shop_tag'])
    # 每種類別的人均購買頻次
    def _shop_tag_frequency_txn(self):
        shop_tag_features = self.matrix.query('txn_cnt > 0').groupby(['shop_tag', 'chid'])['dt'].nunique().reset_index(name='dt_cnt')
        shop_tag_features = shop_tag_features.groupby(['shop_tag'])['dt_cnt'].mean().reset_index(name='shop_tag_frequency_txn')
        self.matrix = self.matrix.merge(shop_tag_features, on=['shop_tag'])

    def _shop_tag_feature_engineering(self):
        logger.info('Start shop tagging engineering.')
        self._shop_tag_mean_txn()
        self._shop_tag_frequency_txn()
        self._shrink_mem_new_cols(allow_categorical=False)
        logger.info('Finish shop tagging engineering.')

    def _cluster_feature(self, target_feature, clust_feature, level_feature, n_components=4, n_clusters=5, aggfunc="mean", exclude=None):
        matrix = self.matrix
        start_month = 12
        end_month = 24
        pt = matrix.query(f"dt>{start_month} & dt<={end_month}")
        if exclude is not None:
            pt = matrix[~matrix[clust_feature].isin(exclude)]
        pt = pt.pivot_table(values=target_feature, columns=clust_feature, index=level_feature, fill_value=0, aggfunc=aggfunc)
        pt = pt.transpose()
        pca = PCA(n_components=10)
        components = pca.fit_transform(pt)
        components = pd.DataFrame(components)
        # Plot PCA explained variance
        sns.set()
        features = list(range(pca.n_components_))
        fig = plt.figure(figsize=(10,4))
        ax = fig.add_subplot(121)
    #     ax.bar(features, pca.explained_variance_ratio_, color="black")
        sns.barplot(x=features, y=pca.explained_variance_ratio_, ax=ax)
        plt.title("Variance by PCA components")
        plt.xlabel("component")
        plt.ylabel("explained variance")
        plt.xticks(features)

        scorelist = []
        nrange = range(2, 10)
        for n in nrange:
            clusterer = AgglomerativeClustering(n_clusters=n)
            labels = clusterer.fit_predict(components)
            silscore = silhouette_score(pt, labels)
            scorelist.append(silscore)
        ax = fig.add_subplot(122)
        sns.lineplot(x=nrange, y=scorelist, ax=ax)
        plt.title("Clustering quality by number of clusters")
        plt.xlabel("n clusters")
        plt.ylabel("silhouette score")

        pca = PCA(n_components=n_components)
        components = pca.fit_transform(pt)
        components = pd.DataFrame(components)
        clusterer = AgglomerativeClustering(n_clusters=n_clusters, linkage="average")
        labels = clusterer.fit_predict(components)
        x = components[0]
        y = components[1]
        fig = plt.figure(figsize=(10, 4))
        ax = fig.add_subplot(111)
        sns.scatterplot(x=x, y=y, hue=labels, palette=sns.color_palette("hls", n_clusters), ax=ax)
        plt.title("Items by cluster")
        plt.xlabel("component 1 score")
        plt.ylabel("component 2 score")
        for i, txt in enumerate(pt.index.to_list()):
            ax.annotate(str(txt), (x[i], y[i]))
        groups = {}
        for i, s in enumerate(pt.index):
            groups[s] = labels[i]
        return groups
    def _cluster_feature_engineering(self):
        logger.info('Start clustering feature engineering.')
        shop_tag_group_dict = self._cluster_feature(
                                            'txn_amt',
                                            'shop_tag',
                                            'dt',
                                            n_components=2,
                                            n_clusters=4,
                                            aggfunc="mean",
                                            exclude =[])
        self.matrix['shop_tag_cluster'] = self.matrix['shop_tag'].map(shop_tag_group_dict)
        # 行業別根據不同類別的消費來分類
        chid_group_dict = self._cluster_feature(
                                        'txn_amt',
                                        'trdtp',
                                        'shop_tag',
                                        n_components=10,
                                        n_clusters=5,
                                        aggfunc="mean",
                                        exclude =[])
        self.matrix['trdtp_cluster'] = self.matrix['trdtp'].map(chid_group_dict)
        self._shrink_mem_new_cols(allow_categorical=False)
        gc.collect()
        logger.info('Finish clustering feature engineering.')

    def _add_pct_change(
        self,
        group_feats,
        target="txn_amt",
        aggfunc="mean",
        periods=1,
        lag=1,
        clip_value=None,
    ):
        periods = self._list_if_not(periods, int)
        group_feats = self._list_if_not(group_feats)
        group_feats_full = ["dt"] + group_feats
        dat = self.matrix.pivot_table(
            index=group_feats + ["dt"],
            values=target,
            aggfunc=aggfunc,
            fill_value=0,
            dropna=False,
        ).astype("float32")
        for g in group_feats:
            firsts = self.matrix.groupby(g).dt.min().rename("firsts")
            dat = dat.merge(firsts, left_on=g, right_index=True, how="left")
            dat.loc[dat.index.get_level_values("dt") < dat["firsts"], target] = float(
                "nan"
            )
            del dat["firsts"]
        for period in periods:
            feat_name = "_".join(
                group_feats + [target] + [aggfunc] + ["delta"] + [str(period)] + [f"lag_{lag}"]
            )
            print(f"Adding feature {feat_name}")
            dat = (
                dat.groupby(group_feats)[target]
                .transform(lambda x: x.pct_change(periods=period, fill_method="pad"))
                .rename(feat_name)
            )
            if clip_value is not None:
                dat = dat.clip(lower=-clip_value, upper=clip_value)
        dat = dat.reset_index()
        dat["dt"] += lag
        self.matrix = self.matrix.merge(dat, on=["dt"] + group_feats, how="left")
        self.matrix[feat_name] = self._reduce_mem_usage(self.matrix[feat_name])

    def _pct_change_feature_engineeing(self):
        logger.info('Start percentage changing feature engineering.')
        self._add_pct_change(["shop_tag"], "txn_amt", clip_value=3)
        self._add_pct_change(["shop_tag"], "txn_amt", lag=12, clip_value=3,)
        self._add_pct_change(["chid"], "txn_amt", clip_value=3)
        self._add_pct_change(["chid"], "txn_amt", lag=12, clip_value=3,)
        self._add_pct_change(["educd"], "txn_amt", clip_value=3)
        self._add_pct_change(["trdtp"], "txn_amt", clip_value=3)
        self._add_pct_change(["gender_code"], "txn_amt", clip_value=3)
        self._add_pct_change(["age"], "txn_amt", clip_value=3)
        self._shrink_mem_new_cols(allow_categorical=False)
        gc.collect()
        logger.info('Finish percentage changing feature engineering.')

    def _add_rolling_stats(
        self,
        features,
        window=12,
        kind="rolling",
        argfeat="txn_amt",
        aggfunc="mean",
        rolling_aggfunc="mean",
        dtype="float16",
        reshape_source=True,
        lag_offset=0,
    ):
        def rolling_stat(
            matrix,
            source,
            feats,
            feat_name,
            window=12,
            argfeat="txn_amt",
            aggfunc="mean",
            dtype=dtype,
            lag_offset=0,
        ):
            # Calculate a statistic on a windowed section of a source table,  grouping on specific features
            store = []
            for i in range(2 + lag_offset, 25 + lag_offset):
                if len(feats) > 0:
                    mes = (
                        source[source.dt.isin(
                            range(max([i - window, 0]), i))]
                        .groupby(feats)[argfeat]
                        .agg(aggfunc)
                        .astype(dtype)
                        .rename(feat_name)
                        .reset_index()
                    )
                else:
                    mes = {}
                    mes[feat_name] = (
                        source.loc[
                            source.dt.isin(
                                range(max([i - window, 0]), i)), argfeat
                        ]
                        .agg(aggfunc)
                        .astype(dtype)
                    )
                    mes = pd.DataFrame(data=mes, index=[i])
                mes["dt"] = i - lag_offset
                store.append(mes)
            store = pd.concat(store)
            self.matrix = matrix.merge(store, on=feats + ["dt"], how="left")

        """ An issue when using windowed functions is that missing values from months when items recorded no sales are skipped rather than being correctly
        treated as zeroes. Creating a pivot_table fills in the zeros."""
        if (reshape_source == True) or (kind == "ewm"):
            source = self.matrix.pivot_table(
                index=features + ["dt"],
                values=argfeat,
                aggfunc=aggfunc,
                fill_value=0,
                dropna=False,
            ).astype(dtype)
            for g in features:
                firsts = self.matrix.groupby(g).dt.min().rename("firsts")
                source = source.merge(
                    firsts, left_on=g, right_index=True, how="left")
                # Set values before the items first appearance to nan so they are ignored rather than being treated as zero sales.
                source.loc[
                    source.index.get_level_values(
                        "dt") < source["firsts"], argfeat
                ] = float("nan")
                del source["firsts"]
            source = source.reset_index()
        else:
            source = self.matrix
        if kind == "rolling":
            feat_name = (
                f"{'_'.join(features)}_{argfeat}_{aggfunc}_rolling_{rolling_aggfunc}_win_{window}"
            )
            print(f'Creating feature "{feat_name}"')
            rolling_stat(
                self.matrix,
                source,
                features,
                feat_name,
                window=window,
                argfeat=argfeat,
                aggfunc=rolling_aggfunc,
                dtype=dtype,
                lag_offset=lag_offset,
            )
        elif kind == "expanding":
            feat_name = f"{'_'.join(features)}_{argfeat}_{aggfunc}_expanding_{rolling_aggfunc}"
            print(f'Creating feature "{feat_name}"')
            rolling_stat(
                self.matrix,
                source,
                features,
                feat_name,
                window=100,
                argfeat=argfeat,
                aggfunc=aggfunc,
                dtype=dtype,
                lag_offset=lag_offset,
            )
        elif kind == "ewm":
            feat_name = f"{'_'.join(features)}_{argfeat}_{aggfunc}_ewm_hl_{window}"
            print(f'Creating feature "{feat_name}"')
            # source[feat_name] = (
            #     source.groupby(features)[argfeat]
            #     .ewm(halflife=window, min_periods=1)
            #     .agg(rolling_aggfunc)
            #     .to_numpy(dtype=dtype)
            # )
            source[feat_name] = (
                    source.groupby(features)[argfeat].apply(lambda x: x.ewm(halflife=window, min_periods=1).mean()).to_numpy(dtype=dtype)
                )
            del source[argfeat]
            #         source = source.reset_index()
            source["dt"] += 1 - lag_offset
            self.matrix = self.matrix.merge(source, on=["dt"] + features, how="left")

    def _rolling_feature_engineering(self):
        logger.info('Start rolling feature engineering.')
        self._add_rolling_stats(
            ["chid", "shop_tag"],
            window=12,
            kind="rolling",
            reshape_source=False,
        )
        self._add_rolling_stats(
            ["chid", "shop_tag"],
            kind="expanding",
            reshape_source=False,
        )
        self._add_rolling_stats(
            ["chid", "shop_tag"],
            window=1,
            kind="ewm",
        )

        self._add_rolling_stats(
            ["age", "shop_tag"],
            window=12,
            kind="rolling",
            reshape_source=False,
        )
        self._add_rolling_stats(
            ["age", "shop_tag"],
            kind="expanding",
            reshape_source=False,
        )
        self._add_rolling_stats(
            ["age", "shop_tag"],
            window=1,
            kind="ewm",
        )

        self._add_rolling_stats(
            ["age", "gender_code", "shop_tag"],
            window=12,
            kind="rolling",
            reshape_source=False,
        )
        self._add_rolling_stats(
            ["age", "gender_code", "shop_tag"],
            kind="expanding",
            reshape_source=False,
        )
        self._add_rolling_stats(
            ["age", "gender_code", "shop_tag"],
            window=1,
            kind="ewm",
        )
        self._shrink_mem_new_cols(allow_categorical=False)
        gc.collect()
        logger.info('Finish rolling feature engineering.')

    def _simple_lag_feature(self, lag_feature, lags):
        for lag in lags:
            newname = lag_feature + f"_lag_{lag}"
            print(f"Adding feature {newname}")
            targetseries = self.matrix.loc[:, ["dt", "chid", "shop_tag"] + [lag_feature]]
            targetseries["dt"] += lag
            targetseries = targetseries.rename(columns={lag_feature: newname})
            self.matrix = self.matrix.merge(
                targetseries, on=["dt", "chid", "shop_tag"], how="left"
            )
            self.matrix.loc[
                (self.matrix[newname].isna()),
                newname,
            ] = 0

    def _simple_lag_feature_engineering(self):
        logger.info('Start simple lag feature engineering.')
        self._simple_lag_feature('txn_amt', lags=[1,2,3])
        self._simple_lag_feature('txn_cnt', lags=[1, 2, 3])
        gc.collect()
        logger.info('Finish simple lag feature engineering.')

    def _create_apply_ME(
            self, grouping_fields, lags=[1], target="txn_amt", aggfunc="mean"
        ):
        grouping_fields = self._list_if_not(grouping_fields)
        for lag in lags:
            newname = "_".join(grouping_fields +
                            [target] + [aggfunc] + [f"lag_{lag}"])
            print(f"Adding feature {newname}")
            me_series = (
                self.matrix.groupby(["dt"] + grouping_fields)[target]
                .agg(aggfunc)
                .rename(newname)
                .reset_index()
            )
            me_series["dt"] += lag
            self.matrix = self.matrix.merge(
                me_series, on=["dt"] + grouping_fields, how="left")
            del me_series
            self.matrix[newname] = self.matrix[newname].fillna(0)
            for g in grouping_fields:
                firsts = self.matrix.groupby(g).dt.min().rename("firsts")
                self.matrix = self.matrix.merge(
                    firsts, left_on=g, right_index=True, how="left")
                self.matrix.loc[
                    self.matrix["dt"] < (self.matrix["firsts"] + (lag)), newname
                ] = float("nan")
                del self.matrix["firsts"]
            self.matrix[newname] = self._reduce_mem_usage(self.matrix[newname])

    def _create_apply_ME_feature_engineering(self):
        logger.info('Start ME feature engineering.')
        self._create_apply_ME(["shop_tag"], target="txn_amt")
        self._create_apply_ME(["shop_tag_cluster"], target="txn_amt")
        self._create_apply_ME(["trdtp"], target="txn_amt")
        self._create_apply_ME(["trdtp_cluster"], target="txn_amt")
        self._create_apply_ME(["chid"], target="txn_amt")
        self._create_apply_ME(["educd"], target="txn_amt")
        self._create_apply_ME(["poscd"], target="txn_amt")
        self._create_apply_ME(["masts", "gender_code", "age"], target="txn_amt")
        self._shrink_mem_new_cols(allow_categorical=False)
        gc.collect()
        logger.info('Finish ME feature engineering.')

    def execute(self):
        self._month_feature_engineering()
        self._shop_tag_feature_engineering()
        self._cluster_feature_engineering()
        self._pct_change_feature_engineeing()
        self._rolling_feature_engineering()
        self._create_apply_ME_feature_engineering()
        logger.removeHandler(self.chlr)