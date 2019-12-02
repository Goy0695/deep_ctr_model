import tensorflow as tf


class FeatureColumns:

    # 离散特征编码
    @staticmethod
    def one_hot(indx, table_set):
        column = tf.feature_column.categorical_column_with_vocabulary_list(key=str(indx),
                                                                           vocabulary_list=table_set,
                                                                           num_oov_buckets=0)
        return column

    # 连续特征数值化
    @staticmethod
    def numeric(indx):
        column = tf.feature_column.numeric_column(str(indx))
        return column

    # 对onehot之后的离散特征列进行embedding
    @staticmethod
    def column_embedding(column, dimension=4):
        embedding = tf.feature_column.embedding_column(column, dimension)
        return embedding

    # 对数值化的连续特征列进行分桶
    @staticmethod
    def column_bucket(column, bounary_list):
        bucket = tf.feature_column.bucketized_column(column, bounary_list)
        return bucket

    # 特征交叉
    @staticmethod
    def column_cross(columns: list, bucket_size: int):
        cross_bucket = tf.feature_column.crossed_column(keys=columns,
                                                        hash_bucket_size=bucket_size)
        return cross_bucket


