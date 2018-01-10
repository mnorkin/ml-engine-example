import argparse
import subprocess
import tempfile
import uuid

import os
import pandas as pd
import numpy as np
import tensorflow as tf
from pprint import pprint
from google.cloud import storage
from tensorflow.python.lib.io import file_io

storage_client = storage.Client()

flags = tf.app.flags
FLAGS = flags.FLAGS


def tf_confusion_metrics(model, actual_classes, session, feed_dict):
    predictions = tf.argmax(model, 1)
    actuals = tf.argmax(actual_classes, 1)

    ones_like_actuals = tf.ones_like(actuals)
    zeros_like_actuals = tf.zeros_like(actuals)
    ones_like_predictions = tf.ones_like(predictions)
    zeros_like_predictions = tf.zeros_like(predictions)
    # True positives
    tp_op = tf.reduce_sum(tf.cast(tf.logical_and(
        tf.equal(actuals, ones_like_actuals),
        tf.equal(predictions, ones_like_predictions)
    ), "float"))
    # True negatives
    tn_op = tf.reduce_sum(tf.cast(tf.logical_and(
        tf.equal(actuals, zeros_like_actuals),
        tf.equal(predictions, zeros_like_predictions)
    ), "float"))
    # False positives
    fp_op = tf.reduce_sum(tf.cast(tf.logical_and(
        tf.equal(actuals, zeros_like_actuals),
        tf.equal(predictions, ones_like_predictions)
    ), "float"))
    # False negatives
    fn_op = tf.reduce_sum(tf.cast(tf.logical_and(
        tf.equal(actuals, ones_like_actuals),
        tf.equal(predictions, zeros_like_predictions)
    ), "float"))

    tp, tn, fp, fn = session.run([ tp_op, tn_op, fp_op, fn_op ], feed_dict)

    tpfn = float(tp) + float(fn)
    tpr = 0 if tpfn == 0 else float(tp)/tpfn
    fpr = 0 if tpfn == 0 else float(fp)/tpfn

    total = float(tp) + float(fp) + float(fn) + float(tn)
    accuracy = 0 if total == 0 else (float(tp) + float(tn))/total

    recall = tpr
    tpfp = float(tp) + float(fp)
    precision = 0 if tpfp == 0 else float(tp)/tpfp

    f1_score = 0 if recall == 0 else (2 * (precision * recall)) / (precision + recall)

    print('Precision = ', precision)
    print('Recall = ', recall)
    print('F1 Score = ', f1_score)
    print('Accuracy = ', accuracy)


def copy_data_to_tmp(input_files):
    """Copies data to /tmp/ and returns glob matching the files."""
    files = []
    for e in input_files:
        for path in e.split(','):
            files.extend(file_io.get_matching_files(path))

    for path in files:
        if not path.startswith('gs://'):
            return input_files

    tmp_path = os.path.join('/tmp/', str(uuid.uuid4()))
    os.makedirs(tmp_path)
    subprocess.check_call(['gsutil', '-m', '-q', 'cp', '-r'] + files + [tmp_path])
    return tmp_path


def get_data(name):
    file_name = ".".join((name, "json"))
    file_path = "/".join((FLAGS.data_dir, file_name))
    return pd.read_json(copy_data_to_tmp(file_path), typ='series')


def run_training():

    closing_data = pd.DataFrame()

    indexes = ['snp', 'nyse', 'djia', 'nikkei', 'hangseng', 'ftse', 'dax', 'aord']

    for index in indexes:
        closing_data['{}_close'.format(index)] = get_data(index)

    closing_data = closing_data.fillna(method='ffill')

    for index in indexes:
        closing_data['{}_close_scaled'.format(index)] = closing_data['{}_close'.format(index)] / max(closing_data['{}_close'.format(index)])

    # Log Return
    log_return_data = pd.DataFrame()

    for index in indexes:
        log_return_data['{}_log_return'.format(index)] = np.log(
            closing_data['{}_close'.format(index)] / closing_data['{}_close'.format(index)].shift()
        )

    log_return_data['snp_log_return_positive'] = 0
    log_return_data.loc[log_return_data['snp_log_return'] >= 0, 'snp_log_return_positive'] = 1
    log_return_data['snp_log_return_negative'] = 0
    log_return_data.loc[log_return_data['snp_log_return'] < 0, 'snp_log_return_negative'] = 1

    training_test_data = pd.DataFrame(columns=[
        'snp_log_return_positive',
        'snp_log_return_negative',
        'snp_log_return_1',
        'snp_log_return_2',
        'snp_log_return_3',
        'nyse_log_return_1',
        'nyse_log_return_2',
        'nyse_log_return_3',
        'djia_log_return_1',
        'djia_log_return_2',
        'djia_log_return_3',
        'nikkei_log_return_0',
        'nikkei_log_return_1',
        'nikkei_log_return_2',
        'hangseng_log_return_0',
        'hangseng_log_return_1',
        'hangseng_log_return_2',
        'ftse_log_return_0',
        'ftse_log_return_1',
        'ftse_log_return_2',
        'dax_log_return_0',
        'dax_log_return_1',
        'dax_log_return_2',
        'aord_log_return_0',
        'aord_log_return_1',
        'aord_log_return_2'
    ])

    for i in range(7, len(log_return_data)):
        training_test_data = training_test_data.append({
            'snp_log_return_positive': log_return_data['snp_log_return_positive'].iloc[i],
            'snp_log_return_negative': log_return_data['snp_log_return_negative'].iloc[i],
            'snp_log_return_1': log_return_data['snp_log_return'].iloc[i-1],
            'snp_log_return_2': log_return_data['snp_log_return'].iloc[i-2],
            'snp_log_return_3': log_return_data['snp_log_return'].iloc[i-3],
            'nyse_log_return_1': log_return_data['nyse_log_return'].iloc[i-1],
            'nyse_log_return_2': log_return_data['nyse_log_return'].iloc[i-2],
            'nyse_log_return_3': log_return_data['nyse_log_return'].iloc[i-3],
            'djia_log_return_1': log_return_data['djia_log_return'].iloc[i-1],
            'djia_log_return_2': log_return_data['djia_log_return'].iloc[i-2],
            'djia_log_return_3': log_return_data['djia_log_return'].iloc[i-3],
            'nikkei_log_return_0': log_return_data['nikkei_log_return'].iloc[i],
            'nikkei_log_return_1': log_return_data['nikkei_log_return'].iloc[i-1],
            'nikkei_log_return_2': log_return_data['nikkei_log_return'].iloc[i-2],
            'hangseng_log_return_0': log_return_data['hangseng_log_return'].iloc[i],
            'hangseng_log_return_1': log_return_data['hangseng_log_return'].iloc[i-1],
            'hangseng_log_return_2': log_return_data['hangseng_log_return'].iloc[i-2],
            'ftse_log_return_0': log_return_data['ftse_log_return'].iloc[i],
            'ftse_log_return_1': log_return_data['ftse_log_return'].iloc[i-1],
            'ftse_log_return_2': log_return_data['ftse_log_return'].iloc[i-2],
            'dax_log_return_0': log_return_data['dax_log_return'].iloc[i],
            'dax_log_return_1': log_return_data['dax_log_return'].iloc[i-1],
            'dax_log_return_2': log_return_data['dax_log_return'].iloc[i-2],
            'aord_log_return_0': log_return_data['aord_log_return'].iloc[i],
            'aord_log_return_1': log_return_data['aord_log_return'].iloc[i-1],
            'aord_log_return_2': log_return_data['aord_log_return'].iloc[i-2]
        }, ignore_index=True)

    predictors_tf = training_test_data[training_test_data.columns[2:]]
    classes_tf = training_test_data[training_test_data.columns[:2]]

    training_set_size = int(len(training_test_data) * 0.8)

    training_predictors_tf = predictors_tf[:training_set_size]
    training_classes_tf = classes_tf[:training_set_size]

    test_predictors_tf = predictors_tf[training_set_size:]
    test_classes_tf = classes_tf[training_set_size:]

    sess = tf.Session()

    num_predictors = len(training_predictors_tf.columns)
    num_classes = len(training_classes_tf.columns)

    feature_data = tf.placeholder("float", [None, num_predictors])
    actual_classes = tf.placeholder("float", [None, num_classes])

    weights1 = tf.Variable(tf.truncated_normal([24, 50], stddev=0.0001))
    biases1 = tf.Variable(tf.ones([50]))

    weights2 = tf.Variable(tf.truncated_normal([50, 25], stddev=0.0001))
    biases2 = tf.Variable(tf.ones([25]))

    weights3 = tf.Variable(tf.truncated_normal([25, 2], stddev=0.0001))
    biases3 = tf.Variable(tf.ones([2]))

    hidden_layer_1 = tf.nn.relu(tf.matmul(feature_data, weights1) + biases1)
    hidden_layer_2 = tf.nn.relu(tf.matmul(hidden_layer_1, weights2) + biases2)
    model = tf.nn.softmax(tf.matmul(hidden_layer_2, weights3) + biases3)

    cost = -tf.reduce_sum(actual_classes*tf.log(model))

    training_step = tf.train.AdamOptimizer(learning_rate=0.0001).minimize(cost)

    init = tf.initialize_all_variables()
    sess.run(init)

    for i in range(1, 30001):
        sess.run(training_step, feed_dict={
            feature_data: training_predictors_tf.values,
            actual_classes: training_classes_tf.values.reshape(len(training_classes_tf.values), 2)
        })

    tf_confusion_metrics(model, actual_classes, sess, feed_dict={
        feature_data: test_predictors_tf.values,
        actual_classes: test_classes_tf.values.reshape(len(test_classes_tf.values), 2)
    })


def main(_):
    run_training()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default=None)
    FLAGS, _ = parser.parse_known_args()
    pprint(FLAGS.data_dir)
    tf.app.run()
