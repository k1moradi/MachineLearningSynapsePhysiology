import os
import platform
import psutil
psutil.Process().nice()
from time import time, strftime, localtime
from tensorflow.keras import models, layers, optimizers, regularizers, constraints, initializers, losses
from tensorflow.keras.wrappers.scikit_learn import KerasRegressor
from tensorflow.keras.utils import get_custom_objects
from tensorflow.python.framework import ops
from tensorflow.python.ops import math_ops
from tensorflow.python.keras import backend as K
from sklearn.preprocessing import MaxAbsScaler, MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.utils import shuffle
from sklearn.ensemble import RandomForestRegressor
from eli5.sklearn import PermutationImportance
import tensorflow as tf
import tensorflow_addons as tfa
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import plotly.express as px
import plotly.io as pio

if platform.system() is 'Linux':
    pio.orca.config.use_xvfb = True
matplotlib.interactive(False)
if len(tf.config.experimental.list_physical_devices('GPU')) > 0:
    tf.config.experimental.set_memory_growth(tf.config.experimental.list_physical_devices('GPU')[0], True)
tf.config.optimizer.set_jit(True)
# tf.keras.mixed_precision.experimental.set_policy('mixed_float16')
tf.get_logger().setLevel('INFO')
random.seed(a=int(time() * 10 ** 7) % 4294967295, version=2)
tf.random.set_seed(int(time() * 10 ** 7) % 4294967295)
np.random.seed(int(time() * 10 ** 7) % 4294967295)
np.set_printoptions(
    threshold=np.inf,
    suppress=True,
    edgeitems=30,
    linewidth=100000,
    formatter=dict(float=lambda x: "%6.3f" % x)
)
pd.set_option('display.max_rows', 3300)
pd.set_option('display.max_columns', 3300)
pd.set_option('precision', 3)
epsilon = 3.5e-07  # 0.001 np.min(pooled_targets_na_interpolated)


def wsigmoid(x):
    return epsilon + tf.math.sigmoid(0.9 * x)


def gelu(x):
    return 0.5 * x * (1 + tf.tanh(tf.sqrt(2 / np.pi) * (x + 0.044715 * tf.pow(x, 3))))


def mish(x):
    return x * tf.tanh(tf.math.softplus(x))


def mml(x):  # min_max_linear_unit
    return tf.math.minimum(tf.math.maximum(x, -1), 1)


def isrlu(x, a=0.5):  # min_max_linear_unit
    return tf.math.maximum(tf.math.minimum(x / tf.math.sqrt(1 + a * tf.math.square(x)), 0), x)


def lisht(x):
    return x * tf.math.tanh(x)


get_custom_objects().update({'gelu': layers.Activation(gelu)})
get_custom_objects().update({'mish': layers.Activation(mish)})
get_custom_objects().update({'mml': layers.Activation(mml)})
get_custom_objects().update({'isrlu': layers.Activation(isrlu)})
get_custom_objects().update({'lisht': layers.Activation(lisht)})
get_custom_objects().update({'wsigmoid': layers.Activation(wsigmoid)})


def mean_arctan_absolute_percentage_error(y_true, y_pred):  # symetric version
    y_pred = ops.convert_to_tensor_v2(y_pred)
    y_true = math_ops.cast(y_true, y_pred.dtype)
    diff = math_ops.atan(math_ops.abs((y_true - y_pred) / (y_true + y_pred) * 2))
    return 100.0 * K.mean(diff, axis=-1)


def mean_log1p_absolute_percentage_error(y_true, y_pred):
    y_pred = ops.convert_to_tensor_v2(y_pred)
    y_true = math_ops.cast(y_true, y_pred.dtype)
    diff = math_ops.log1p(math_ops.abs((y_true - y_pred) / y_true))
    return 100.0 * K.mean(diff, axis=-1)


def mean_squared_log1p_absolute_percentage_error(y_true, y_pred):
    y_pred = ops.convert_to_tensor_v2(y_pred)
    y_true = math_ops.cast(y_true, y_pred.dtype)
    diff = math_ops.square(
        math_ops.log1p(
            math_ops.abs(
                (y_true - y_pred) / y_true)))
    return 100.0 * K.mean(diff, axis=-1)


def mean_soft_l1_absolute_percentage_error(y_true, y_pred):
    y_pred = ops.convert_to_tensor_v2(y_pred)
    y_true = math_ops.cast(y_true, y_pred.dtype)
    diff = 2 * (math_ops.sqrt(1 + math_ops.abs((y_true - y_pred) / y_true)) - 1)
    return 100.0 * K.mean(diff, axis=-1)


def scaled_msle(y_true, y_pred):
    scale = 10.0
    y_pred = ops.convert_to_tensor_v2(y_pred) * scale
    y_true = math_ops.cast(y_true, y_pred.dtype) * scale
    diff = math_ops.square(math_ops.log((1 + y_true) / (1 + y_pred)))
    return K.mean(diff, axis=-1)


def symetric_ape(y_true, y_pred):
    y_pred = ops.convert_to_tensor_v2(y_pred)
    y_true = math_ops.cast(y_true, y_pred.dtype)
    return 2 * math_ops.abs(math_ops.divide(
        math_ops.subtract(y_true, y_pred),
        math_ops.add(math_ops.abs(y_true), math_ops.abs(y_pred)))
    )


def symetric_mape(y_true, y_pred):
    diff = symetric_ape(y_true, y_pred)
    return 100.0 * K.mean(diff, axis=-1)


def symetric_mlape(y_true, y_pred):
    diff = math_ops.log1p(symetric_ape(y_true, y_pred))
    return 100.0 * K.mean(diff, axis=-1)


def symetric_mslape(y_true, y_pred):
    diff = math_ops.log1p(symetric_ape(y_true, y_pred))
    diff = 2 * (math_ops.sqrt(1 + math_ops.square(diff)) - 1)
    return 100.0 * K.mean(diff, axis=-1)


class BoundConstrainedMSLE(losses.MeanSquaredLogarithmicError):
    def __init__(self, penalty_coefficient=0.5, lb=0.0, ub=1.0, *args, **kwargs):
        losses.MeanSquaredLogarithmicError.__init__(self, *args, **kwargs)
        self.penalty_coefficient = float(penalty_coefficient)
        self.lb = lb
        self.ub = ub

    def __call__(self, y_true, y_pred, sample_weight=None):
        loss = losses.MeanSquaredLogarithmicError.__call__(self, y_true, y_pred, sample_weight=sample_weight)
        penalty_lb = tf.keras.backend.less(y_pred, self.lb)
        penalty_ub = tf.keras.backend.greater(y_pred, self.ub)
        penalty_sum = tf.keras.backend.cast(penalty_lb, tf.keras.backend.floatx()) + \
                      tf.keras.backend.cast(penalty_ub, tf.keras.backend.floatx())
        penalty = tf.keras.backend.sum(penalty_sum) * self.penalty_coefficient + 1.0
        return loss * penalty


def build_model(
        num_features=319, num_nodes=[8192, 512, 128, 512, 8192], activation='mish',
        L1_weight=None, L2_weight=2e-4,
        min_weight=None, max_weight=3.0,
        batch_normalization=True, drop_out=[0.5] * 5, noise=0.2,
        optimizer='ADAM', lookahead=False, learning_rate=1.25e-4,
        loss='MLAPE', negative_penalty_coef=0.0,
        template_model=None
):
    weight_regularizer, kernel_constraint = regularizers.l2(1e-6), None

    if L1_weight and L2_weight:
        weight_regularizer = regularizers.l1_l2(L1_weight, L2_weight)
    elif L1_weight:
        weight_regularizer = regularizers.l1(L1_weight)
    elif L2_weight and optimizer not in ('ADAMW', 'SGDW'):
        weight_regularizer = regularizers.l2(L2_weight)

    if min_weight and max_weight:
        kernel_constraint = constraints.MinMaxNorm(min_value=min_weight, max_value=max_weight)
    elif min_weight == 0 and not max_weight:
        kernel_constraint = constraints.NonNeg()
    elif max_weight and not min_weight:
        kernel_constraint = constraints.MaxNorm(max_value=max_weight)

    model = models.Sequential()
    model.add(layers.InputLayer(input_shape=(num_features,)))
    if noise > 0:
        model.add(layers.GaussianNoise(noise))

    for n, do in zip(num_nodes, drop_out):
        seed = int(time() * 10 ** 7) % 4294967295
        model.add(
            layers.Dense(
                units=n,
                use_bias=False,
                kernel_regularizer=weight_regularizer,
                kernel_constraint=kernel_constraint,
                # kernel_initializer=initializers.he_uniform(seed=seed)
                kernel_initializer=initializers.he_normal(seed=seed)
            ))
        if batch_normalization:
            model.add(layers.BatchNormalization())

        if activation is 'tanh':
            model.add(layers.Activation('tanh'))
        elif activation is 'softsign':
            model.add(layers.Activation('softsign'))
        elif activation is 'softmax':
            model.add(layers.Activation('softmax'))
        elif activation is 'PReLU':
            model.add(layers.PReLU(alpha_initializer=initializers.Constant(1e-9)))
        elif activation is 'ReLU':
            model.add(layers.ReLU(max_value=6, negative_slope=0.01, threshold=0))
        elif activation is 'LeakyReLU':
            model.add(layers.LeakyReLU(alpha=0.001))
        elif activation is 'ELU':
            model.add(layers.ELU(alpha=0.001))
        elif activation is 'selu':
            model.add(layers.Activation('selu'))
        elif activation is 'swish':
            model.add(layers.Activation('swish'))
        elif activation is 'mish':
            model.add(layers.Activation('mish'))
        elif activation is 'gelu':
            model.add(layers.Activation('gelu'))
        elif activation is 'lisht':
            model.add(layers.Activation('lisht'))
        elif activation is 'isrlu':
            model.add(layers.Activation('isrlu'))
        else:
            model.add(layers.Activation('relu'))

        if do > 0:
            model.add(layers.AlphaDropout(do) if activation is 'selu' else layers.Dropout(do))
    if noise > 0:
        model.add(layers.GaussianNoise(noise / 4))
    model.add(layers.Dense(units=5, activation='wsigmoid'))  # None 'relu' 'linear' 'tanh' 'mml' 'softsign'

    if template_model:
        model = models.load_model(template_model)

    if optimizer is 'ADAMW':
        opt = tfa.optimizers.AdamW(learning_rate=learning_rate, weight_decay=L2_weight, beta_1=0.89, epsilon=2.5e-07)
    elif optimizer is 'SGDW':
        opt = tfa.optimizers.SGDW(learning_rate=learning_rate, weight_decay=L2_weight)
    elif optimizer is 'ADAM':
        opt = optimizers.Adam(learning_rate=learning_rate, beta_1=0.89, epsilon=2.5e-07)
    elif optimizer is 'SGD':
        opt = optimizers.SGD(learning_rate=learning_rate)
    else:
        opt = tfa.optimizers.extend_with_decoupled_weight_decay(optimizers.Nadam)(
            learning_rate=learning_rate, weight_decay=L2_weight, beta_1=0.89, epsilon=2.5e-07)
    if lookahead:
        opt = tfa.optimizers.Lookahead(opt)

    if loss is 'MAPE':
        loss_fun = 'mean_absolute_percentage_error'
    elif loss is 'MLAPE':
        loss_fun = mean_log1p_absolute_percentage_error
    elif loss is 'MSLAPE':
        loss_fun = mean_squared_log1p_absolute_percentage_error
    elif loss is 'SMAAPE':
        loss_fun = mean_arctan_absolute_percentage_error
    elif loss is 'ML1APE':
        loss_fun = mean_soft_l1_absolute_percentage_error
    elif loss is 'SMAPE':
        loss_fun = symetric_mape
    elif loss is 'SMLAPE':
        loss_fun = symetric_mlape
    elif loss is 'SMSLAPE':
        loss_fun = symetric_mslape
    elif loss is 'SMSLE':
        loss_fun = scaled_msle
    elif loss is 'MSLE':
        if negative_penalty_coef > 0:
            loss_fun = BoundConstrainedMSLE(penalty_coefficient=negative_penalty_coef)
        else:
            loss_fun = 'mean_squared_logarithmic_error'
    else:
        print('loss function is not recognized')
        loss_fun = loss

    model.compile(optimizer=opt, loss=loss_fun, metrics=[mean_arctan_absolute_percentage_error])
    return model


class StopGoal(tf.keras.callbacks.Callback):
    def __init__(self, monitor1='mean_arctan_absolute_percentage_error', goal1=100.0,
                 monitor2='loss', goal2=25.0, patience=None):
        super(StopGoal, self).__init__()
        self.counter = 0
        self.monitor1 = monitor1
        self.goal1 = goal1
        self.best1 = goal1 * 1.1

        self.monitor2 = monitor2
        self.goal2 = goal2
        self.best2 = goal2 * 1.1

        self.best_weights = None
        self.best_epoch = 0
        self.patience = 0
        self.patience_base = patience

    def get_monitor_values(self, logs):
        logs = logs or {}
        monitor_value1 = logs.get(self.monitor1)
        monitor_value2 = logs.get(self.monitor2)
        if monitor_value1 is None:
            print(
                f'Early stopping metric 1 {self.monitor1} is not available. '
                f'Available metrics are: {", ".join(list(logs.keys()))}')
        if monitor_value2 is None:
            print(
                f'Early stopping metric 2 {self.monitor2} is not available. '
                f'Available metrics are: {", ".join(list(logs.keys()))}')
        return monitor_value1, monitor_value2

    def on_train_begin(self, logs=None):
        self.best_weights = None
        self.best1 = self.goal1 * 1.1
        self.best2 = self.goal2 * 1.1
        self.patience += self.patience_base

    def on_epoch_end(self, epoch, logs=None):
        self.counter += 1
        current1, current2 = self.get_monitor_values(logs)
        if current1 is None or current2 is None:
            return

        if current1 < self.best1 or current2 < self.best2:
            self.best1 = current1
            self.best2 = current2
            self.best_weights = self.model.get_weights()
            self.best_epoch = self.counter

        if current1 <= self.goal1 and current2 <= self.goal2:
            print(
                f'{strftime("%H:%M:%S", localtime())} reached goal {self.monitor1}: {current1:5.2f}, best {self.monitor2}: {current2:5.2f} at epoch {self.counter}.')
            self.best_weights = None  # to block restoring the best weights
            self.model.stop_training = True

        if self.best_weights and self.patience_base and (self.counter - self.best_epoch) > self.patience:
            self.model.stop_training = True

    def on_train_end(self, logs=None):
        if self.best_weights:
            print(
                f'{strftime("%H:%M:%S", localtime())} restored best {self.monitor1}: {self.best1:5.2f}, best {self.monitor2}: {self.best2:5.2f} at epoch {self.best_epoch}.')
            self.model.set_weights(self.best_weights)


class KerasRegressorLoadable(KerasRegressor):
    def __init__(self, model=None, *args, **kwargs):
        KerasRegressor.__init__(self, *args, **kwargs)
        import types
        if model:
            self.model = model
        elif not hasattr(self, 'model'):
            if self.build_fn is None:
                self.model = self.__call__(**self.filter_sk_params(self.__call__))
            elif (not isinstance(self.build_fn, types.FunctionType) and
                  not isinstance(self.build_fn, types.MethodType)):
                self.model = self.build_fn(
                    **self.filter_sk_params(self.build_fn.__call__))
            else:
                self.model = self.build_fn(**self.filter_sk_params(self.build_fn))

    def fit(self, x, y, **kwargs):
        import copy
        from tensorflow.python.keras.models import Sequential
        fit_args = copy.deepcopy(self.filter_sk_params(Sequential.fit))
        fit_args.update(kwargs)
        history = self.model.fit(x, y, **fit_args)
        return history

    def re_initialize(self):
        self.model.reset_metrics()
        self.model.reset_states()
        for layer in self.model.layers:
            if isinstance(layer, tf.keras.Model):
                for k, initializer in layer.__dict__.items():
                    if "initializer" not in k:
                        continue
                    # find the corresponding variable
                    var = getattr(layer, k.replace("_initializer", ""))
                    var.assign(initializer(var.shape, var.dtype))

    def load_weights(self, *args, **kwargs):
        self.model.load_weights(*args, **kwargs)


def smooth_curve(points, factor=0.25):
    smoothed_points = []
    for point in points:
        if point and smoothed_points:
            previous = smoothed_points[-1]
            smoothed_points.append(previous * factor + point * (1 - factor))
        elif smoothed_points and not point:
            smoothed_points.append(smoothed_points[-1])
        elif point and not smoothed_points:
            smoothed_points.append(point)
        else:
            smoothed_points.append(np.nanmax(points))

    return smoothed_points


def mini_describe(np_x):
    mean = np_x.mean()
    std = np_x.std()
    cv = std / mean
    return cv, mean, std


def find_index(a_list, value):
    try:
        return str(np.where(np.array(a_list) == value)[0][0])
    except:
        return 'not found'


def plot_result(
        result, loss_score, maape_score, out_of_bound_values,
        predicted_targets=None,
        skip=10, save_path_name=None
):
    history = result.history
    val_maape, min_val_maape = None, None
    val_loss, min_val_loss = None, None
    if 'val_mean_arctan_absolute_percentage_error' in history:
        min_val_maape = np.nanmin(history['val_mean_arctan_absolute_percentage_error'])
        val_maape = smooth_curve(history['val_mean_arctan_absolute_percentage_error'][skip:])
    if 'val_loss' in history:
        min_val_loss = np.nanmin(history['val_loss'])
        val_loss = smooth_curve(history['val_loss'][skip:])

    min_maape = np.nanmin(history['mean_arctan_absolute_percentage_error'])
    min_loss = np.nanmin(history['loss'])
    maape = smooth_curve(history['mean_arctan_absolute_percentage_error'][skip:])
    loss = smooth_curve(history['loss'][skip:])

    try:
        np_loss = np.array(history['loss'])
        loss_location = np.where((loss_score - 1e-4 <= np_loss) & (np_loss <= loss_score + 1e-4))[0][-1]
    except:
        loss_location = 'not found'

    try:
        np_maape = np.array(history['mean_arctan_absolute_percentage_error'])
        maape_location = np.where((maape_score - 1e-2 <= np_maape) & (np_maape <= maape_score + 1e-2))[0][-1]
    except:
        maape_location = 'not found'

    text1 = \
        f'Model fitness:\n' \
        f'Tm Loss: {loss_score:7.5f}@{loss_location}\n' \
        f'Tl Loss: {min_loss:7.5f}@{find_index(history["loss"], min_loss)}\n' \
        f'{"Vl Loss: " + "%7.5f" % min_val_loss + "@" + find_index(history["val_loss"], min_val_loss) if min_val_loss else ""}\n' \
        f'Tm SMAAPE: {maape_score: 6.1f}@{maape_location}\n' \
        f'Tl SMAAPE: {min_maape: 6.1f}@{find_index(history["mean_arctan_absolute_percentage_error"], min_maape)}\n' \
        f'{"Vl SMAAPE: " + "% 7.1f" % min_val_maape + "@" + find_index(history["val_mean_arctan_absolute_percentage_error"], min_val_maape) if min_val_maape else ""}\n' \
        f'OBVs: {out_of_bound_values}'

    if predicted_targets is not None:
        text2 = '\n'.join((
            'Predictions:',
            'g  CV=%.1f% 5.1f \u00B1 %.1f' % mini_describe(predicted_targets[:, 0]),
            '\u03C4d CV=%.1f% 5.1f \u00B1 %.1f' % mini_describe(predicted_targets[:, 1]),
            '\u03C4r CV=%.1f% 5.0f \u00B1 %.0f' % mini_describe(predicted_targets[:, 2]),
            '\u03C4f CV=%.1f% 5.0f \u00B1 %.0f' % mini_describe(predicted_targets[:, 3]),
            'U  CV=%.1f% 5.1f \u00B1 %.2f' % mini_describe(predicted_targets[:, 4]),
            'g  < 1.0: #%d' % predicted_targets[predicted_targets[:, 0] < 1.0].shape[0],
            '\u03C4d > 30 : #%d' % predicted_targets[predicted_targets[:, 1] > 30].shape[0],
            '\u03C4r < 10 : #%d' % predicted_targets[predicted_targets[:, 2] < 10].shape[0],
            '\u03C4f < 10 : #%d' % predicted_targets[predicted_targets[:, 3] < 10].shape[0],
            'U  > 0.3: #%d' % predicted_targets[predicted_targets[:, 4] > 0.3].shape[0],
            'U  < 0.1: #%d' % predicted_targets[predicted_targets[:, 4] < 0.1].shape[0],
        ))
    else:
        text2 = ''
    epochs = range(skip, len(maape) + skip)

    fig = plt.figure(figsize=(15, 5))
    ax1 = plt.subplot(1, 2, 1)
    plt.plot(epochs, loss, 'm', label='Training loss')
    if val_loss:
        plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.yscale('log')
    plt.legend(loc='lower left')
    font = {'fontname': 'monospace', 'size': '10'}
    plt.text(0.55, 0.63, text1, fontweight='bold', transform=ax1.transAxes, **font)

    ax2 = plt.subplot(1, 2, 2)
    plt.plot(epochs, maape, 'm', label='Training SMAAPE')
    if val_maape:
        plt.plot(epochs, val_maape, 'b', label='Validation SMAAPE')
    plt.title('Training SMAAPE')
    plt.xlabel('Epochs')
    plt.ylabel('SMAAPE')
    plt.yscale('log')
    plt.legend(loc='lower left')
    plt.text(0.55, 0.47, text2, fontweight='bold', transform=ax2.transAxes, **font)

    if save_path_name:
        plt.savefig(save_path_name + '.svg')
    plt.show()


def plot_feature_importance(perm, names_, head,
                            title=None, save_path_name=None, round_digits=None):
    feature_importances, feature_importances_std = perm.feature_importances_, perm.feature_importances_std_
    if round_digits:
        feature_table = sorted(zip(
            map(lambda i: round(i, round_digits), feature_importances), feature_importances_std, names_), reverse=True)
    else:
        feature_table = sorted(zip(feature_importances, feature_importances_std, names_), reverse=True)

    predictive_powers, labels, stds = [], [], []
    for predictive_power, std, label in feature_table:
        predictive_powers.append(predictive_power)
        stds.append(std)
        labels.append(label)
    x = range(len(labels))
    end_at = head if head < len(feature_importances) else len(feature_importances)
    fig = plt.figure(figsize=(8, 5 * int(end_at / 20)))
    bar_plot = plt.barh(x[0:end_at], predictive_powers[0:end_at], height=0.5, xerr=stds)
    plt.legend(bar_plot, [
        'important features #%i' % len(
            feature_importances[
                np.round(feature_importances, round_digits) >= 10 ** -round_digits
                ] if round_digits else feature_importances[feature_importances >= 0])])
    plt.yticks(x[0:end_at], labels[0:end_at])
    plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
    plt.xscale('log')
    plt.ylim([-1, end_at])
    if title:
        plt.title(title)
    if save_path_name:
        plt.savefig(save_path_name)
        np.save(save_path_name[:-4], feature_table)
    plt.show()


class StopExecution(Exception):
    def _render_traceback_(self):
        pass


class Perm():
    def __init__(self, path=None):
        self.feature_importances_ = []
        self.feature_importances_std_ = []
        self.feature_names = []
        if path:
            feature_table = np.load(path)
            self.feature_importances_ = feature_table[:, 1].astype(float)
            self.feature_importances_std_ = feature_table[:, 0].astype(float)
            self.feature_names = feature_table[:, 2]


def num_out_of_bound_values(predicted_targets):
    # columns 0 = g, 1 = tau_d, 2 = tau_r, 3 = tau_f, 4 = U
    out_of_bound_values = (
            predicted_targets[predicted_targets[:, 0] < 0].shape[0] +
            predicted_targets[predicted_targets[:, 1] < 0].shape[0] +
            predicted_targets[predicted_targets[:, 1] > 70].shape[0] +
            predicted_targets[predicted_targets[:, 2] < 0].shape[0] +
            predicted_targets[predicted_targets[:, 3] < 0].shape[0] +
            predicted_targets[predicted_targets[:, 4] > 1].shape[0] +
            predicted_targets[predicted_targets[:, 4] < 0].shape[0])
    return out_of_bound_values


def are_bio_plausible(predicted_targets):
    # columns 0 = g, 1 = tau_d, 2 = tau_r, 3 = tau_f, 4 = U
    # predicted_targets[predicted_targets[:, 1] > 30   ].shape[0]    <  500   and \
    # predicted_targets[predicted_targets[:, 0]  < 4.0 ].shape[0]    >  0     and \
    # (predicted_targets[predicted_targets[:, 4]  > 0.20].shape[0] ) >  0     and \
    # np.corrcoef(predicted_targets[:, 0], predicted_targets[:, 1])[0,1] <= 0.1 and \
    required_unique_values = 2948
    conditions = []
    if num_out_of_bound_values(predicted_targets) == 0:
        conditions += [True]
    else:
        conditions += [False]
        print('out of bound values detected.')
    ######################################################
    if np.unique(predicted_targets[:, 0]).shape[0] >= required_unique_values:
        conditions += [True]
    else:
        conditions += [False]
        print(f'not all g values were unique (n={np.unique(predicted_targets[:, 0]).shape[0]}).')

    if np.unique(predicted_targets[:, 1]).shape[0] >= required_unique_values:
        conditions += [True]
    else:
        conditions += [False]
        print(f'not all tau_d values were unique (n={np.unique(predicted_targets[:, 1]).shape[0]}).')

    if np.unique(predicted_targets[:, 2]).shape[0] >= required_unique_values:
        conditions += [True]
    else:
        conditions += [False]
        print(f'not all tau_r values were unique (n={np.unique(predicted_targets[:, 2]).shape[0]}).')

    if np.unique(predicted_targets[:, 3]).shape[0] >= required_unique_values:
        conditions += [True]
    else:
        conditions += [False]
        print(f'not all tau_f values were unique (n={np.unique(predicted_targets[:, 3]).shape[0]}).')

    if np.unique(predicted_targets[:, 4]).shape[0] >= required_unique_values:
        conditions += [True]
    else:
        conditions += [False]
        print(f'not all U values were unique (n={np.unique(predicted_targets[:, 4]).shape[0]}).')
    ######################################################
    if predicted_targets[:, 0].std() / predicted_targets[:, 0].mean() >= 0.005:
        conditions += [True]
    else:
        conditions += [False]
        print('CV of g values were less than 0.005.')

    if predicted_targets[:, 1].std() / predicted_targets[:, 1].mean() >= 0.005:
        conditions += [True]
    else:
        conditions += [False]
        print('CV of tau_d values were less than 0.005.')

    if predicted_targets[:, 2].std() / predicted_targets[:, 2].mean() >= 0.005:
        conditions += [True]
    else:
        conditions += [False]
        print('CV of tau_r values were less than 0.005.')

    if predicted_targets[:, 3].std() / predicted_targets[:, 3].mean() >= 0.005:
        conditions += [True]
    else:
        conditions += [False]
        print('CV of tau_f values were less than 0.005.')

    if predicted_targets[:, 4].std() / predicted_targets[:, 4].mean() >= 0.005:
        conditions += [True]
    else:
        conditions += [False]
        print('CV of U values were less than 0.005.')

    return all(conditions)


def matrix_plot(df, path_file_name):
    # df=pd.read_csv(path_file_name)
    df['type'] = df['Presynaptic Neuron'].apply(lambda x: 'Excitatory' if '(+)' in x else 'Inhibitory')
    fig = px.scatter_matrix(df,
                            dimensions=['g', 'tau_d', 'tau_r', 'tau_f', 'U'],
                            color='type', color_discrete_map={'Excitatory': 'coral', 'Inhibitory': 'cornflowerblue'},
                            opacity=0.2,
                            width=1920, height=1080)
    try:
        fig.write_image(path_file_name)
    except:
        print('Error saving ' + path_file_name)
        pass
    # fig.show()


def join_results(result1, result2):
    if result1 is None:
        return result2
    else:
        result1.history['mean_arctan_absolute_percentage_error'] += result2.history[
            'mean_arctan_absolute_percentage_error']
        result1.history['loss'] += result2.history['loss']
        if 'val_mean_arctan_absolute_percentage_error' in result2.history:
            result1.history['val_mean_arctan_absolute_percentage_error'] += result2.history[
                'val_mean_arctan_absolute_percentage_error']
        if 'val_loss' in result2.history:
            result1.history['val_loss'] += result2.history['val_loss']
        return result1


def assess_model(
        train_features_filename, train_targets_filename, predict_data_startswith, potential_connections_df,
        numerical_columns = [# List of feature columns that need normalization
            'Slice_Thickness', 'ISI', 'Temperature', 'Vm', 'Erev_GABA_B', 'Erev_NMDA',
            'Erev_GABA_A', 'Erev_AMPA', 'Cai', 'Cao', 'Cli', 'Clo', 'Csi', 'H2PO4o',
            'HCO3i', 'HCO3o', 'HEPESi', 'Ki', 'Ko', 'Mgi', 'Mgo', 'Nai', 'Nao', 'Bri',
            'gluconatei', 'QX314i', 'ATPi', 'EGTAi', 'EGTAo', 'GTPi', 'OHi', 'SO4i',
            'SO4o', 'phosphocreatinei', 'methanesulfonatei', 'acetatei', 'methylsulfatei',
            'NMDGi', 'Trisi', 'CeSO4i', 'pyruvateo', 'TEAi', 'Bao', 'HPO4o', 'Age'],
        passage_num=1,
        complete_stp=True, isi_column_name='ISI',
        validation_data=None, jack_knife=False, random_forest=False,
        matrix_plot_is_needed=False, feature_importance_iterations=0,
        source_directory=None, destination_directory=None,
        template_model=None, template_weights=None,
        num_nodes=[8192, 512, 128, 512, 8192], activation='mish',
        drop_out=[0.5, 0.5, 0.05, 0.5, 0.5], batch_normalization=True, noise=0.2,
        L1_weight=None, L2_weight=1e-3, min_weight=None, max_weight=1.0,
        batch_size=2621, num_epochs=9999, repetitions=4,
        optimizer='ADAMW', lookahead=True, learning_rate=0.015, LROPpatience=100, LROPfactor=0.9,
        loss='SMAPE', loss_threshold=29, loss_goal=29, negative_penalty_coef=0.0,
        patience=500, maape_threshold=27, maape_goal=27, ob_threshold=100,
        skip=200, never_stop=False, save_models=False, verbose=0):

    # Load training & testing [features vs targets] and prediction features
    train_features_df = pd.read_csv(train_features_filename).astype(np.float)
    train_features_non_zero_isi_df = train_features_df.copy(deep=True)
    if complete_stp:
        isi_mode = train_features_df[train_features_df[isi_column_name] > 0][isi_column_name].mode()[0]
        train_features_non_zero_isi_df[isi_column_name].loc[(train_features_non_zero_isi_df[isi_column_name] < 0.001)] = isi_mode

    # reindex data columns
    feature_names = numerical_columns + sorted(list(set(train_features_df.columns) - set(numerical_columns)))
    train_features = train_features_df.reindex(columns=feature_names)
    train_features_non_zero_isi_df = train_features_non_zero_isi_df.reindex(columns=feature_names)
    data_scaler_model = ColumnTransformer([(
        'NumericalValues', MaxAbsScaler(), numerical_columns)],
        remainder='passthrough').fit(train_features)
    train_features = data_scaler_model.transform(train_features)
    train_features_non_zero_isi = data_scaler_model.transform(train_features_non_zero_isi_df)

    # get all prediction features and normalize them
    predict_features = get_prediction_features('./', predict_data_startswith, data_scaler_model, feature_names)

    # get all training targets and normalize them
    train_targets_df = pd.read_csv(train_targets_filename).astype(np.float)
    targets_scaler_model = MinMaxScaler(feature_range=(0 + epsilon, 1 - epsilon)).fit(train_targets_df)
    train_targets = targets_scaler_model.transform(train_targets_df)

    if destination_directory is None:
        if random_forest:
            destination_directory = f'random_forest{"-JK" if jack_knife else ""}'
        else:
            destination_directory = (
                f'{"source" if source_directory or template_weights else "%02d-pass" % passage_num}'
                f'{"-" + source_directory if source_directory else ""}'
                f'{"-JK" if jack_knife else ""}'
                f'-L={num_nodes}-{activation}-{loss}-SMAAPE={maape_goal}'
                f'{"-DO=" + str(drop_out) if drop_out else ""}'  # f'{"-DO=%.2f" % drop_out if drop_out else ""}'
                f'{"-GN=%.2f" % noise if noise else ""}'
                f'{"-Min=" + str(min_weight) if min_weight else ""}'
                f'{"-Max=" + str(max_weight) if max_weight else ""}'
                f'{"-L1w=" + str(L1_weight) if L1_weight else ""}'
                f'{"-L2w=" + str(L2_weight) if L2_weight else ""}'
                f'{"-BN" if batch_normalization else ""}'
                f'-b={batch_size}'
                f'-e={num_epochs}x{repetitions}'
                f'{"-p=" + str(patience) if patience else ""}'
                f'{"-LROPp=" + str(LROPpatience) + "-LROPf=" + str(LROPfactor) if LROPpatience else ""}'
                f'{"-LA" if lookahead else ""}-{optimizer}m89-lr={learning_rate}'
            )
            destination_directory = destination_directory.replace(', ', '-')
        print(destination_directory)
        destination_directory = (os.path.join(os.getcwd(), destination_directory))

    if not os.path.exists(destination_directory):
        os.makedirs(destination_directory)
    file = None
    if source_directory and jack_knife:
        print('Jack-knife is not possible when retraining models.')
        raise StopExecution
    elif source_directory:
        for file_ in os.listdir(source_directory):
            if (file_.endswith('.h5') or file_.endswith('.tf')) and \
                    os.path.getsize(os.path.join(source_directory, file_)) > 0:
                destination_w_file_path = os.path.join(destination_directory, file_)
                destination_feature_file_path = os.path.join(
                    destination_directory, file_[0:-10] + 'feature_importance.svg')
                if os.path.exists(destination_w_file_path):
                    if os.path.getsize(destination_w_file_path) == 0 or \
                            os.path.exists(destination_feature_file_path) or \
                            feature_importance_iterations == 0:
                        if os.path.exists(destination_feature_file_path) and \
                                os.path.getsize(destination_feature_file_path) == 0 and \
                                feature_importance_iterations == 0:
                            os.remove(destination_feature_file_path)
                        continue
                else:
                    file = file
                    del file_
                    break
    else:
        if os.path.exists(os.path.join(destination_directory, 'counter.txt')):
            with open(os.path.join(destination_directory, 'counter.txt'), 'r') as counter_file:
                n = int(counter_file.read()) + 1
        else:
            n = 1
        file = f'model_{n:0>5}_weights.h5'
        with open(os.path.join(destination_directory, 'counter.txt'), 'w') as counter_file:
            counter_file.write(str(n))
        if jack_knife:
            train_features_set_aside = np.array([train_features[n - 1]])
            train_features_set_aside_df = train_features_df.iloc[[n - 1]].reset_index(drop=True)
            train_targets_set_aside_df = train_targets_df.iloc[[n - 1]].reset_index(drop=True)
            train_features = np.delete(train_features, n - 1, axis=0)
            train_targets = np.delete(train_targets, n - 1, axis=0)

    if file:
        file_name = file[0:-11]
        destination_path_file = os.path.join(destination_directory, file)
        print(f'{strftime("%H:%M:%S", localtime())} fitting {file_name}')
    result = None
    if file and not random_forest:
        callbacks = []
        if LROPpatience is not None:
            callbacks += [tf.keras.callbacks.ReduceLROnPlateau(
                monitor='loss', patience=LROPpatience, factor=LROPfactor, cooldown=0,
                min_lr=1.25e-4, verbose=0)]

        if maape_goal and loss_goal:
            callbacks += [StopGoal(
                monitor1='loss', goal1=loss_goal,
                monitor2='mean_arctan_absolute_percentage_error', goal2=maape_goal,
                patience=patience
            )]

        model = build_model(
            num_features=train_features.shape[1], num_nodes=num_nodes,
            activation=activation,
            L1_weight=L1_weight, L2_weight=L2_weight,
            min_weight=min_weight, max_weight=max_weight,
            batch_normalization=batch_normalization,
            drop_out=drop_out, noise=noise,
            optimizer=optimizer, lookahead=lookahead, learning_rate=learning_rate,
            loss=loss, negative_penalty_coef=negative_penalty_coef,
            template_model=template_model
        )

        if os.path.exists(destination_path_file) and os.path.getsize(destination_path_file) > 0:
            model.load_weights(destination_path_file, by_name=True, skip_mismatch=True)
            print(f'{file_name} already fitted\n'
                  f'loading weights from the destination folder\n')
        else:
            if source_directory:
                source_path_file = os.path.join(source_directory, file)
                if os.path.exists(source_path_file):
                    model.load_weights(source_path_file, by_name=True, skip_mismatch=True)
                    print(f'{file_name} is refitting for the 2nd time\n'
                          f'loading weights from the source folder')
            elif template_weights:
                model.load_weights(template_weights, by_name=True, skip_mismatch=True)
                print(f'template weights loaded from {template_weights}')
            elif template_model:
                print(f'template model loaded from {template_model}')

            open(destination_path_file, 'a').close()

        if validation_data:
            val_loss_score, val_maape_score = model.evaluate(*validation_data, verbose=0)
            print(f'{strftime("%H:%M:%S", localtime())} before training val_loss: {val_loss_score:0>5.2f}'
                  f'val_maape: {val_maape_score:0>5.2f}')

        loss_score, maape_score = model.evaluate(train_features, train_targets, verbose=0)
        print(
            f'{strftime("%H:%M:%S", localtime())} before training loss: {loss_score:0>5.2f} maape: {maape_score:0>5.2f}')

        i = 0
        all_are_bio_plausible = False
        while not (all_are_bio_plausible and loss_score < loss_threshold and maape_score < maape_threshold):
            i += 1
            _data_ = (train_features, train_targets) if i % 2 else (np.flipud(train_features), np.flipud(train_targets))
            result = join_results(
                result,
                model.fit(
                    *_data_, validation_data=validation_data,
                    batch_size=batch_size, epochs=num_epochs, callbacks=callbacks,
                    verbose=verbose, use_multiprocessing=True, workers=14
                )
            )
            loss_score, maape_score = model.evaluate(*_data_, verbose=0)
            out_of_bound_values = num_out_of_bound_values(
                targets_scaler_model.inverse_transform(model.predict(train_features)))
            if maape_score < maape_threshold and loss_score < loss_threshold and out_of_bound_values <= ob_threshold:
                predicted_targets = {
                    k: targets_scaler_model.inverse_transform(model.predict(v)) for k, v in predict_features.items()}
                all_are_bio_plausible = all([are_bio_plausible(v) for k, v in predicted_targets.items()])
            else:
                all_are_bio_plausible = None

            print(f'{strftime("%H:%M:%S", localtime())} '
                  f'iteration: {i:0>2} '
                  f'loss: {loss_score:0>5.2f} maape: {maape_score:0>5.2f} ob: {out_of_bound_values:0>2} '
                  f'bio-plausible: {all_are_bio_plausible} ')

            if i >= repetitions and not all_are_bio_plausible:
                print(f'\n{strftime("%H:%M:%S", localtime())} starting to train a new model')
                random.seed(a=int(time() * 10 ** 7) % 4294967295, version=2)
                tf.random.set_seed(int(time() * 10 ** 7) % 4294967295)
                np.random.seed(int(time() * 10 ** 7) % 4294967295)
                model = build_model(
                    num_features=train_features.shape[1], num_nodes=num_nodes,
                    activation=activation,
                    L1_weight=L1_weight, L2_weight=L2_weight,
                    min_weight=min_weight, max_weight=max_weight,
                    batch_normalization=batch_normalization,
                    drop_out=drop_out, noise=noise,
                    optimizer=optimizer, lookahead=lookahead, learning_rate=learning_rate,
                    loss=loss, negative_penalty_coef=negative_penalty_coef,
                    template_model=template_model
                )
                result = None
                i = 0

    elif random_forest:
        model = RandomForestRegressor(n_estimators=10000, n_jobs=os.cpu_count())
        model.fit(train_features, train_targets)
        print(f'coefficient of determination was {model.score(train_features, train_targets):.3f}')
    else:
        print('could not make any model')
        if not never_stop:
            raise StopExecution
        return None

    print(f'{strftime("%H:%M:%S", localtime())} training ended')

    if not (random_forest and jack_knife):
        predicted_targets = {
            k: targets_scaler_model.inverse_transform(model.predict(v)) for k, v in predict_features.items()}
        all_predicted_targets = np.concatenate([v for k, v in predicted_targets.items()])

    if not random_forest:
        plot_result(
            result, loss_score, maape_score, out_of_bound_values,
            predicted_targets=all_predicted_targets, skip=skip,
            save_path_name=os.path.join(destination_directory, file_name + '_result'))

    if jack_knife:
        predicted_targets_set_aside = targets_scaler_model.inverse_transform(model.predict(train_features_set_aside))
        pd.concat([
            pd.DataFrame([n], columns=['n']),
            pd.DataFrame(predicted_targets_set_aside,
                         columns=['g.model', 'tau_d.model', 'tau_r.model', 'tau_f.model', 'U.model']),
            train_targets_set_aside_df,
            train_features_set_aside_df
        ], axis=1).to_csv(destination_path_file[0:-3] + '.csv', index=False)
    else:
        if save_models and not random_forest:
            model.save_weights(destination_path_file, save_format='h5', overwrite=True)
            model.save(destination_path_file[0:-11] + '.h5', overwrite=True)
        elif os.path.exists(destination_path_file) and os.path.getsize(destination_path_file) == 0:
            os.remove(destination_path_file)
        for prediction_name, prediction in predicted_targets.items():
            destination_csv_file = os.path.join(destination_directory, file_name + prediction_name)
            destination_matrix_plot = os.path.join(
                destination_directory, file_name + prediction_name[0:-4] + '_matrixplot.png')

            df = pd.concat([
                potential_connections_df,
                pd.DataFrame(prediction, columns=['g', 'tau_d', 'tau_r', 'tau_f', 'U'])
            ], axis=1)

            if not os.path.exists(destination_csv_file):
                df.to_csv(destination_csv_file, index=False)
            else:
                df = pd.read_csv(destination_csv_file)

            if matrix_plot_is_needed and not os.path.exists(destination_matrix_plot):
                matrix_plot(df, destination_matrix_plot)

        learned_targets_path = os.path.join(destination_directory, file_name + '_learned_targets.csv')
        if not os.path.exists(learned_targets_path):
            pd.DataFrame(
                targets_scaler_model.inverse_transform(model.predict(train_features)),
                columns=['g', 'tau_d', 'tau_r', 'tau_f', 'U']
            ).to_csv(learned_targets_path, index=False)

        learned_targets_path = os.path.join(destination_directory,
                                            file_name + '_learned_targets_na_interpolated.csv')
        if not os.path.exists(learned_targets_path):
            pd.DataFrame(
                targets_scaler_model.inverse_transform(model.predict(train_features_non_zero_isi)),
                columns=['g', 'tau_d', 'tau_r', 'tau_f', 'U']
            ).to_csv(learned_targets_path, index=False)

        destination_ft_importance_file = os.path.join(destination_directory, file_name + '_feature_importance.svg')
        if feature_importance_iterations > 0 and not os.path.exists(destination_ft_importance_file):
            print(f'{file_name} needs feature importance calculation')
            open(destination_ft_importance_file, 'a').close()
            if not random_forest:
                perm = PermutationImportance(
                    KerasRegressorLoadable(
                        model=model,
                        build_fn=build_model,
                        num_features=train_features.shape[1], num_nodes=num_nodes,
                        activation=activation,
                        L1_weight=L1_weight, L2_weight=L2_weight,
                        min_value=min_weight, max_value=max_weight,
                        batch_normalization=batch_normalization,
                        drop_out=drop_out, noise=noise,
                        optimizer=optimizer, lookahead=lookahead, learning_rate=learning_rate,
                        loss=loss, negative_penalty_coef=negative_penalty_coef,
                        epochs=num_epochs, batch_size=batch_size,
                        verbose=0, use_multiprocessing=True, workers=14,
                        callbacks=callbacks),
                    random_state=1, n_iter=feature_importance_iterations
                ).fit(train_features, train_targets)
            else:
                perm = Perm()
                perm.feature_importances_ = model.feature_importances_
                perm.feature_importances_std_ = np.zeros(len(model.feature_importances_))

            plot_feature_importance(
                perm, feature_names, len(feature_names),
                title=file_name, save_path_name=destination_ft_importance_file)

    if not source_directory and os.path.exists(destination_path_file) and \
            os.path.getsize(destination_path_file) == 0:
        os.remove(destination_path_file)
    return model


def k_means_cross_validation(
        train_features_filename, train_targets_filename, 
        numerical_columns = [# List of feature columns that need normalization
                'Slice_Thickness', 'ISI', 'Temperature', 'Vm', 'Erev_GABA_B', 'Erev_NMDA',
                'Erev_GABA_A', 'Erev_AMPA', 'Cai', 'Cao', 'Cli', 'Clo', 'Csi', 'H2PO4o',
                'HCO3i', 'HCO3o', 'HEPESi', 'Ki', 'Ko', 'Mgi', 'Mgo', 'Nai', 'Nao', 'Bri',
                'gluconatei', 'QX314i', 'ATPi', 'EGTAi', 'EGTAo', 'GTPi', 'OHi', 'SO4i',
                'SO4o', 'phosphocreatinei', 'methanesulfonatei', 'acetatei', 'methylsulfatei',
                'NMDGi', 'Trisi', 'CeSO4i', 'pyruvateo', 'TEAi', 'Bao', 'HPO4o', 'Age'],
        k=4,
        num_nodes=[8192, 512, 128, 512, 8192], activation='mish',
        drop_out=[0.5, 0.25, 0.05, 0.25, 0.5], batch_normalization=True, noise=0.2,
        L1_weight=None, L2_weight=1e-3, min_weight=None, max_weight=1.0,
        optimizer='ADAMW', lookahead=True, loss='SMAPE', negative_penalty_coef=0.0,
        learning_rate=0.01, LROPpatience=100, LROPfactor=0.9,
        num_epochs=10000, batch_size=2621, template_model=None, skip=500
):
    
    # Load training & testing [features vs targets] and prediction features
    train_features_df = pd.read_csv(train_features_filename).astype(np.float)
    # reindex data columns
    feature_names = numerical_columns + sorted(list(set(train_features_df.columns) - set(numerical_columns)))
    train_features = train_features_df.reindex(columns=feature_names)
    data_scaler_model = ColumnTransformer([(
        'NumericalValues', MaxAbsScaler(), numerical_columns)],
        remainder='passthrough').fit(train_features)
    train_features = data_scaler_model.transform(train_features)

    # get all training targets and normalize them
    train_targets_df = pd.read_csv(train_targets_filename).astype(np.float)
    targets_scaler_model = MinMaxScaler(feature_range=(0 + epsilon, 1 - epsilon)).fit(train_targets_df)
    train_targets = targets_scaler_model.transform(train_targets_df)

    callbacks = None
    if LROPpatience is not None:
        callbacks = [tf.keras.callbacks.ReduceLROnPlateau(
            monitor='loss', patience=LROPpatience, factor=LROPfactor, cooldown=0,
            min_lr=1.25e-4, verbose=0)]
    data_features, data_targets = shuffle(train_features, train_targets, random_state=20)
    all_maape_histories, all_val_maape_histories = [], []
    num_val_samples = len(data_features) // k
    for i in range(k):
        print(f'{strftime("%H:%M:%S", localtime())} starting processing fold #{i + 1}')
        val_data = data_features[i * num_val_samples: (i + 1) * num_val_samples]
        val_targets = data_targets[i * num_val_samples: (i + 1) * num_val_samples]
        partial_train_data = np.concatenate((
            data_features[:i * num_val_samples], data_features[(i + 1) * num_val_samples:]), axis=0)
        partial_train_targets = np.concatenate((
            data_targets[:i * num_val_samples], data_targets[(i + 1) * num_val_samples:]), axis=0)
        model = build_model(
            num_features=data_features.shape[1], num_nodes=num_nodes,
            activation=activation,
            L1_weight=L1_weight, L2_weight=L2_weight,
            min_weight=min_weight, max_weight=max_weight,
            batch_normalization=batch_normalization,
            drop_out=drop_out, noise=noise,
            optimizer=optimizer, lookahead=lookahead, learning_rate=learning_rate,
            loss=loss, negative_penalty_coef=negative_penalty_coef,
            template_model=template_model
        )
        result = model.fit(
            partial_train_data, partial_train_targets,
            validation_data=(val_data, val_targets),
            epochs=num_epochs, batch_size=batch_size, verbose=0, callbacks=callbacks
        )
        all_maape_histories.append(result.history['mean_arctan_absolute_percentage_error'])
        all_val_maape_histories.append(result.history['val_mean_arctan_absolute_percentage_error'])
    print(f'{strftime("%H:%M:%S", localtime())} done')
    del k, num_val_samples
    average_maape_history = [np.mean([x[i] for x in all_maape_histories]) for i in range(num_epochs)]
    average_val_maape_history = [np.mean([x[i] for x in all_val_maape_histories]) for i in range(num_epochs)]
    smooth_maape_history = smooth_curve(average_maape_history[skip:], factor=0.9)
    smooth_val_maape_history = smooth_curve(average_val_maape_history[skip:], factor=0.9)
    epochs = np.array(range(1, len(smooth_maape_history) + 1)) + skip
    fig = plt.figure(figsize=(15, 5))
    ax1 = plt.subplot(1, 2, 1)
    plt.plot(epochs, smooth_maape_history, 'm', label='Training')
    plt.plot(epochs, smooth_val_maape_history, 'b', label='Validation')
    plt.xlabel('Epochs')
    plt.ylabel(loss)
    # plt.yscale('log')
    plt.legend()
    plt.title(f'e={num_epochs}, b={batch_size}, lr={learning_rate}-{num_nodes}')
    min_maape = np.nanmin(smooth_val_maape_history)
    try:
        min_maape_index = str(np.where(np.array(smooth_val_maape_history) == min_maape)[0][0] + skip)
    except:
        min_maape_index = 'not found'
    plt.text(
        0.2, 0.9,
        f'min was {min_maape:.2f} at epoch {min_maape_index}',
        fontweight='bold', transform=ax1.transAxes)
    plt.show()


def get_prediction_features(folder, name_starts, normalizer, names_):
    predict_data_ = dict()
    for path, dirs, files in os.walk(folder):
        for file in files:
            if file.endswith('csv') and file.startswith(name_starts):
                predict_data_[file] = normalizer.transform(
                    pd.read_csv(os.path.join(path, file)).astype(np.float).reindex(columns=names_))
    return predict_data_


print(f'TensorFlow version is {tf.__version__}')
