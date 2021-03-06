import tensorflow as tf
import tensorflow_addons as tfa
import tensorflow.keras as keras
from tensorflow.keras.layers import *
from tensorflow.keras.utils import plot_model


# ==============================================================================
# =                                  networks                                  =
# ==============================================================================

def _get_norm_layer(norm):
    if norm == 'none':
        return lambda: lambda x: x
    elif norm == 'batch_norm':
        return keras.layers.BatchNormalization
    elif norm == 'instance_norm':
        return tfa.layers.InstanceNormalization
    elif norm == 'layer_norm':
        return keras.layers.LayerNormalization


def ResnetGenerator(input_shape=(227, 227, 3),
                    output_channels=3,
                    dim=64,
                    n_downsamplings=2,
                    n_blocks=9,
                    norm='instance_norm',
                    attention=False):
    if attention:
        output_channels = output_channels + 1
    Norm = _get_norm_layer(norm)

    # 受保护的用法
    def _residual_block(x):
        dim = x.shape[-1]
        h = x

        # 为什么这里不用padding参数呢？使用到了‘REFLECT’
        h = tf.pad(h, [[0, 0], [1, 1], [1, 1], [0, 0]], mode='REFLECT')

        h = keras.layers.Conv2D(dim, 3, padding='valid', use_bias=False)(h)
        h = Norm()(h)
        h = tf.nn.relu(h)

        h = tf.pad(h, [[0, 0], [1, 1], [1, 1], [0, 0]], mode='REFLECT')
        h = keras.layers.Conv2D(dim, 3, padding='valid', use_bias=False)(h)
        h = Norm()(h)

        return keras.layers.add([x, h])

    # 0
    h = inputs = keras.Input(shape=input_shape)

    # 1
    h = tf.pad(h, [[0, 0], [3, 3], [3, 3], [0, 0]], mode='REFLECT')
    h = keras.layers.Conv2D(dim, 7, padding='valid', use_bias=False)(h)
    h = Norm()(h)
    h = tf.nn.relu(h)

    # 2
    for _ in range(n_downsamplings):
        dim *= 2
        h = keras.layers.Conv2D(dim, 3, strides=2, padding='same', use_bias=False)(h)
        h = Norm()(h)
        h = tf.nn.relu(h)

    # 3
    for _ in range(n_blocks):
        h = _residual_block(h)

    # 4
    for _ in range(n_downsamplings):
        dim //= 2
        h = tf.pad(h, [[0, 0], [3, 3], [3, 3], [0, 0]], mode='REFLECT')
        h = keras.layers.Conv2D(dim, 3, strides=1, padding='valid', use_bias=False)(h)
        h = keras.layers.Conv2D(dim, 5, strides=1, padding='valid', use_bias=False)(h)
        h = keras.layers.Conv2DTranspose(dim, 3, strides=2, padding='same', use_bias=False)(h)
        h = Norm()(h)
        h = tf.nn.relu(h)
        h = keras.layers.Conv2DTranspose(dim, 3, strides=1, padding='same', use_bias=False)(h)
        h = Norm()(h)
        h = tf.nn.relu(h)

    # 5
    h = tf.pad(h, [[0, 0], [3, 3], [3, 3], [0, 0]], mode='REFLECT')
    h = keras.layers.Conv2D(output_channels, 8, padding='valid')(h)
    if not attention:
        h = tf.tanh(h)
        return keras.Model(inputs=inputs, outputs=h)
    # 假如我不添加tanh的话，又会出现报错
    if attention:

        attention_mask = h[:, :, :, 0:1]
        attention_mask = tf.pad(attention_mask, [[0, 0], [1, 1], [1, 1], [0, 0]], mode='REFLECT')
        attention_mask = Conv2D(64, (3, 3), (1, 1), 'same', use_bias=False)(attention_mask)
        attention_mask = Norm()(attention_mask)
        attention_mask = tf.nn.relu(attention_mask)
        attention_mask = Conv2D(3, (3, 3), (1, 1), 'valid', use_bias=False)(attention_mask)
        attention_mask = Norm()(attention_mask)
        attention_mask = tf.sigmoid(attention_mask)
        # attention_mask = tf.expand_dims(attention_mask, axis=3)
        # attention_mask = tf.concat([attention_mask, attention_mask, attention_mask], axis=3)

        # 上述式子对应的式attention_v2，在此次实验中可以发现裂缝都被涂上了绿色，可以以此来做无监督学习
        # 应该对上述式子进行更进一步的研究
        # attention_mask = tf.sigmoid(h[:, :, :, 0])  # 91

        content_mask = h[:, :, :, 1:]
        content_mask = tf.tanh(content_mask)
        h = content_mask * attention_mask + inputs * (1 - attention_mask)

        return keras.Model(inputs=inputs, outputs=[h, attention_mask])


attention_mask, content_mask = None, None


def AttentionCycleGAN_v1_Generator(input_shape=(227, 227, 3), output_channel=3,
                                   n_downsampling=2, n_ResBlock=9,
                                   norm='batch_norm', attention=False):
    global attention_mask, content_mask
    Norm = _get_norm_layer(norm)
    a = keras.Input(shape=input_shape)
    h = tf.pad(a, [[0, 0], [3, 3], [3, 3], [0, 0]], mode='REFLECT')

    def model_layer_1(y):
        y = Conv2D(64, (7, 7), (1, 1), 'valid')(y)
        y = Norm()(y)
        return y

    h = model_layer_1(h)

    n_downsampling = n_downsampling
    n_ResBlock = n_ResBlock
    if attention:
        output_channel = output_channel + 1

    for i in range(n_downsampling):
        mult = 2 ** i

        def model_layer_2(y):
            y = Conv2D(64 * mult * 2, (3, 3), (2, 2), 'same')(y)
            y = Norm()(y)
            y = ReLU()(y)
            return y

        h = model_layer_2(h)

    mult = 2 ** n_downsampling

    for i in range(n_ResBlock):
        x = h

        def model_layer_3(y):
            y = Conv2D(64 * mult, (3, 3), padding='valid')(y)
            y = Norm()(y)
            return y

        h = tf.pad(h, [[0, 0], [1, 1], [1, 1], [0, 0]], mode='REFLECT')
        h = model_layer_3(h)
        h = ReLU()(h)
        h = tf.pad(h, [[0, 0], [1, 1], [1, 1], [0, 0]], mode='REFLECT')
        h = model_layer_3(h)

        h = keras.layers.add([x, h])

    upsampling = n_downsampling

    for i in range(upsampling):
        def model_layer_4(y):
            y = Conv2DTranspose(64 * mult / 2, (3, 3), (2, 2), 'same')(y)
            y = Norm()(y)
            y = ReLU()(y)
            return y
        h = model_layer_4(h)
        mult = mult / 2

    h = tf.pad(h, [[0, 0], [3, 3], [3, 3], [0, 0]], mode='REFLECT')
    h = Conv2D(output_channel, (8, 8), (1, 1), 'valid')(h)
    result_layer = h
    if attention:
        attention_mask = tf.sigmoid(h[:, :, :, :1])
        content_mask = h[:, :, :, 1:]
        attention_mask = tf.concat([attention_mask, attention_mask, attention_mask], axis=3)
        result_layer = content_mask * attention_mask + a * (1 - attention_mask)

        return keras.Model(inputs=a, outputs=[result_layer, attention_mask, content_mask])
    return keras.Model(inputs=a, outputs=result_layer)




def ConvDiscriminator(input_shape=(256, 256, 3),
                      dim=64,
                      n_downsamplings=3,
                      norm='instance_norm'):
    dim_ = dim
    Norm = _get_norm_layer(norm)

    # 0
    h = inputs = keras.Input(shape=input_shape)

    # 1
    h = keras.layers.Conv2D(dim, 4, strides=2, padding='same')(h)
    h = tf.nn.leaky_relu(h, alpha=0.2)

    for _ in range(n_downsamplings - 1):
        dim = min(dim * 2, dim_ * 8)
        h = keras.layers.Conv2D(dim, 4, strides=2, padding='same', use_bias=False)(h)
        h = Norm()(h)
        h = tf.nn.leaky_relu(h, alpha=0.2)

    # 2
    dim = min(dim * 2, dim_ * 8)
    h = keras.layers.Conv2D(dim, 4, strides=1, padding='same', use_bias=False)(h)
    h = Norm()(h)
    h = tf.nn.leaky_relu(h, alpha=0.2)

    # 3
    h = keras.layers.Conv2D(1, 4, strides=1, padding='same')(h)

    return keras.Model(inputs=inputs, outputs=h)


# ==============================================================================
# =                          learning rate scheduler                           =
# ==============================================================================

class LinearDecay(keras.optimizers.schedules.LearningRateSchedule):
    # if `step` < `step_decay`: use fixed learning rate
    # else: linearly decay the learning rate to zero

    def __init__(self, initial_learning_rate, total_steps, step_decay):
        super(LinearDecay, self).__init__()
        self._initial_learning_rate = initial_learning_rate
        self._steps = total_steps
        self._step_decay = step_decay
        self.current_learning_rate = tf.Variable(initial_value=initial_learning_rate, trainable=False, dtype=tf.float32)

    def __call__(self, step):
        self.current_learning_rate.assign(tf.cond(
            step >= self._step_decay,
            true_fn=lambda: self._initial_learning_rate * (
                        1 - 1 / (self._steps - self._step_decay) * (step - self._step_decay)),
            false_fn=lambda: self._initial_learning_rate
        ))
        return self.current_learning_rate
