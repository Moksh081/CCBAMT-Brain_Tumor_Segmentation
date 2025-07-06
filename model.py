##Model - 
def channel_attention(input_feature, ratio=8):
    channel = input_feature.shape[-1]

    shared_layer_one = layers.Dense(channel // ratio, activation='relu', kernel_initializer='he_normal', use_bias=True, bias_initializer='zeros')
    shared_layer_two = layers.Dense(channel, kernel_initializer='he_normal', use_bias=True, bias_initializer='zeros')

    avg_pool = layers.GlobalAveragePooling2D()(input_feature)
    avg_pool = layers.Reshape((1, 1, channel))(avg_pool)
    avg_pool = shared_layer_one(avg_pool)
    avg_pool = shared_layer_two(avg_pool)

    max_pool = layers.GlobalMaxPooling2D()(input_feature)
    max_pool = layers.Reshape((1, 1, channel))(max_pool)
    max_pool = shared_layer_one(max_pool)
    max_pool = shared_layer_two(max_pool)

    cbam_feature = layers.Add()([avg_pool, max_pool])
    cbam_feature = layers.Activation('sigmoid')(cbam_feature)

    return layers.multiply([input_feature, cbam_feature])


def spatial_attention(input_feature):
    kernel_size = 7

    avg_pool = layers.Lambda(lambda x: tf.reduce_mean(x, axis=-1, keepdims=True))(input_feature)
    max_pool = layers.Lambda(lambda x: tf.reduce_max(x, axis=-1, keepdims=True))(input_feature)
    concat = layers.Concatenate(axis=-1)([avg_pool, max_pool])
    
    cbam_feature = layers.Conv2D(filters=1, kernel_size=kernel_size, strides=1, padding='same', activation='sigmoid', kernel_initializer='he_normal', use_bias=False)(concat)

    return layers.multiply([input_feature, cbam_feature])

def CBAM(input_feature, ratio=8):
    ca = channel_attention(input_feature, ratio)
    sa = spatial_attention(ca)
    return sa

import tensorflow as tf
from tensorflow.keras import layers, Model, Input
from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, MaxPooling2D, UpSampling2D, concatenate

# CBAM integration
def Convolution(input_tensor, filters):
    x = Conv2D(filters=filters, kernel_size=(3, 3), padding='same', strides=(1, 1))(input_tensor)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = CBAM(x)  # Apply CBAM
    return x

# Transformer Encoder Block
def TransformerBlock(x, num_heads, ff_dim, dropout=0.1):
    attention_output = layers.MultiHeadAttention(num_heads=num_heads, key_dim=ff_dim)(x, x)
    attention_output = layers.Dropout(dropout)(attention_output)
    attention_output = layers.LayerNormalization(epsilon=1e-6)(attention_output + x)

    ffn = layers.Dense(ff_dim, activation='relu')(attention_output)
    ffn = layers.Dense(x.shape[-1])(ffn)
    ffn = layers.Dropout(dropout)(ffn)

    x = layers.LayerNormalization(epsilon=1e-6)(attention_output + ffn)
    return x

# Model definition with Transformer block and CBAM
def create_model(input_shape):
    inputs = Input((input_shape))

    conv_1 = Convolution(inputs, 32)
    maxp_1 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(conv_1)

    conv_2 = Convolution(maxp_1, 64)
    maxp_2 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(conv_2)

    conv_3 = Convolution(maxp_2, 128)
    maxp_3 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(conv_3)

    conv_4 = Convolution(maxp_3, 256)
    maxp_4 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(conv_4)

    conv_5 = Convolution(maxp_4, 512)

    reshaped = layers.Reshape((-1, 512))(conv_5)

    transformer_output = TransformerBlock(reshaped, num_heads=8, ff_dim=512)

    reshaped_back = layers.Reshape((conv_5.shape[1], conv_5.shape[2], 512))(transformer_output)

    upsample_6 = UpSampling2D((2, 2))(reshaped_back)
    conv_6 = Convolution(upsample_6, 256)

    upsample_7 = UpSampling2D((2, 2))(conv_6)
    upsample_7 = concatenate([upsample_7, conv_3])

    conv_7 = Convolution(upsample_7, 128)
    upsample_8 = UpSampling2D((2, 2))(conv_7)

    conv_8 = Convolution(upsample_8, 64)
    upsample_9 = UpSampling2D((2, 2))(conv_8)
    upsample_9 = concatenate([upsample_9, conv_1])
    conv_9 = Convolution(upsample_9, 32)
    outputs = Conv2D(1, (1, 1), activation='sigmoid')(conv_9)

    model = Model(inputs=[inputs], outputs=[outputs])

    return model

from tensorflow.keras.utils import plot_model
from tensorflow.keras.models import load_model

# Save the model summary image
model = create_model(input_shape=(240, 240, 1))
model.summary()
model.save('HYBRID.h5')