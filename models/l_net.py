import tensorflow as tf
from tensorflow import keras

def L_net(pretrained_weights = None,
          input_size = (128,128,1),
          embedding_size = 128,
          lambda_normalization = False,
          concatenation = False,
          fully_conv=False):
  inputs = keras.Input(input_size)
  conv1 = keras.layers.Conv2D(16, 3, strides=(1,1), activation = 'relu', padding = 'same', name='conv1-1')(inputs)
  conv1 = keras.layers.Conv2D(16, 3, strides=(1,1), activation = 'relu', padding = 'same', name='conv1-2')(conv1)
  pool1 = keras.layers.MaxPooling2D(pool_size=(2, 2), name='pool1')(conv1)

  conv2 = keras.layers.Conv2D(32, 3, strides=(1,1), activation = 'relu', padding = 'same', name='conv2-1')(pool1)
  conv2 = keras.layers.Conv2D(32, 3, strides=(1,1), activation = 'relu', padding = 'same', name='conv2-2')(conv2)
  pool2 = keras.layers.MaxPooling2D(pool_size=(2, 2), name='pool2')(conv2)

  conv3 = keras.layers.Conv2D(64, 3, strides=(1,1), activation = 'relu', padding = 'same', name='conv3-1')(pool2)
  conv3 = keras.layers.Conv2D(64, 3, strides=(1,1), activation = 'relu', padding = 'same', name='conv3-2')(conv3)
  pool3 = keras.layers.MaxPooling2D(pool_size=(2, 2), name='pool3')(conv3)

  conv4 = keras.layers.Conv2D(128, 3, strides=(1,1), activation = 'relu', padding = 'same', name='conv4-1')(pool3)
  conv4 = keras.layers.Conv2D(128, 3, strides=(1,1), activation = 'relu', padding = 'same', name='conv4-2')(conv4)
  pool4 = keras.layers.MaxPooling2D(pool_size=(2, 2), name='pool4')(conv4)

  flatten = keras.layers.Flatten()(pool4)

  if fully_conv and lambda_normalization:
    conv5 = keras.layers.Conv2D(embedding_size, 3, strides=(1,1), padding = 'same')(pool4)
    rs = keras.layers.Lambda(lambda x: tf.keras.backend.l2_normalize(x, axis=1), name='embeddings')(conv5)
  if fully_conv and (not lambda_normalization):
    rs = keras.layers.Conv2D(embedding_size, 3, strides=(1,1), padding = 'same', name='embeddings')(pool4)
  if (not fully_conv) and lambda_normalization:
    dense = keras.layers.Dense(embedding_size)(flatten)
    emb = keras.layers.Lambda(lambda x: tf.keras.backend.l2_normalize(x, axis=1), name='embedding')(dense)
    dec = keras.layers.Dense(8*8*128, name='dense-dec')(emb)
    rs = keras.layers.Reshape((8,8,128), name='reshape')(dec)
  if (not fully_conv) and (not lambda_normalization):
    emb = keras.layers.Dense(embedding_size, name='embeddings')(flatten)
    dec = keras.layers.Dense(8*8*128, name='dense-dec')(emb)
    rs = keras.layers.Reshape((8,8,128), name='reshape')(dec)

  up1 = keras.layers.Conv2DTranspose(128, 3, strides=(1,1), activation='relu', padding='same', name='up1-1')(keras.layers.UpSampling2D(size = (2,2))(rs))
  if concatenation:
    up1 = keras.layers.concatenate([pool3,up1], axis = 3, name='concat1')
  up1 = keras.layers.Conv2DTranspose(128, 3, strides=(1,1), activation='relu', padding='same', name='up1-2')(up1)

  up2 = keras.layers.Conv2DTranspose(64, 3, strides=(1,1), activation='relu', padding='same', name='up2-1')(keras.layers.UpSampling2D(size = (2,2))(up1))
  if concatenation:
    up2 = keras.layers.concatenate([pool2,up2], axis = 3, name='concat2')
  up2 = keras.layers.Conv2DTranspose(64, 3, strides=(1,1), activation='relu', padding='same', name='up2-2')(up2)

  up3 = keras.layers.Conv2DTranspose(32, 3, strides=(1,1), activation='relu', padding='same', name='up3-1')(keras.layers.UpSampling2D(size = (2,2))(up2))
  if concatenation:
    up3 = keras.layers.concatenate([pool1,up3], axis = 3, name='concat3')
  up3 = keras.layers.Conv2DTranspose(32, 3, strides=(1,1), activation='relu', padding='same', name='up3-2')(up3)

  up4 = keras.layers.Conv2DTranspose(16, 3, strides=(1,1), activation='relu', padding='same', name='up4-1')(keras.layers.UpSampling2D(size = (2,2))(up3))
  up4 = keras.layers.Conv2DTranspose(16, 3, strides=(1,1), activation='relu', padding='same', name='up4-2')(up4)
  up4 = keras.layers.Conv2DTranspose(1, 3, strides=(1,1), activation='relu', padding='same', name='up4-3')(up4)
  model=keras.models.Model(inputs=inputs, outputs=up4)
  model.compile(optimizer='adam', loss='mse')
  model.summary()
  return model

