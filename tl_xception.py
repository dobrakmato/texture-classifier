import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer
from tensorflow.keras import layers, models, applications
import wandb
from wandb.keras import WandbCallback

IMAGE_SIZE = 150

# load X, y data
X = np.load(f'./imgs_{IMAGE_SIZE}_3.npy', allow_pickle=True)
y = np.load(f'./labels_{IMAGE_SIZE}_3.npy', allow_pickle=True)
print(X.shape)
print(y.shape)

# transform the labels to binary representation so that we can train on the data
mlb = MultiLabelBinarizer()
y = mlb.fit_transform(y)

print(list(mlb.classes_))

X = applications.xception.preprocess_input(X)

base_model = applications.Xception(weights='imagenet', input_shape=(150, 150, 3), include_top=False)
base_model.trainable = False

inputs = layers.Input(shape=(150, 150, 3))
x = base_model(inputs, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Flatten()(x)
x = layers.Dense(2048, activation='relu')(x)
outputs = layers.Dense(len(mlb.classes_), activation='sigmoid')(x)
model = models.Model(inputs, outputs)
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

wandb.init(project="texture-classification", entity="dobrakmato",
           config={'image_size': IMAGE_SIZE, 'labels': len(mlb.classes_), 'source_dtype': X.dtype,
                   'base_model': 'Xception'})

model.fit(X, y, epochs=250, callbacks=[WandbCallback()], validation_split=0.2)
