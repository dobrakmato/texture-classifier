import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import MultiLabelBinarizer
from tensorflow.keras import layers, models
import wandb
from wandb.keras import WandbCallback

IMAGE_SIZE = 64

# load X, y data
X = np.load(f'./imgs_{IMAGE_SIZE}_3.npy', allow_pickle=True)
y = np.load(f'./labels_{IMAGE_SIZE}_3.npy', allow_pickle=True)
print(X.shape)
print(y.shape)

# transform the labels to binary representation so that we can train on the data
mlb = MultiLabelBinarizer()
y = mlb.fit_transform(y)

print(list(mlb.classes_))


def get_model():
    model = models.Sequential()
    model.add(layers.Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 3)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(256, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Flatten())
    model.add(layers.Dense(2048, activation='relu'))
    model.add(layers.Dense(len(mlb.classes_), activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


skf = KFold(n_splits=3, random_state=7, shuffle=True)
wandb.init(project="texture-classification", entity="dobrakmato",
           config={'image_size': IMAGE_SIZE, 'labels': len(mlb.classes_), 'source_dtype': X.dtype,
                   'kfold_splits': skf.n_splits})

model = get_model()
model.fit(X, y, epochs=250, callbacks=[WandbCallback()], validation_split=0.2)

exit(0)

results = []
for train_index, test_index in skf.split(X, y):
    X_train = X[train_index]
    y_train = y[train_index]
    X_test = X[test_index]
    y_test = y[test_index]

    print(len(y_train), len(y_test))

    model = get_model()
    model.fit(X_train, y_train, epochs=100, callbacks=[WandbCallback()], batch_size=200, validation_split=0.2)
    yhat = model.predict(X_test)
    yhat = yhat.round()
    acc = accuracy_score(y_test, yhat)
    print('test acc %.3f' % acc)
    results.append(acc)
print('Accuracy: %.3f (%.3f)' % (np.mean(results), np.std(results)))
