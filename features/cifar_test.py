import dtlpy as dl
from PIL import Image
import numpy as np
import torchvision
import torch
import clip
import tqdm
import umap
import matplotlib.pyplot as plt
import tensorflow as tf
import json
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
import scipy
import pandas as pd

batch_size = 4

cifar10 = tf.keras.datasets.cifar10
(X_train, y_train), (X_test, y_test) = cifar10.load_data()

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)


def load():
    checkpoint = torch.load("weights/model_10.pt")
    # Use these 3 lines if you use default model setting(not training setting) of the clip. For example, if you set context_length to 100 since your string is very long during training, then assign 100 to checkpoint['model_state_dict']["context_length"]
    # checkpoint['model_state_dict']["input_resolution"] = model.input_resolution  # default is 224
    # checkpoint['model_state_dict']["context_length"] = model.context_length  # default is 77
    # checkpoint['model_state_dict']["vocab_size"] = model.vocab_size
    model.load_state_dict(checkpoint['model_state_dict'])


pbar = tqdm.tqdm(total=X_train.data.shape[0])
features = list()
with torch.no_grad():
    count = 0
    for orig_image, target in zip(X_train, y_train):
        # orig_image = Image.open(filepath)
        image = preprocess(Image.fromarray(orig_image)).unsqueeze(0).to(device)
        image_features = model.encode_image(image)
        output = image_features[0].cpu().detach().numpy().tolist()
        features.append({'feature': output,
                         'label': target})
        # print(i_filename)
        pbar.update()
        count += 1
        if count > 10000:
            break
    # assert False


def reduction():
    vectors = [a['feature'] for a in features]
    labels = [a['label'] for a in features]
    embedding = umap.UMAP(n_neighbors=100,
                          n_epochs=100,
                          min_dist=5,
                          spread=5,
                          metric="euclidean").fit_transform(np.asarray(vectors))

    x = embedding[:, 0]
    y = embedding[:, 1]

    x -= np.min(x)
    x /= np.max(x)
    x -= 0.5
    x *= 2

    y -= np.min(y)
    y /= np.max(y)
    y -= 0.5
    y *= 2
    plt.figure()
    plt.scatter(x=x, y=y, c=labels)
    plt.show()


def selection():
    def ent(data):
        """Calculates entropy of the passed `pd.Series`
        """
        p_data = data.value_counts()  # counts occurrence of each value
        entropy = scipy.stats.entropy(p_data)  # get entropy from counts
        return entropy

    df = pd.DataFrame(np.asarray(list([v['feature'] for v in features])))
    print(ent(df))
    selected = df[:, 0].copy()


def train():
    import os

    def create_model(n_classes=10):
        base_model = tf.keras.applications.vgg16.VGG16(weights='imagenet',
                                                       classes=n_classes,
                                                       include_top=False,
                                                       input_shape=(224, 224, 3))
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(256, activation='relu')(x)
        predictions = Dense(n_classes, activation='softmax')(x)
        return Model(inputs=base_model.input, outputs=predictions)

    def make_data_generators(img_size=(224, 224), data_path='/home/develooper/cifar10-distilled',
                             selection_path="/home/develooper/distillator-pkg/final_selections.json"):
        with open(selection_path, 'r') as f:
            selections = json.load(f)
        cifar10 = tf.keras.datasets.cifar10
        (X_train, y_train), (X_test, y_test) = cifar10.load_data()
        X_train_sub = X_train[selections]
        y_train_sub = y_train[selections]
        y_test = tf.keras.utils.to_categorical(y_test)
        y_train_sub = tf.keras.utils.to_categorical(y_train_sub)
        trdata = tf.keras.preprocessing.image.ImageDataGenerator(
            featurewise_center=True,
            featurewise_std_normalization=True,
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            horizontal_flip=True
        )
        trdata.fit(X_train_sub)
        train_data = trdata.flow(X_train_sub, y_train_sub, batch_size=25)
        tsdata = tf.keras.preprocessing.image.ImageDataGenerator(
            featurewise_center=True,
            featurewise_std_normalization=True,
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            horizontal_flip=True,
            validation_split=0.2
        )
        tsdata.fit(X_test)
        test_data = tsdata.flow(X_test, y_test, batch_size=25)
        return train_data, test_data

    class ValidationCallback(tf.keras.callbacks.Callback):
        def __init__(self, validation_data, validation_interval):
            super(ValidationCallback, self).__init__()
            self.validation_data = validation_data
            self.validation_interval = validation_interval
            self.predictions = []

        def on_batch_end(self, batch, logs=None):
            if (batch + 1) % self.validation_interval == 0:
                print("Running validation")
                results = self.model.evaluate(self.validation_data)
                print(f'Validation results: {results}')
                self.predictions.append(results)

    def main():
        model = create_model()
        train_data, test_data = make_data_generators()
        opt = tf.keras.optimizers.Adam(learning_rate=1.0e-6)
        model.compile(optimizer=opt, loss='categorical_crossentropy',
                      metrics=['accuracy'])
        checkpoint = tf.keras.callbacks.ModelCheckpoint("vgg16_random.h5",
                                                        monitor='val_accuracy',
                                                        verbose=1,
                                                        save_best_only=True,
                                                        save_weights_only=True,
                                                        mode='auto',
                                                        save_freq=10)

        validator = ValidationCallback(test_data, 10)

        hist = model.fit(train_data, steps_per_epoch=len(train_data),
                         validation_data=test_data,
                         epochs=30,  # callbacks=[checkpoint],
                         validation_steps=len(test_data))
        hist.history["distillation_eval"] = validator.predictions
        with open("history.json", "w") as f:
            json.dump(hist.history, f)

    if __name__ == '__main__':
        main()
