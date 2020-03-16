import pathlib
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from model import MyModel

test_path = pathlib.Path('data/test')
CLASSES = np.array([dire.name for dire in test_path.iterdir()])
BATCH_SIZE = 64
IMAGE_HEIGHT = 112
IMAGE_WIDTH = 112


def get_label(file_path):
    parts = tf.strings.split(file_path, '/')
    label = tf.where(CLASSES == parts[-2])[0][0]
    return label


def decode_img(img):
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)
    return tf.image.resize(img, [IMAGE_WIDTH, IMAGE_HEIGHT])


def process_path(file_path):
    label = get_label(file_path)
    img = tf.io.read_file(file_path)
    img = decode_img(img)
    img = tf.expand_dims(img, 0)
    return img, label


test_list_ds = tf.data.Dataset.list_files(str(test_path / '*/*'))
test_labeled_ds = test_list_ds.map(
    process_path,
    num_parallel_calls=tf.data.experimental.AUTOTUNE
)

model = MyModel()

optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=model)
manager = tf.train.CheckpointManager(
    checkpoint, directory="checkpoints", max_to_keep=5
)
checkpoint.restore(manager.latest_checkpoint)
if manager.latest_checkpoint:
    print("Restored from {}".format(manager.latest_checkpoint))
else:
    print("Initializing from scratch.")

test_accuracy = tf.keras.metrics.BinaryAccuracy()
for image, label in test_labeled_ds:
    prediction = model(image, training=False)
    test_accuracy(label, prediction)

print("Accuracy: {}".format(test_accuracy.result()*100))
