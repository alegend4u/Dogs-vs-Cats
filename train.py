import pathlib
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from model import MyModel

print("GPU: ", len(tf.config.list_physical_devices('GPU')) > 0)

train_path = pathlib.Path('data/train')
test_path = pathlib.Path('data/test')

NO_OF_SAMPLES = len(list(train_path.glob('**/*.jpg')))
CLASSES = np.array([dire.name for dire in train_path.iterdir()])

BATCH_SIZE = 64
IMAGE_HEIGHT = 112
IMAGE_WIDTH = 112
STEPS_PER_EPOCH = np.ceil(NO_OF_SAMPLES / BATCH_SIZE)

list_ds = tf.data.Dataset.list_files(str(train_path / '*/*'))
for f in list_ds.take(1):
    print(f)


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
    return img, label


def prepare_for_training(ds, cache=True, shuffle_buffer_size=1000, batch_size=BATCH_SIZE):
    if cache:
        if isinstance(cache, str):
            ds = ds.cache(cache)
        else:
            ds = ds.cache()
    ds = ds.shuffle(buffer_size=shuffle_buffer_size)
    ds = ds.repeat()  # repeat forever
    ds = ds.batch(batch_size)
    ds = ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    return ds


def show_batch(images, labels):
    for i in range(10):
        plt.subplot(2, 5, i + 1)
        plt.imshow(images[i])
        plt.title(CLASSES[labels[i]])
        plt.axis('off')
    plt.show()


labeled_ds = list_ds.map(
    process_path,
    num_parallel_calls=tf.data.experimental.AUTOTUNE
)

dataset = prepare_for_training(labeled_ds)

image_batch, label_batch = next(iter(dataset))
# show_batch(image_batch.numpy(), label_batch.numpy())

# Split the dataset
train_size = np.ceil(NO_OF_SAMPLES / BATCH_SIZE * 0.7)
val_size = np.ceil(NO_OF_SAMPLES / BATCH_SIZE * 0.3)

train_ds = dataset.take(train_size)
val_ds = dataset.skip(train_size).take(val_size)
test_ds = dataset.skip(train_size + val_size)

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


def train():
    loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=True)

    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_accuracy = tf.keras.metrics.BinaryAccuracy(name='train_accuracy')

    val_loss = tf.keras.metrics.Mean(name='test_loss')
    val_accuracy = tf.keras.metrics.BinaryAccuracy(name='test_accuracy')

    @tf.function
    def train_step(images, labels):
        with tf.GradientTape() as tape:
            predictions = model(images, training=True)
            loss = loss_fn(labels, predictions)
            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        train_loss(loss)
        train_accuracy(labels, predictions)

    @tf.function
    def val_step(images, labels):
        predictions = model(images, training=True)
        loss = loss_fn(labels, predictions)

        val_loss(loss)
        val_accuracy(labels, predictions)

    # model.build()
    # model.summary()
    epochs = 20
    print("Steps per epochs:", STEPS_PER_EPOCH)
    print("Training...")
    prev_loss = np.inf

    for epoch in range(epochs):
        train_loss.reset_states()
        train_accuracy.reset_states()
        val_loss.reset_states()
        val_accuracy.reset_states()
        step = 0
        for images, labels in train_ds:
            train_step(images, labels)
            step += 1
            # print("Step: {}/{}".format(step, STEPS_PER_EPOCH))
        vstep = 0
        for test_images, test_labels in val_ds:
            val_step(test_images, test_labels)
            vstep += 1
            # print("VStep: ", vstep)

        template = 'Epoch: {}, Loss: {}, Accuracy; {}, Val Loss: {}, Val Accuracy: {}'
        print(template.format(epoch + 1,
                              train_loss.result(),
                              train_accuracy.result() * 100,
                              val_loss.result(),
                              val_accuracy.result() * 100))
        if val_loss.result() < prev_loss:
            print("Saving weights! Loss decreased: {} ==> {}".format(prev_loss, val_loss.result()))
            prev_loss = val_loss.result()
            manager.save()


def test():
    test_list_ds = tf.data.Dataset.list_files(str(test_path / '*/*'))
    test_labeled_ds = test_list_ds.map(
        process_path,
        num_parallel_calls=tf.data.experimental.AUTOTUNE
    )
    test_labeled_ds = test_labeled_ds.batch(1)
    test_accuracy = tf.keras.metrics.BinaryAccuracy(name='test_accuracy')
    for images, labels in test_labeled_ds:
        predictions = model(images, training=False)
        test_accuracy(labels, predictions)
    print("Test Accuracy: {}".format(test_accuracy.result()))


if __name__ == "__main__":
    train()
    # test()
