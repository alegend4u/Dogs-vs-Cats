import random
import pathlib
import tensorflow as tf
import numpy as np
import csv
import matplotlib.pyplot as plt
from model import MyModel

print("GPU: ", len(tf.config.list_physical_devices('GPU')) > 0)

BATCH_SIZE = 64
IMAGE_HEIGHT = 112
IMAGE_WIDTH = 112
CLASSES = np.array(['cat', 'dog'])


def get_label(file_path):
    parts = tf.strings.split(file_path, '/')
    label = tf.where(CLASSES == parts[-2])[0][0]
    return label


def decode_img(img):
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)
    return img


def preprocess(img):
    # Augment
    img = tf.image.random_flip_left_right(img)
    img = tf.image.random_brightness(img, 0.5)
    img = tf.image.random_contrast(img, 1, 2)
    img = random.choice([img, tf.image.central_crop(img, 0.6)])
    img = tf.image.random_saturation(img, 1, 3)
    img = tf.image.random_flip_up_down(img)

    img = tf.image.resize(img, [IMAGE_WIDTH, IMAGE_HEIGHT])
    # Normalize
    img = img / 255.

    return img


def process_path(file_path):
    label = get_label(file_path)
    img = tf.io.read_file(file_path)
    img = decode_img(img)
    img = preprocess(img)
    return img, label


def test_process_path(file_path):
    label = get_label(file_path)
    img = tf.io.read_file(file_path)
    img = decode_img(img)
    img = tf.expand_dims(img, 0)
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


def train():
    train_path = pathlib.Path('data/train')

    no_of_samples = len(list(train_path.glob('**/*.jpg')))

    steps_per_epoch = np.ceil(no_of_samples / BATCH_SIZE)
    list_ds = tf.data.Dataset.list_files(str(train_path / '*/*'))
    labeled_ds = list_ds.map(
        process_path,
        num_parallel_calls=tf.data.experimental.AUTOTUNE
    )

    dataset = prepare_for_training(labeled_ds)

    # image_batch, label_batch = next(iter(dataset))
    # show_batch(image_batch.numpy(), label_batch.numpy())

    # Split the dataset
    train_size = np.ceil(no_of_samples / BATCH_SIZE * 0.7)
    val_size = np.ceil(no_of_samples / BATCH_SIZE * 0.3)

    train_ds = dataset.take(train_size)
    val_ds = dataset.skip(train_size).take(val_size)
    loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=True)

    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_accuracy = tf.keras.metrics.BinaryAccuracy(name='train_accuracy')

    val_loss = tf.keras.metrics.Mean(name='test_loss')
    val_accuracy = tf.keras.metrics.BinaryAccuracy(name='test_accuracy')

    @tf.function
    def train_step(b_images, b_labels):
        with tf.GradientTape() as tape:
            predictions = model(b_images, training=True)
            loss = loss_fn(b_labels, predictions)
            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        train_loss(loss)
        train_accuracy(b_labels, predictions)

    @tf.function
    def val_step(b_images, b_labels):
        predictions = model(b_images, training=True)
        loss = loss_fn(b_labels, predictions)

        val_loss(loss)
        val_accuracy(b_labels, predictions)

    # model.build()x
    # model.summary()
    epochs = 50
    print("Steps per epochs:", steps_per_epoch)
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


def kaggle_test(test_path):
    output_file = 'output.csv'
    with open(output_file, 'w') as f:
        writer = csv.writer(f, delimiter=',')
        writer.writerow(['id', 'label'])
    test_path = pathlib.Path(test_path)

    for img_id in range(1, 12501):
        file = "{}/{}.jpg".format(test_path, img_id)
        image = tf.io.read_file(file)
        image = decode_img(image)
        input_image = tf.expand_dims(image, 0)
        prediction = model(input_image, training=False)

        prediction = int(prediction > 0)
        result = [img_id, prediction]
        # print(*result, sep=',')
        with open(output_file, 'a+') as f:
            writer = csv.writer(f, delimiter=',')
            writer.writerow(result)


def test():
    test_path = pathlib.Path('data/test')
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
    model = MyModel()

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
    checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=model)
    manager = tf.train.CheckpointManager(checkpoint, directory="checkpoints", max_to_keep=5)
    checkpoint.restore(manager.latest_checkpoint)
    if manager.latest_checkpoint:
        print("Restored from {}".format(manager.latest_checkpoint))
    else:
        print("Initializing from scratch.")

    train()
    # test()
    # kaggle_test('data/test1')
