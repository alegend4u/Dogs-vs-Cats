from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPool2D, BatchNormalization, Activation


class MyModel(Model):
    def __init__(self):
        super(MyModel, self).__init__()
        self.conv1 = Conv2D(32, 5, padding='same')
        self.pool1 = MaxPool2D(pool_size=(2, 2))
        self.bn1 = BatchNormalization()

        self.conv2 = Conv2D(64, 3, padding='same')
        self.pool2 = MaxPool2D(pool_size=(2, 2))
        self.bn2 = BatchNormalization()

        self.conv3 = Conv2D(64, 3, padding='same')
        self.pool3 = MaxPool2D(pool_size=(2, 2))
        self.bn3 = BatchNormalization()

        self.conv4 = Conv2D(128, 3, padding='same')
        self.pool4 = MaxPool2D(pool_size=(2, 2))
        self.bn4 = BatchNormalization()

        self.flatten = Flatten()
        self.d1 = Dense(128)
        self.bn5 = BatchNormalization()

        self.d2 = Dense(1)

        self.activattion = Activation('relu')

    def call(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.activattion(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.activattion(x)
        x = self.pool2(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.activattion(x)
        x = self.pool3(x)

        x = self.conv4(x)
        x = self.bn4(x)
        x = self.activattion(x)
        x = self.pool4(x)

        x = self.flatten(x)
        x = self.d1(x)
        x = self.bn5(x)
        x = self.activattion(x)

        x = self.d2(x)

        return x
