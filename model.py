from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPool2D, BatchNormalization, Activation, Dropout


class MyModel(Model):
    def __init__(self):
        super(MyModel, self).__init__()
        self.conv1 = Conv2D(32, 3, padding='same', activation='relu')
        self.bn1 = BatchNormalization()
        self.pool1 = MaxPool2D(pool_size=(2, 2))
        self.drop1 = Dropout(0.25)

        self.conv2 = Conv2D(64, 3, padding='same', activation='relu')
        self.bn2 = BatchNormalization()
        self.pool2 = MaxPool2D(pool_size=(2, 2))
        self.drop2 = Dropout(0.25)

        self.conv3 = Conv2D(128, 3, padding='same', activation='relu')
        self.bn3 = BatchNormalization()
        self.pool3 = MaxPool2D(pool_size=(2, 2))
        self.drop3 = Dropout(0.25)

        self.flatten = Flatten()
        self.d1 = Dense(512, activation='relu')
        self.bn4 = BatchNormalization()
        self.drop4 = Dropout(0.5)

        self.d2 = Dense(1, activation='sigmoid')

    def call(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.pool1(x)
        x = self.drop1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.pool2(x)
        x = self.drop2(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.pool3(x)
        x = self.drop3(x)

        x = self.flatten(x)
        x = self.d1(x)
        x = self.bn4(x)
        x = self.drop4(x)

        x = self.d2(x)

        return x
