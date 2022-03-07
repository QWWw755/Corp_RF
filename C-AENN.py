from keras.models import Sequential
from keras.layers import Conv1D,MaxPooling1D,Flatten,Dense,BatchNormalization
from keras.layers import Input, Dense, Flatten
import pickle


def C_AENN(train_x, train_y, test_x, test_y):

    model = Sequential()
    model.add(Conv1D(input_shape=(12, 2), filters=32, kernel_size=7, padding='same', activation='relu'))
    model.add(MaxPooling1D(pool_size=2, padding='same'))  # MaxPooling
    model.add(BatchNormalization())

    model.add(Conv1D(filters=64, kernel_size=5, padding='same', activation='relu'))
    model.add(MaxPooling1D(pool_size=2, padding='same'))  # MaxPooling
    model.add(BatchNormalization())

    model.add(Conv1D(filters=128, kernel_size=3, padding='same', activation='relu'))
    model.add(MaxPooling1D(pool_size=2, padding='same'))  # MaxPooling
    model.add(BatchNormalization())
    model.add(Flatten())  # Flatten256

    model.add(Dense(128, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(16, activation='relu'))

    model.add(Dense(32, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(4, activation='softmax'))
    model.summary()
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(train_x, train_y, batch_size=128, nb_epoch=200, verbose=2)
    tes_label = model.predict_classes(test_data) + 1  # 测试集的预测标签
    print("test data：", model.evaluate(test_x, test_y))

    file = open(SavePath, "wb")
    pickle.dump(model, file)
    file.close()
C_AENN(train_data, train_label, test_data, test_label)
