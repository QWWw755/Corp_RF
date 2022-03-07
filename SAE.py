from keras.models import Sequential
from keras.layers import  Dense
import pickle

def SAE(train_x, train_y, test_x, test_y):
    model = Sequential()
    model.add(Dense(256, activation='relu',input_dim=24))
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
    tes_label = model.predict_classes(test_data) + 1
    print("test data：", model.evaluate(test_x, test_y))
    file = open(SavePath, "wb")
    pickle.dump(model, file)
    file.close()
SAE(train_data, train_label, test_data, test_label)
