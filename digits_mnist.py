import tensorflow as tf

#The fubction on_epoch_end will be called after each epoch
#When accuracy is reached 99%, the training will be stopped
class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self,epoch,logs = {}):
        if(logs.get('accuracy')>=0.99):
            print("\nAccuracy reached 99%")
            self.model.stop_training = True

callbacks = myCallback()
mnist = tf.keras.datasets.mnist

(x_train,y_train), (x_test,y_test) = mnist.load_data()

x_train, x_test = x_train/255.0, x_test/255.0

model = tf.keras.models.Sequential([tf.keras.layers.Flatten(input_shape = (28,28)),
                                    tf.keras.layers.Dense(128,activation = tf.nn.relu),
                                    tf.keras.layers.Dense(10,activation = tf.nn.softmax)])

model.compile(optimizer = tf.optimizers.Adam(),
              loss = 'sparse_categorical_crossentropy',
              metrics = ['accuracy'])

model.fit(x_train,y_train,epochs = 10,callbacks = [callbacks])

test_loss, test_acc = model.evaluate(x_test, y_test)

output = model.predict(x_test)