import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental import preprocessing
from sklearn.metrics import roc_auc_score, accuracy_score
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
s = tf.InteractiveSession()

#download data
X_train = pd.read_csv("training_set_features.csv").drop(['respondent_id'], axis=1)
y_train = pd.read_csv("training_set_labels.csv").drop(['respondent_id'], axis=1)

for i in range(15):  # ! A optimiser
    X_train.iloc[:, [i]] = X_train.iloc[:, [i]].fillna(X_train.iloc[:, [i]].median())

for i in range(21,31):  # ! A optimiser
    X_train.iloc[:, [i]] = X_train.iloc[:, [i]].fillna(X_train.iloc[:, [i]].median())

for i in range(33,len(X_train.columns)):  # ! A optimiser
    X_train.iloc[:, [i]] = X_train.iloc[:, [i]].fillna(X_train.iloc[:, [i]].median())


#pre-processing
X_train = pd.get_dummies(X_train)
X_train = X_train.fillna(X_train.mean())

#Validation set
X_test = np.array(X_train.iloc[25000:])
y_test = np.array(y_train.iloc[25000:])

#Train set
X_train = np.array(X_train.iloc[:25000])
y_train = np.array(y_train.iloc[:25000])

#X_unlabeled = pd.read_csv("test_set_features.csv")

## Defining various initialization parameters
num_classes = y_train.shape[1]
num_features = X_train.shape[1]  # 105
num_output = y_train.shape[1]
num_layers_0 = int(num_features * 2/3)
num_layers_1 = int(num_layers_0 * 2/3)
num_layers_2 = int(num_layers_1 / 2)
print(num_layers_0, num_layers_1, num_layers_2)
starter_learning_rate = 0.001
regularizer_rate = 0.05

# Placeholders for the input data
input_X = tf.placeholder('float32', shape=(None, num_features), name="input_X")
input_y = tf.placeholder('float32', shape=(None, num_classes), name='input_Y')
## for dropout layer
keep_prob = tf.placeholder(tf.float32)

## Weights initialized by random normal function with std_dev = 1/sqrt(number of input features)
weights_0 = tf.Variable(tf.random_normal([num_features, num_layers_0], stddev=(1 / tf.sqrt(float(num_features)))))
bias_0 = tf.Variable(tf.random_normal([num_layers_0]))
weights_1 = tf.Variable(tf.random_normal([num_layers_0, num_layers_1], stddev=(1 / tf.sqrt(float(num_layers_0)))))
bias_1 = tf.Variable(tf.random_normal([num_layers_1]))
weights_2 = tf.Variable(tf.random_normal([num_layers_1, num_layers_2], stddev=(1 / tf.sqrt(float(num_layers_1)))))
bias_2 = tf.Variable(tf.random_normal([num_layers_2]))
weights_3 = tf.Variable(tf.random_normal([num_layers_2, num_output], stddev=(1 / tf.sqrt(float(num_layers_2)))))
bias_3 = tf.Variable(tf.random_normal([num_output]))

## Initializing weigths and biases
hidden_output_0 = tf.nn.relu(tf.matmul(input_X, weights_0) + bias_0)
hidden_output_0_0 = tf.nn.dropout(hidden_output_0, keep_prob)
hidden_output_1 = tf.nn.relu(tf.matmul(hidden_output_0_0, weights_1) + bias_1)
hidden_output_1_1 = tf.nn.dropout(hidden_output_1, keep_prob)
hidden_output_2 = tf.nn.relu(tf.matmul(hidden_output_1_1, weights_2) + bias_2)
hidden_output_2_2 = tf.nn.dropout(hidden_output_2, keep_prob)
predicted_y = tf.sigmoid(tf.matmul(hidden_output_2_2, weights_3) + bias_3)

## Defining the loss function
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=predicted_y, labels=input_y)) \
       + regularizer_rate * (tf.reduce_sum(tf.square(bias_0)) + tf.reduce_sum(tf.square(bias_1)))

## Variable learning rate
learning_rate = tf.train.exponential_decay(starter_learning_rate, 0, 5, 0.85, staircase=True)
## Adam optimzer for finding the right weight
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss, var_list=[weights_0, weights_1, weights_2,
                                                                           bias_0, bias_1, bias_2])

## Metrics definition
correct_prediction = tf.equal(tf.argmax(y_train, 1), tf.argmax(predicted_y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

## Training parameters
batch_size = 128
epochs = 1000
dropout_prob = 0.4
training_accuracy = []
training_loss = []
testing_accuracy = []
s.run(tf.global_variables_initializer())
for epoch in range(epochs):
    arr = np.arange(X_train.shape[0])
    np.random.shuffle(arr)
    for index in range(0, X_train.shape[0], batch_size):
        s.run(optimizer, {input_X: X_train[arr[index:index + batch_size]],
                          input_y: y_train[arr[index:index + batch_size]],
                          keep_prob: dropout_prob})
    training_accuracy.append(s.run(accuracy, feed_dict={input_X: X_train,
                                                        input_y: y_train, keep_prob: 1}))
    training_loss.append(s.run(loss, {input_X: X_train,
                                      input_y: y_train, keep_prob: 1}))

    ## Evaluation of model
    testing_accuracy.append(roc_auc_score(y_test.argmax(1),
                                           s.run(predicted_y, {input_X: X_test, keep_prob: 1}).argmax(1)))
    print("Epoch:{0}, Train loss: {1:.2f} Train acc: {2:.3f}, Test ROC-acc:{3:.3f}".format(epoch,
                                                                                       training_loss[epoch],
                                                                                       training_accuracy[epoch],
                                                                                       testing_accuracy[epoch]))

