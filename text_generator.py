import tensorflow as tf
import numpy as np
import re


class Network:

    def __init__(self):
        # Variable initialization
        self.text = None
        self.dictionary = {}
        self.unique_characters = 0
        self.probabilities = []

    def input_text(self, file_name):
        # open file and copy content and close file
        book = open(file_name)
        content = book.read()
        book.close()

        # Content to lowercase
        self.text = content.lower()

        # remove numbers
        self.text = re.sub(r'\d+', '', self.text)

        # Count unique characters frequency
        for c in self.text:
            if c in self.dictionary:
                self.dictionary[c] += 1
            else:
                self.dictionary[c] = 0

        # Set number of unique characters
        self.unique_characters = len(self.dictionary)

        # Compute characters probability distribution
        for i, v in enumerate(self.dictionary.values()):
            self.probabilities.append(v / len(self.text))

        # Normalize distribution (otherwise it do not sum up exactly to 1 -> problem for np.random.choice)
        self.probabilities = np.array(self.probabilities)
        self.probabilities /= self.probabilities.sum()

    def generate_batches(self, batch_size, sequence_length):
        """
        Generate batches
        """
        block_length = len(self.text) // batch_size
        batches = []
        for i in range(0, block_length, sequence_length):
            batch = []
            targ = []
            for j in range(batch_size):
                start = j * block_length + i
                end = min(start + sequence_length, j * block_length + block_length)

                # Decided to have two output which represent input and targets

                # input and target have only 255 elements because the input goes from 0 to 255
                # (last character do not have a prediction so it is useless) wheres target goes from 1 to 256 so that
                # it represents the character to predict

                input = np.zeros(sequence_length - 1)
                input[0:end - start - 1] = [(list(self.dictionary.keys()).index(c)) for c in
                                            self.text[start:end - 1]]
                batch.append(input)
                target = np.zeros(sequence_length - 1)
                target[0:end - start - 1] = [(list(self.dictionary.keys()).index(c)) for c in
                                             self.text[start + 1:end]]
                targ.append(target)

            batches.append((np.array(batch, dtype=int), np.array(targ, dtype=int)))
        return batches


# Hyper parameters
batch_size = 16
sequence_length = 256
hidden_units = 256
# smaller learning rate otherwise the loss explodes
learning_rate = 0.001
network = Network()
network.input_text('book.txt')

# get batches
input = network.generate_batches(batch_size, sequence_length)

# Number of step to pass all the data once
number_of_steps = len(input)

print("\n\n_________MODEL CREATION_________")

session = tf.Session()

# inputs (-1 because the last element does not have a prediction)
X_int = tf.placeholder(shape=[batch_size, sequence_length - 1], dtype=tf.int64)
Y_int = tf.placeholder(shape=[batch_size, sequence_length - 1], dtype=tf.int64)

# one hot encoded inputs
X = tf.one_hot(X_int, depth=network.unique_characters)
Y = tf.one_hot(Y_int, depth=network.unique_characters)

# output layer values
Wout = tf.Variable(tf.truncated_normal(shape=(hidden_units, network.unique_characters)))
bout = tf.Variable(tf.zeros(shape=[network.unique_characters]))

# output
Y_flat = tf.reshape(Y, [-1, network.unique_characters])

# mask lengths (not all elements have 255 values ie the last one has less)
lengths = tf.placeholder(shape=[batch_size], dtype=tf.int64)

# LSTM cells
cells = [tf.contrib.rnn.LSTMCell(num_units=hidden_units), tf.contrib.rnn.LSTMCell(num_units=hidden_units)]
# stacked LSTM cells
cell = tf.contrib.rnn.MultiRNNCell(cells)

# initial state
init_state = cell.zero_state(batch_size, dtype=tf.float32)

# output and new state of the LSTM
output, state = tf.nn.dynamic_rnn(cell, X, initial_state=init_state, sequence_length=lengths)

# reshape output 4080x256
o = tf.reshape(output, [-1, hidden_units])

# compute output values 4080x78
Z = tf.matmul(o, Wout) + bout

print('shape of o:{}'.format(o.shape))
print('shape of Z:{}'.format(Z.shape))
print('shape of Y:{}'.format(Y.shape))
print('shape of X:{}'.format(X.shape))
print('shape of X_int:{}'.format(X_int.shape))
print('shape of Y_int:{}'.format(Y_int.shape))
print('shape of Y_flat:{}'.format(Y_flat.shape))

# mask
mask = tf.sequence_mask(lengths, dtype=tf.float32)
mask = tf.reshape(mask, [-1])

# loss with mask
loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=Y_flat, logits=Z)
loss = tf.reduce_sum(loss * mask) / tf.reduce_sum(mask)

# optimizer and train variable
optimizer = tf.train.AdamOptimizer(learning_rate)
train = optimizer.minimize(loss)

# get softmax predictions of the output (probabilities)
pred = tf.nn.softmax(Z)

# run variable initializer and get the initial LSTM cell state
session.run(tf.global_variables_initializer())

# initial state
n_state = session.run(init_state)

####### It is better to create the graph with all the elements of an entire epoch and then execute it 5 times?
####### Here I create a simple LSTM and then I run it for each element batch. I give the new cell state as input
####### for the new LSTM so that it has a memory.


print("\n\n_________START TO TRAIN_________")
l = None
# Train the network for 5 epochs
for e in range(5):
    for i in range(number_of_steps):
        # get n_state of the network and reuse it in the next step
        l, t, n_state = session.run([loss, train, state],
                                    {X_int: input[i][0], Y_int: input[i][1], init_state: n_state,
                                     lengths: [len(input[i][0][0]) for j in range(batch_size)]})
        # Generating the lengths by looking at an element size and copying it for batch_size times
    # print loss after every epoch
    print(l)

print("_________START TO WRITE_________")

# create lengths array with so that the LSTM considers only the first element
len_pred = np.zeros(batch_size)
len_pred[0] = 1

# create input array as the training so that the shape are the same. Full it with zeros except the first element
letter = np.zeros((batch_size, sequence_length - 1))

# get keys elements of the dictionary
keys = list(network.dictionary.keys())

# open output file
out = open("output.txt", "w+")

# create 20 sequences
for j in range(20):
    # Create new state for each sentence (it starts fresh)
    # n_state = session.run(init_state)  # should I use the final state of the training ??? Yes better text
    # start with random character from initial distribution
    # Set it to the first element of letter
    letter[0][0] = np.random.choice([i for i in range(network.unique_characters)], 1, p=network.probabilities)[0]
    # write on the out put file this first character
    out.write(keys[int(letter[0][0])])
    # write 255 characters
    for i in range(255):
        # predict new character probability distribution
        # Reuse old state so that it know what it has written before
        p, n_state = session.run([pred, state], {X_int: letter, init_state: n_state, lengths: len_pred})
        # take random choice over the probability distribution and I give it the the next run of the network
        letter[0][0] = np.random.choice([k for k in range(network.unique_characters)], 1, p=p[0])[0]
        # it is better to take the argmax -> bad choice see report
        # letter[0][0] = np.argmax(p[0])
        # write character
        out.write(keys[int(letter[0][0])])

out.close()

print("_________END_________")
