import keras
keras.__version__

from keras.datasets import mnist
from keras import models
from keras import layers
from keras.utils import to_categorical
import numpy as np
from random import random

# from __future__ import print_function
import neat
import visualize

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

train_images_sum = 60000
train_images = train_images.reshape((train_images_sum, 28 * 28))
train_images = train_images.astype('float32') / 255

test_images_sum = 10000
test_images = test_images.reshape((test_images_sum, 28 * 28))
test_images = test_images.astype('float32') / 255

eval_len = 10
eval_image = train_images[:eval_len]
eval_labels = train_labels[:eval_len]

# for layer design
train_images = train_images[:,:5]
test_images = test_images[:, :5]

# opt for MNIST
def static():
    list1 = [[0 for col in range(2)] for row in range(784)]

    for i in range(0, 784):
        list1[i][0] = i

    count = dict(list1)

    for i in range(0, 60000):
        print(i)
        for j in range(0, 784):
            if (train_images[i, j] > 0):
                count[j] += 1;
#    count = sorted(count.items(),key = lambda count:count[1],reverse = True)

    res = open("mnist_count.txt", "w")
    for j in count:
        res.write("{0}, {1}\n".format(j[0],j[1]))
    res.close()

#static()

"""
network = models.Sequential()
network.add(layers.Dense(512, activation='relu', input_shape=(28 * 28,)))
network.add(layers.Dense(10, activation='softmax'))

network.compile(optimizer='rmsprop',
                loss='categorical_crossentropy',
                metrics=['accuracy'])

train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

network.fit(train_images, train_labels, epochs=5, batch_size=128)

test_loss, test_acc = network.evaluate(test_images, test_labels)

print('test_acc:', test_acc)

"""

def hit(label, input, net):
    output = net.activate(input)
    result = output.index(max(output))
    if (result == label):
        return True
    else:
        return False

def eval_genomes(genomes, config):
    batch_size = 20
    # start = int(random() * (train_images_sum - batch_size))
    start = 0
    for genome_id, genome in genomes:
        hitCount = 0

        for i in range(start, start + batch_size):
            mnist_inputs = train_images[i]
            net = neat.nn.FeedForwardNetwork.create(genome, config)
            if (hit(train_labels[i], mnist_inputs, net)):
                hitCount += 1
        genome.fitness = hitCount / batch_size

# Load configuration.
config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                     neat.DefaultSpeciesSet, neat.DefaultStagnation,
                     'config-mnist')

# reset result file
res = open("result.txt", "w")
res.close()

# Create the population, which is the top-level object for a NEAT run.
p = neat.Population(config)

# Add a stdout reporter to show progress in the terminal.
p.add_reporter(neat.StdOutReporter(False))

# Run until a solution is found.
winner = p.run(eval_genomes)

# Run for up to 300 generations.
# pe = neat.ThreadedEvaluator(4, eval_genomes)
# winner = p.run(pe.evaluate)
# pe.stop()

# Display the winning genome.
print('\nBest genome:\n{!s}'.format(winner))

"""
# print results on evaluate set
winner_net = neat.nn.FeedForwardNetwork.create(winner, config)
for i in range(0, 10):
    output = winner_net.activate(eval_image[i])
    fitness = 10;
    mnist_outputs = [0.0] * 10
    mnist_outputs[eval_labels[i]] = 1.0
    for j in range(0, 10):
        fitness -= (output[j] - mnist_outputs[j]) ** 2
    print(eval_labels[i], fitness)
    print("got {!r}".format(output))
"""

# test on test dataset
winner_net = neat.nn.FeedForwardNetwork.create(winner, config)
hitCount = 0
for i in range(0, len(test_labels) // 10): #len(test_labels)
    if (hit(test_labels[i], test_images[i], winner_net)):
        hitCount += 1
print("hit %d of %d"%(hitCount, len(test_labels) / 10))

"""
# Show output of the most fit genome against training data.
print('\nOutput:')
winner_net = neat.nn.FeedForwardNetwork.create(winner, config)
for xi, xo in zip(xor_inputs, xor_outputs):
    output = winner_net.activate(xi)
    print("  input {!r}, expected output {!r}, got {!r}".format(xi, xo, output))
"""

node_names = {# -28: '-28', -27: '-27', -26: '-26', -25: '-25', -24: '-24', -23: '-23', -22: '-22', -21: '-21',
              # -20: '-20', -19: '-19', -18: '-18', -17: '-17', -16: '-16', -15: '-15', -14: '-14', -13: '-13',
              # -12: '-12', -11: '-11', -10: '-10', -9: '-09', -8: '-08', -7: '-07', -6: '-06',
              -5: '-05', -4: '-04', -3: '-03', -2: '-02', -1: '-01', 0: '00', 1: '01', 2: '02',
              3: '03', 4: '04', 5: '05', 6: '06', 7: '07', 8: '08', 9: '09'}

visualize.draw_net(config, winner, True, node_names=node_names)
