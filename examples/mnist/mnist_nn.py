import sys
#print(sys.path)
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import numpy as np
import tensorflow as tf
mnist = tf.keras.datasets.mnist

# from keras.datasets import mnist
# from keras import models
# from keras import layers
# from keras.utils import to_categorical
# import numpy as np
from random import random

# from __future__ import print_function
import neat
# import visualize

# ---------------------
import torch
import torchvision
import torchvision.transforms as transforms
from torch.autograd import Variable
import evaluate_torch
# ---------------------


transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=1,
                                          shuffle=True, num_workers=0)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=1,
                                         shuffle=False, num_workers=0)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

def eval_genomes(genomes, config):

    # start = int(random() * (train_images_sum - batch_size))
    #start = 0
    for genome_id, genome in genomes:
        #hitCount = 0
        # visualize.draw_net(config, genome, True, node_names=node_names)
        #for i in range(start, start + batch_size):
        #

        batch_size = 500
        hit_count = 0
        start = int(random() * (len(trainloader) - batch_size))
        #batch = trainloader(start::batch_size)
        i = 0
        for num, data in enumerate(trainloader, start):
            i += 1
            # 得到输入数据
            inputs, labels = data

            # 包装数据
            inputs, labels = Variable(inputs), Variable(labels)
            try:

                net = evaluate_torch.Net(config, genome)
                outputs = net.forward(inputs)
                #print(net)

                predicted = torch.max(outputs.data, 0)[1]
                if (predicted == labels):
                    hit_count += 1

            except Exception as e:
                print(e)
                genome.fitness = 0
            if (i == batch_size - 1):
                break

        genome.fitness = hit_count / batch_size
        print(genome.fitness)

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

# visualize.draw_net(config, winner, True, node_names=node_names)