import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# import numpy as np
from random import random

# from __future__ import print_function
import neat

import torch
import torchvision
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import numpy as np

import evaluate_torch

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

torch_batch_size = 4

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=torch_batch_size,
                                          shuffle=True, num_workers=0)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=torch_batch_size,
                                         shuffle=False, num_workers=0)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

def eval_genomes(genomes, config):

    j = 0
    for genome_id, genome in genomes:
        j += 1

        #setup the network
        net = evaluate_torch.Net(config, genome)

        criterion = nn.CrossEntropyLoss()  # use a Classification Cross-Entropy loss
        optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

        delta = 0.1
        losses_len = 100
        losses = np.array([0.0] * losses_len)

        #train the network
        for epoch in range(1):  # loop over the dataset multiple times

            running_loss = 0.0
            for i, data in enumerate(trainloader, 0):
                # get the inputs
                inputs, labels = data

                # wrap them in Variable
                inputs, labels = Variable(inputs), Variable(labels)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                outputs = net(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                # record the losses
                running_loss += loss.data.item()
                """
                losses[i % losses_len] = loss.data.item()

                # if loss do not change apperantly, then break
                if (i % losses_len == losses_len - 1):
                    a = np.std(losses)
                    print(a)
                    if (a < delta):
                        break;
                """
                # print statistics
                if i % 2000 == 1999:  # print every 2000 mini-batches
                    print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 2000))
                    running_loss = 0.0
        print('Finished Training')

        #evaluate the fitness
        evaluate_batch_size = 100
        hit_count = 0
        start = int(random() * (len(trainloader) - evaluate_batch_size * torch_batch_size))

        i = 0
        for num, data in enumerate(trainloader, start):
            i += 1
            # 得到输入数据
            inputs, labels = data

            # 包装数据
            inputs, labels = Variable(inputs), Variable(labels)
            try:

                outputs = net.forward(inputs)

                _, predicted = torch.max(outputs.data, 1)
                hit_count += (predicted == labels).sum()

            except Exception as e:
                print(e)
                genome.fitness = 0
            if (i == evaluate_batch_size - 1):
                break
                
        net.write_back_parameters(genome)
        genome.fitness = hit_count.item() / (evaluate_batch_size * torch_batch_size)
        print('{0}: {1:3.3f}'.format(j,genome.fitness))

# Load configuration.
config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                     neat.DefaultSpeciesSet, neat.DefaultStagnation,
                     'config-mnist')

# reset result file
res = open("result.csv", "w")
best = open("best.txt", "w")
res.close()
best.close()

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

correct = 0
total = 0
net = evaluate_torch.Net(config, winner)

for data in testloader:
    images, labels = data
    outputs = net.forward(images)
    predicted = torch.max(outputs.data, 0)[1]
    if (predicted == labels):
        correct += 1

print("hit %d of %d"%(correct, len(testset)))

#TODO: wirte model to pytorch files

node_names = {# -28: '-28', -27: '-27', -26: '-26', -25: '-25', -24: '-24', -23: '-23', -22: '-22', -21: '-21',
              # -20: '-20', -19: '-19', -18: '-18', -17: '-17', -16: '-16', -15: '-15', -14: '-14', -13: '-13',
              # -12: '-12', -11: '-11', -10: '-10', -9: '-09', -8: '-08', -7: '-07', -6: '-06',
              -5: '-05', -4: '-04', -3: '-03', -2: '-02', -1: '-01', 0: '00', 1: '01', 2: '02',
              3: '03', 4: '04', 5: '05', 6: '06', 7: '07', 8: '08', 9: '09'}

# visualize.draw_net(config, winner, True, node_names=node_names)