import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

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

torch_batch_size = 40

gpu = False

#net_dict = {}

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


# evaluate the fitness
# batch_size = 0 means evaluate until reach the end of the evaluate set
def eval_fitness(net, loader, batch_size, torch_batch_size, start, gpu):

    hit_count = 0
    total = 0

    i = 0
    for num, data in enumerate(loader, start):
        i += 1
        total += torch_batch_size

        # 得到输入数据
        inputs, labels = data

        # 包装数据
        if gpu:
            inputs, labels = Variable(inputs).cuda(), Variable(labels).cuda()
        else:
            inputs, labels = Variable(inputs), Variable(labels)
        try:

            outputs = net.forward(inputs)
            _, predicted = torch.max(outputs.data, 1)
            hit_count += (predicted == labels).sum()

        except Exception as e:
            print(e)

        if (i == batch_size): #because i > 0 then no need to judge batch_size != 0
            break

    return (hit_count.item() / total)

def eval_genomes(genomes, config):

    if torch.cuda.is_available():
        gpu = True
        print("Running on GPU!")
    else:
        gpu = False
        print("Running on CPU!")


    lrfile= open("lr.txt", "r")
    tmp = lrfile.readline().rstrip('\n')
    lr = float(tmp)
    tmp = lrfile.readline().rstrip('\n')
    delta = float(tmp)
    lrfile.close()

    #genomes_id_set = set()

    j = 0
    for genome_id, genome in genomes:
        j += 1

        #setup the network, use saved net
        #if genome_id in net_dict:
        #    net = net_dict[genome_id]
        #else:
        net = evaluate_torch.Net(config, genome)

        if gpu:
            net.cuda()

        criterion = nn.CrossEntropyLoss()  # use a Classification Cross-Entropy loss
        optimizer = optim.SGD(net.parameters(), lr, momentum=0.9)

        #Evalute the fitness before trainning
        evaluate_batch_size = 100
        start = int(random() * (len(trainloader) - evaluate_batch_size))

        fit = eval_fitness(net, trainloader, evaluate_batch_size, torch_batch_size, start, gpu)


        comp = open("comp.csv", "a")
        comp.write('{0},{1:3.3f},'.format(j, fit))
        print('Before: {0}: {1:3.3f}'.format(j, fit))
        ###

        losses_len = 100
        losses = np.array([0.0] * losses_len)

        #train the network
        epoch = 0
        training = True
        train_epoch = 10
        while training and epoch < train_epoch:  # loop over the dataset multiple times
        #for epoch in range(10):
            epoch += 1
            running_loss = 0.0
            last_running_loss = 0.0

            for i, data in enumerate(trainloader, 0):
                # get the inputs
                inputs, labels = data

                # wrap them in Variable
                if gpu:
                    inputs, labels = Variable(inputs).cuda(), Variable(labels).cuda()
                else:
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

                # print statistics
                if i % 200 == 199:  # print every 200 mini-batches
                    print('[%d, %5d] loss: %.3f' % (epoch, i + 1, running_loss / 200))
                    if ((abs(last_running_loss - running_loss)/200 < delta) or
                            (last_running_loss != 0) and (running_loss > last_running_loss)):
                        training = False
                        print("Stopped trainning")
                        break;
                    #print(abs(last_running_loss - running_loss))
                    last_running_loss = running_loss
                    running_loss = 0.0
        print('Finished Training')

        #evaluate the fitness

        net.write_back_parameters(genome)

        evaluate_batch_size = 0
        start = 0
        fit = eval_fitness(net, trainloader, evaluate_batch_size, torch_batch_size, start, gpu)

        genome.fitness = fit
        print('After: {0:3.3f}, {1}'.format(genome.fitness, genome_id))
        comp.write('{0:3.3f},{1},{2:3.6f},{3:3.6f}\n'.format(genome.fitness, genome_id, lr, delta))
        comp.close

    #del the net not in current population
    #net_id = set(net_dict.keys())
    #net_to_del = net_id - genomes_id_set
    #for genome_id in net_to_del:
    #    del net_dict[genome_id]

# Load configuration.
config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                     neat.DefaultSpeciesSet, neat.DefaultStagnation,
                     'config-mnist')

# reset result file
res = open("result.csv", "w")
best = open("best.txt", "w")
res.close()
best.close()
comp = open("comp.csv", "w")
comp.close()

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


net = evaluate_torch.Net(config, winner)
if gpu:
    net.cuda()

# train the winner for some epoche
lrfile = open("lr.txt", "r")
tmp = lrfile.readline().rstrip('\n')
lr = float(tmp)
lrfile.close()

criterion = nn.CrossEntropyLoss()  # use a Classification Cross-Entropy loss
optimizer = optim.SGD(net.parameters(), lr, momentum=0.9)

final_train_epoch = 10

for epoch in range(final_train_epoch):

    running_loss = 0.0

    for i, data in enumerate(trainloader, 0):
        # get the inputs
        inputs, labels = data

        # wrap them in Variable
        if gpu:
            inputs, labels = Variable(inputs).cuda(), Variable(labels).cuda()
        else:
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

        # print statistics
        if i % 200 == 199:  # print every 200 mini-batches
            print('[%d, %5d] loss: %.3f' % (epoch, i + 1, running_loss / 200))
            running_loss = 0.0
        print('Finished Final Training')

evaluate_batch_size = 0
start = 0
fit = eval_fitness(net, testloader, evaluate_batch_size, torch_batch_size, start, gpu)

print("Final fitness: {0:3.3f}".format(fit))

#TODO: wirte model to pytorch files

node_names = {# -28: '-28', -27: '-27', -26: '-26', -25: '-25', -24: '-24', -23: '-23', -22: '-22', -21: '-21',
              # -20: '-20', -19: '-19', -18: '-18', -17: '-17', -16: '-16', -15: '-15', -14: '-14', -13: '-13',
              # -12: '-12', -11: '-11', -10: '-10', -9: '-09', -8: '-08', -7: '-07', -6: '-06',
              -5: '-05', -4: '-04', -3: '-03', -2: '-02', -1: '-01', 0: '00', 1: '01', 2: '02',
              3: '03', 4: '04', 5: '05', 6: '06', 7: '07', 8: '08', 9: '09'}

# visualize.draw_net(config, winner, True, node_names=node_names)


# to delete:


"""
# evaluate the fitness

hit_count = 0
start = int(random() * (len(trainloader) - evaluate_batch_size * torch_batch_size))

i = 0
for num, data in enumerate(trainloader, start):
    i += 1
    # 得到输入数据
    inputs, labels = data

    # 包装数据
    if gpu:
        inputs, labels = Variable(inputs).cuda(), Variable(labels).cuda()
    else:
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
"""
"""
     evaluate_batch_size = 0
     start = 0 #int(random() * (len(trainloader) - evaluate_batch_size * torch_batch_size))

     i = 0
     total = 0
     for num, data in enumerate(trainloader, start):
         i += 1
         total += torch_batch_size
         # 得到输入数据
         inputs, labels = data

         # 包装数据
         if gpu:
             inputs, labels = Variable(inputs).cuda(), Variable(labels).cuda()
         else:
             inputs, labels = Variable(inputs), Variable(labels)

         try:

             outputs = net.forward(inputs)

             _, predicted = torch.max(outputs.data, 1)
             hit_count += (predicted == labels).sum()

         except Exception as e:
             print(e)
             genome.fitness = 0
         #if (i == evaluate_batch_size - 1):
          #   break
"""


"""
correct = 0
total = 0

for num, data in enumerate(testloader, 0):
    total += torch_batch_size
    # 得到输入数据
    inputs, labels = data

    # 包装数据
    if gpu:
        inputs, labels = Variable(inputs).cuda(), Variable(labels).cuda()
    else:
        inputs, labels = Variable(inputs), Variable(labels)

    try:

        outputs = net.forward(inputs)

        _, predicted = torch.max(outputs.data, 1)
        correct += (predicted == labels).sum()

    except Exception as e:
        print(e)
"""