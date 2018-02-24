'''
Several procedures to train a CNN.
# Training with a scheduler
# Fine-tuning with a feature extractor
'''

# Pytorch core
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import torchvision
from torchvision import datasets, models

# PyTorch Utils 
from pytorch_utils import *

# Generic 
import numpy as np
import matplotlib.pyplot as plt
import time
import os
import copy


# Train and Validation
def train_model_val(dataloader, model, criterion, optimizer, scheduler, epochs=25):
    model_file='newcifar10.pth'
    since = time.time()
    use_gpu = torch.cuda.is_available()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(epochs):
        print('Epoch {}/{}'.format(epoch+1, epochs))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                scheduler.step()
                model.train(True)  # Set model to training mode
            else:
                model.train(False)  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for data in dataloaders[phase]:
                # get the inputs
                inputs, labels = data

                # wrap them in Variable
                if use_gpu:
                    inputs = Variable(inputs.cuda())
                    labels = Variable(labels.cuda())
                else:
                    inputs, labels = Variable(inputs), Variable(labels)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                outputs = model(inputs)
                _, preds = torch.max(outputs.data, 1)
                loss = criterion(outputs, labels)

                # backward + optimize only if in training phase
                if phase == 'train':
                    loss.backward()
                    optimizer.step()

                # statistics
                # loss * batch_size
                running_loss += loss.data[0] * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


# Train without a validation folder
def train_model(trainloader, model, criterion, optimizer, scheduler, epochs=25):
    use_gpu = torch.cuda.is_available()
    model.train(True)  # Set model to training mode
    since = time.time()

    # Here we store the best model
    model_file = 's_trained.pth'
    best_model_wts = copy.deepcopy(model.state_dict())
    # Statistics 
    best_acc = 0.0
    best_loss = 5.0
    total_step = 0.0

    for epoch in range(epochs):
        print('Epoch {}/{}'.format(epoch+1, epochs))
        print('-' * 10)

        # scheduler.step()
        running_loss = 0.0
        running_corrects = 0
        # Iterate over data.
        for step, data in enumerate(trainloader, 0):
            # get the input batch
            inputs, labels = data

            # wrap them in Variable
            if use_gpu:
                inputs = Variable(inputs.cuda())
                labels = Variable(labels.cuda())
            else:
                inputs, labels = Variable(inputs), Variable(labels)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            # Compute accuracy
            preds = outputs.cpu().data.max(1)[1]
            #_, preds = torch.max(outputs.data, 1)
            #  running_corrects = (labels.float() == preds.float().squeeze().mean())

            # statistics
            # loss * batch_size
            running_loss += loss.data[0]
	        # compute accuracy
            batch_correct = preds.eq(labels.cpu().data).sum()
            batch_size = labels.size(0)
            running_corrects += batch_correct / batch_size
            # running_corrects += torch.sum(preds == labels.data)

            if step % 1000 == 999:  # print every 1000 mini-batches
                step_loss = running_loss / 1000
                step_acc = running_corrects / 1000
                print('Epoch: {} Step: {} Loss: {:.3f} Acc: {:.3f}'.format(
                    epoch+1, step+1, step_loss, step_acc))
                # print('[%d, %5d] loss: %.3f' % (epoch + 1, step + 1,
                #                               step_loss))

                total_step += step + 1

                # ============ Logging ============#
                # (1) Log the scalar values
                info = {
                    'loss': step_loss,
                    'accuracy': step_acc
                }

                # (2) Log CSV file 
                log_csv(total_step, info['accuracy'], info['loss'])
                # (3) Tensorboard specific logging
                tensorboard_log(total_step, model, info)

                # for each epoch, save best model
                if best_loss > step_loss:
                    print('loss improved from %.3f to %.3f'
                        % (best_loss, step_loss))
                    print('Saving model to ' + model_file + "...\n")
                    best_loss = step_loss
                    best_acc = step_acc
                    best_model_wts = copy.deepcopy(model.state_dict())
                    torch.save(model.state_dict(), model_file)
                    torch.save(model, "dump_model.pth")
                
                ## EARLY STOPPING ## 
                if best_loss <= 0.199 and epoch >= 9:
                    print('EARLY STOPPING!')
                    time_elapsed = time.time() - since
                    print('Training complete in {:.0f}m {:.0f}s'.format(
                        time_elapsed // 60, time_elapsed % 60))
                    print('Best val Acc: {:4f}'.format(best_acc))
                    
                    # load best model weights
                    model.load_state_dict(best_model_wts)
                    return model

                # Reset running loss for next iteration
                running_loss = 0.0

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


def test_model_cifar10(testloader, model):
    classes = ('plane', 'car', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck')
    correct = 0
    total = 0
    total_time = 0 
    model.eval()

    for i, (batch, labels) in enumerate(testloader):
        batch = batch.cuda()
        inputs = Variable(batch, volatile=True)
        t0 = time.time() 
        outputs = model(inputs)
        t1 = time.time() 
        if i % 10 == 9 or i == 0:  
            print('Prediction time for batch %d: %.6f ' % (i+1, t1-t0))
        if not i: 
            total_time = total_time + (t1 - t0)
        _, predicted = torch.max(outputs.data, 1)
        correct += (predicted == labels.cuda()).sum()      
        total += labels.size(0)

    print('Accuracy of the network on the 10000 test images: %d %%' %
          (100 * correct / total))
    print("Average prediction time %.6f %d" % (float(total_time)/(i + 1), i + 1))   

    class_correct = list(0. for i in range(10))
    class_total = list(0. for i in range(10))
    for data in testloader:
        images, labels = data
        outputs = model(Variable(images.cuda(), volatile=True))
        _, predicted = torch.max(outputs.data, 1)
        c = (predicted == labels.cuda()).squeeze()
        for i in range(4):
            label = labels[i]
            class_correct[label] += c[i]
            class_total[label] += 1

    for i in range(10):
        print('Accuracy of %5s : %2d %%' %
              (classes[i], 100 * class_correct[i] / class_total[i]))
