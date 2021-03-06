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
from logger import Logger
from models.metrics import accuracy

# Generic
import numpy as np
import matplotlib.pyplot as plt
import time
import os
import copy


# Train and Validation
def train_model_val(model, dataloaders, criterion, optimizer, scheduler, epochs=50):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    since = time.time()
    logger = Logger(log_dir="./logs")

    # switching optimizer after a certain thresh
    switched_opt = False

    # Here we store the best model
    dirname = os.path.dirname(__file__)
    wts_filename = os.path.join(dirname, './checkpoints/just_trained.spth')
    model_filename = os.path.join(dirname, './models/just_trained.pth')
    best_model_wts = copy.deepcopy(model.state_dict())

    # Statistics
    best_acc = 0.0
    best_loss = 100.0
    total_step = 0.0

    model.train(True)  # Set model to training mode

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
            total = 0

            # Iterate over data.
            for i, batch in enumerate(dataloaders[phase]):
                # get the inputs
                inputs, labels = batch
                inputs, labels = inputs.to(device), labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)

                # backward + optimize only if in training phase
                if phase == 'train':
                    loss.backward()
                    optimizer.step()

                # statistics
                # loss * batch_size
                running_loss += loss.item()
                running_corrects += torch.sum(preds == labels.data)
                total += inputs.size(0)

            # len(batch) or dataloader[phase].batch_size
            batch_size = inputs.size(0)
            epoch_loss = running_loss / i+1
            epoch_acc = 100 * running_corrects/total

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
# % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))

            if phase == 'train':
                train_loss_to_plot = epoch_loss

            else:
                # ============ Logging ============#
                # (1) Log the scalar values
                metrics = {
                    'loss': train_loss_to_plot,  # train loss
                    'accuracy': epoch_acc  # val accuracy
                }

                # (2) Log CSV file
                logger.log_csv(epoch, metrics['accuracy'], metrics['loss'])
                # (3) Tensorboard specific logging
                logger.tensorboard_log(epoch, model, metrics)

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                print('Acc improved from %.3f to %.3f'
                      % (best_acc, epoch_acc))
                logger.log_test(epoch, epoch_acc)

                print('Saving model to ' + model_filename + "...\n")
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                torch.save(model.state_dict(), wts_filename)
                torch.save(model, model_filename)

            # Switch to SGD + Nesterov
            if train_loss_to_plot <= 0.6 and not switched_opt:
                print('Switching to SGD wt Nesterov Momentum...')
                optimizer = optim.SGD(model.parameters(),
                                      lr=1e-4, momentum=0.9, nesterov=True)
                switched_opt = True

            ## EARLY STOPPING ##
            if train_loss_to_plot <= 0.150 and epoch >= 5:
                print('EARLY STOPPING!')
                print(train_loss_to_plot)
                time_elapsed = time.time() - since
                print('Training complete in {:.0f}m {:.0f}s'.format(
                    time_elapsed // 60, time_elapsed % 60))
                print('Best val Acc: {:4f}'.format(best_acc))
                # load best model weights
                model.load_state_dict(best_model_wts)
                return model

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


# Train without a validation but testing every X epochs
def train_test_model(dataloader, model, criterion, optimizer, scheduler, loss_threshold=0.3, epochs=25):
    trainloader = dataloader['train']
    testloader = dataloader['test']
    logger = Logger(log_dir='./logs')
    device = ("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    since = time.time()

    # switching optimizer after a certain thresh
    switched_opt = False

    # Here we store the best model
    dirname = os.path.dirname(__file__)
    wts_filename = 'checkpoints/just_trained.pth.tar'
    model_filename = 'checkpoints/best.pth'
    best_model_wts = copy.deepcopy(model.state_dict())

    # Statistics

    # train
    best_acc = 0.0
    best_loss = 100.0
    total_step = 0.0
    local_minima_loss = 0.0
    counter = 0

    # test
    best_test_acc = 0.0
    current_test_acc = 0.0
    plateau_counter = 0

    model.train(True)  # Set model to training mode

    for epoch in range(epochs):
        print('Epoch {}/{}'.format(epoch + 1, epochs))
        print('-' * 10)

        # scheduler.step()
        running_loss = 0.0
        running_corrects = 0

        # Iterate over data.
        for step, data in enumerate(trainloader, 0):
            # get the input batch
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward - compute loss - backward step - update step
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # Compute stats
            # preds = outputs.cpu().data.max(1)[1]
            _, preds = torch.max(outputs.data, 1)
            #  running_corrects = (labels.float() == preds.float().squeeze().mean())

            # loss * batch_size
            running_loss += loss.item()
            # compute accuracy
            # .item() for pytorch >= 0.4
            batch_correct = preds.eq(labels.data).sum().item()
            batch_size = labels.size(0)
            running_corrects += batch_correct / batch_size
            # running_corrects += torch.sum(preds == labels.data)
            progress_bar(step, len(trainloader), 'Loss: %.3f | Acc: %.3f (%d/%d)'
                         % (running_loss / 1000, running_corrects / 1000, batch_correct, batch_size))

            if step % 1000 == 999:  # print every 1000 mini-batches
                step_loss = running_loss / 1000
                step_acc = running_corrects / 1000

                print('Epoch: {} Step: {} Loss: {:.3f} Acc: {:.3f}'.format(
                    epoch + 1, step + 1, step_loss, step_acc))
                # print('[%d, %5d] loss: %.3f' % (epoch + 1, step + 1,
                #                               step_loss))

                total_step += step + 1

                # ============ Logging ============#
                # (1) Log the scalar values
                metrics = {
                    'loss': step_loss,
                    'accuracy': step_acc
                }

                # (2) Log CSV file
                print('logging...')
                logger.log_csv(
                    total_step, metrics['accuracy'], metrics['loss'])
                # (3) Tensorboard specific logging
                logger.tensorboard_log(total_step, model, metrics)

                # save checkpoint
                save_checkpoint(model.state_dict(),
                                (step_loss < best_loss), 'checkpoints/')

                # for each epoch, save best model
                if best_loss > step_loss:
                    print('loss improved from %.3f to %.3f'
                          % (best_loss, step_loss))
                    best_loss = step_loss
                    best_acc = step_acc
                    best_model_wts = copy.deepcopy(model.state_dict())

                    # Every X epochs, write model to files
                    if((epoch + 1) % 2 == 1 or epoch == 49):
                        print('Saving model to ' +
                              model_filename + "...\n")
                        # torch.save(best_model_wts, os.path.join(
                        #    dirname, wts_filename))
                        torch.save(model, os.path.join(
                            dirname, model_filename))

                    # Switch to SGD + Nesterov
                    if best_loss <= 0.6 and not switched_opt:
                        print('Switching to SGD wt Nesterov Momentum...')
                        optimizer = optim.SGD(
                            model.parameters(), lr=1e-4, momentum=0.9, nesterov=True)
                        switched_opt = True
                else:
                    # if it stagnates in local minima
                    local_minima_loss = best_loss
                    if step_loss - local_minima_loss < 0.020:
                        counter += 1
                        if counter >= 5 and not switched_opt:
                            print(
                                'Stuck in local minima. Switching to SGD wt Nesterov Momentum...')
                            optimizer = optim.SGD(
                                model.parameters(), lr=1e-4, momentum=0.9, nesterov=True)
                            switched_opt = True

                # Every 5 epochs, test model
                if((epoch + 1) % 5 == 0):
                    print("Testing...")
                    current_test_acc = quick_test_cifar(testloader, model)
                    # update accuracy wt. ternary assignment
                    best_test_acc = current_test_acc if current_test_acc > best_test_acc else best_test_acc
                    if current_test_acc > best_test_acc:
                        best_test_acc = current_test_acc
                        # zero the plateau counter
                        plateau_counter = 0
                    else:
                        plateau_counter += 1

                    # log test val
                    logger.log_test(total_step, best_test_acc)

                    # switch back model
                    model.train(True)  # Set model to training mode
                    model.cuda()

                ## EARLY STOPPING ##
                if (plateau_counter > 5) or (best_loss <= loss_threshold and epoch >= 5):
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

# Train without a validation folder


def train_model(trainloader, model, criterion, optimizer, scheduler, loss_threshold=0.3, epochs=25):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    since = time.time()

    switched_opt = False
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
            inputs, labels = inputs.to(device), labels.to(device)

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
            running_loss += loss.item()
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
                metrics = {
                    'loss': step_loss,
                    'accuracy': step_acc
                }

                # (2) Log CSV file
                log_csv(total_step, metrics['accuracy'], metrics['loss'])
                # (3) Tensorboard specific logging
                # tensorboard_log(total_step, model, metrics)

                # for each epoch, save best model
                if best_loss > step_loss:
                    print('loss improved from %.3f to %.3f'
                          % (best_loss, step_loss))
                    best_loss = step_loss
                    best_acc = step_acc

                    if((epoch+1) % 5 == 0):
                        print("Testing...")

                        print('Saving model to ' + model_file + "...\n")
                        best_model_wts = copy.deepcopy(model.state_dict())
                        torch.save(model.state_dict(), model_file)
                        torch.save(model, "dump_model.pth")

                    # Switch to SGD + Nesterov
                    if best_loss <= 0.6 and not switched_opt:
                        print('Switching to SGD wt Nesterov Momentum...')
                        optimizer = optim.SGD(
                            model.parameters(), lr=1e-4, momentum=0.9, nesterov=True)
                        switched_opt = True

                ## EARLY STOPPING ##
                if best_loss <= loss_threshold and epoch >= 500:
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
    model.cpu()

    for i, (inputs, labels) in enumerate(testloader):
        t0 = time.time()
        outputs = model(inputs)
        t1 = time.time()
        if i % 10 == 9 or i == 0:
            print('Prediction time for batch %d: %.6f ' % (i+1, t1-t0))

        if i != 0:
            total_time = total_time + (t1 - t0)
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels.cpu()).sum()
        total += labels.size(0)

    print('Accuracy of the network on the 10000 test images: %d %%' %
          (100 * correct / total))
    print("Average prediction time %.6f %d" %
          (float(total_time)/(i + 1), i + 1))

    class_correct = list(0. for i in range(10))
    class_total = list(0. for i in range(10))
    for data in testloader:
        images, labels = data
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        #_, predicted = torch.max(outputs.data, 1)
        c = (predicted == labels).squeeze()
        for i in range(4):
            label = labels[i]
            class_correct[label] += c[i]
            class_total[label] += 1

    for i in range(10):
        print('Accuracy of %5s : %2d %%' %
              (classes[i], 100 * class_correct[i] / class_total[i]))


# without class scores
def quick_test_cifar(testloader, model):
    classes = ('plane', 'car', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck')
    correct = 0
    total = 0
    total_time = 0
    model.train(False)
    model.cpu()

    for i, (batch, labels) in enumerate(testloader):
        # inputs = Variable(batch, volatile=True) # not needed Pytorch 4.0
        t0 = time.time()
        outputs = model(batch)
        t1 = time.time()

        if i != 0:
            total_time = total_time + (t1 - t0)
        _, predicted = torch.max(outputs.data, 1)
        correct += (predicted == labels.cpu()).sum()
        total += labels.size(0)

    print('Accuracy of the network on the 10000 test images: %d %%' %
          (100 * correct / total))
    print("Average prediction time %.6f %d" %
          (float(total_time) / (i + 1), i + 1))

    return correct/total
