# Code referenced from https://gist.github.com/gyglim/1f8dfb1b5c82627ae3efcfbbadb9f514
import tensorflow as tf
import numpy as np
import scipy.misc
import scipy.io as sio
try:
    from StringIO import StringIO  # Python 2.7
except ImportError:
    from io import BytesIO         # Python 3.x

import torch

# Defines utils functions to log info for Tensorboard
# Integrated new functions to log to simple custom .csv files
class Logger(object):

    def __init__(self, log_dir):
        """Create a summary writer logging to log_dir."""
        self.writer = tf.summary.FileWriter(log_dir)

    def scalar_summary(self, tag, value, step):
        """Log a scalar variable."""
        if isinstance(value, torch.Tensor):
            value = value.item()
        summary = tf.Summary(value=[tf.Summary.Value(tag=tag, simple_value=value)])
        self.writer.add_summary(summary, step)

    def image_summary(self, tag, images, step):
        """Log a list of images."""

        img_summaries = []
        for i, img in enumerate(images):
            # Write the image to a string
            try:
                s = StringIO()
            except:
                s = BytesIO()
            scipy.misc.toimage(img).save(s, format="png")

            # Create an Image object
            img_sum = tf.Summary.Image(encoded_image_string=s.getvalue(),
                                       height=img.shape[0],
                                       width=img.shape[1])
            # Create a Summary value
            img_summaries.append(tf.Summary.Value(tag='%s/%d' % (tag, i), image=img_sum))

        # Create and write Summary
        summary = tf.Summary(value=img_summaries)
        self.writer.add_summary(summary, step)

    def histo_summary(self, tag, values, step, bins=1000):
        """Log a histogram of the tensor of values."""

        # Create a histogram using numpy
        counts, bin_edges = np.histogram(values, bins=bins)

        # Fill the fields of the histogram proto
        hist = tf.HistogramProto()
        hist.min = float(np.min(values))
        hist.max = float(np.max(values))
        hist.num = int(np.prod(values.shape))
        hist.sum = float(np.sum(values))
        hist.sum_squares = float(np.sum(values**2))

        # Drop the start of the first bin
        bin_edges = bin_edges[1:]

        # Add bin edges and counts
        for edge in bin_edges:
            hist.bucket_limit.append(edge)
        for c in counts:
            hist.bucket.append(c)

        # Create and write Summary
        summary = tf.Summary(value=[tf.Summary.Value(tag=tag, histo=hist)])
        self.writer.add_summary(summary, step)
        self.writer.flush()

    def log_csv(self, step, acc, loss, val=0, file='cifar10.csv'):
        with open(file, 'a') as out:
            out.write("%d,%.3f,%.3f\n" % (step, acc, loss))
            out.close()

    def log_test(self, step, val=0, file='test.csv'):
        with open(file, 'a') as out:
            out.write("%d,%.3f\n" % (step, val))
            out.close()

    def log_compression(self, layer_weights, compression_factor, file='compression.txt'):
        with open(file, 'a') as out:
            out.write("Weights before: %d - Weights after:%d - Compression ratio: %.4f\n" %
                    (layer_weights.size, (layer_weights.size / compression_factor), compression_factor))
            out.close()

    def tensorboard_log(self, steps, model, info):
        for tag, value in info.items():
            self.scalar_summary(tag, value, steps)

        # (2) Log values and gradients of the parameters (histogram)
        # too slow, for now leave commented
        '''
        for tag, value in model.named_parameters():
            # print(str(tag)+"  "+str(value))
            tag = tag.replace('.', '/')
            logger.histo_summary(tag, to_np(value), steps)
            if 'bn' not in tag and value.grad is not None:
                logger.histo_summary(
                    tag + '/grad', to_np(value.grad), steps)
        '''

    #### HELPER TO SAVE/LOAD .mat files to use with
    #### Matlab Tensorlab toolbox
    def save_weigths_to_mat(self, allweights, save_dir):
        for idx, weights in enumerate(allweights):
            name = os.path.join(save_dir, "conv" + str(
                idx) + ".mat")  # conv1.mat, conv2.mat, ...
            sio.savemat(name,  {'weights': weights})


    def dump_model_weights(self, model, save_dir='./dumps'):
        '''
        Dump weights for all Conv2D layers and saves it as .mat files
        TODO: Add check if file exists
        '''
        save_dir = os.path.join(os.getcwd(), save_dir)
        # create dir if not exists
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)

        allweights = []
        for layer in model.modules():
            if isinstance(layer, torch.nn.modules.conv.Conv2d):
                print('Saving layer: ' + str(layer) + ' to ' + save_dir)
                tmp = []
                tmp.append(layer.weight)
                tmp.append(layer.bias)
                allweights.append(tmp)

        save_weigths_to_mat(allweights, save_dir)


    def dump_layer_weights(self, layer, filename="weights.mat", save_dir='dumps/'):
        '''
        Dump weights for specified layer as .mat file
        '''
        save_dir = os.path.join(os.getcwd(), save_dir)
        # create dir if not exists
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)

        weights = get_layer_weights(layer, numpy=True)
        print('Saving layer ' + str(layer) + " to" + save_dir)

        name = save_dir + filename
        print(name)
        sio.savemat(name,  {'weights': weights})


    def load_cpd_weights(self, filename):
        import os

        if not os.path.isfile(filename):
            print("ERROR: .mat file not found")
            return

        # load struct 'cpd_s' from file
        mat_contents = sio.loadmat(filename)['cpd_s']

        #  bias = mat_contents['bias'][0][0][0]  # retrieve bias weights
        cpd = mat_contents['weights'][0][0]  # cell of 4 tensors

        f_last = cpd[0][0]
        f_first = cpd[0][1]
        f_vertical = cpd[0][2]
        f_horizontal = cpd[0][3]
        print('Loaded cpd weights succesfully.')

        return f_last, f_first, f_vertical, f_horizontal  # , bias
