import fnmatch
import os
import re
import threading

import librosa
import numpy as np
import tensorflow as tf


def find_files(directory, pattern='s*'):
    '''Recursively finds all files matching the pattern.'''
    files = []
    for root, dirnames, filenames in os.walk(directory):
        for filename in fnmatch.filter(filenames, pattern):
            files.append(os.path.join(root, filename))
    #print('files finded:{0}'.format(files))
    return files


def load_generic_skeleton(directory):
    '''Generator that yields audio waveforms from the directory.'''
    files = find_files(directory)
    print('length of file list {0}'.format(len(files)))
    for filename in files:
        #skeleton, _ = librosa.load(filename, sr=sample_rate, mono=True)
        skeleton = np.loadtxt(filename, delimiter=',')
        #print('skeleton_shape:{0}'.format(skeleton.shape))
        #skeleton = skeleton.reshape(-1, 1)
        yield skeleton, filename


class SkeletonReader(object):
    '''Generic background audio reader that preprocesses audio files
    and enqueues them into a TensorFlow queue.'''

    def __init__(self,
                 skeleton_dir,
                 coord,
                 sample_size=None,
                 queue_size=256):
        self.skeleton_dir = skeleton_dir
        self.coord = coord
        self.sample_size = sample_size
        self.threads = []
        #sample placeholder shape in audio data = n x 1, where n is length of sample points in an audio fragment.
        #self.sample_placeholder = tf.placeholder(dtype=tf.float32, shape=None)
        #taking into one sample of size (T x 42), where T is unknown before time
        self.sample_placeholder = tf.placeholder(dtype=tf.float32, shape=None)
        self.queue = tf.PaddingFIFOQueue(capacity=queue_size,
                                         dtypes=['float32'],
                                         shapes=[(None, 42)])
        self.enqueue = self.queue.enqueue([self.sample_placeholder])

        # TODO Find a better way to check this.
        # Checking inside the AudioReader's thread makes it hard to terminate
        # the execution of the script, so we do it in the constructor for now.
        if not find_files(skeleton_dir):
            raise ValueError("No skeleton files found in '{}'.".format(skeleton_dir))

    def dequeue(self, num_elements):
        output = self.queue.dequeue_many(num_elements)
        #symbolic dequeue... cannot check dimension before run
        #print('shape of output:{0}'.format(output.get_shape()))
        #print('content of output:{0}'.format(output))
        return output

    def thread_main(self, sess):
        #buffer_ = np.array([])
        stop = False
        # Go through the dataset multiple times
        while not stop:
            iterator = load_generic_skeleton(self.skeleton_dir)
            for skeleton, filename in iterator:
                if self.coord.should_stop():
                    stop = True
                    break

                if self.sample_size:
                    # Cut samples into fixed size pieces
                    #buffer_ = np.append(buffer_, skeleton)
                    while skeleton.shape[0] > self.sample_size:
                        #todo:  need to standardize the data and add root location
                        piece = skeleton[:self.sample_size,3:]
                        #print('shape of piece: {0}'.format(piece.shape))
                        #print('content of piece: \n{0}'.format(piece))
                        '''this is where data is fed into the queue for future fetch'''
                        if not np.isnan(np.sum(piece)):
                            sess.run(self.enqueue, feed_dict={self.sample_placeholder: piece})
                        skeleton = skeleton[self.sample_size:,:]
                else:
                    if not np.isnan(np.sum(skeleton)):
                        sess.run(self.enqueue, feed_dict={self.sample_placeholder: skeleton})

    def start_threads(self, sess, n_threads=1):
        for _ in range(n_threads):
            thread = threading.Thread(target=self.thread_main, args=(sess,))
            thread.daemon = True  # Thread will close when parent quits.
            thread.start()
            self.threads.append(thread)
        return self.threads
