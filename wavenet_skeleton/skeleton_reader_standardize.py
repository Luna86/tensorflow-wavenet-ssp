import fnmatch
import os
import re
import threading

import librosa
import numpy as np
import tensorflow as tf


def normalizationStats(completeData):
    data_mean = np.mean(completeData, axis=0)
    data_std = np.std(completeData, axis=0)
    #       dimensions_to_ignore = []
    #       if not full_skeleton:
    #               dimensions_to_ignore = [0,1,2,3,4,5]
    #       dimensions_to_ignore.extend(list(np.where(data_std < 1e-4)[0]))
    '''Returns the mean of data, std, and dimensions with small std. Which we later ignore. '''
    return data_mean, data_std


def normalizeTensor(inputTensor, data_mean, data_std):
    meanTensor = data_mean.reshape((1, 1, inputTensor.shape[2]))
    meanTensor = np.repeat(meanTensor, inputTensor.shape[0], axis=0)
    meanTensor = np.repeat(meanTensor, inputTensor.shape[1], axis=1)
    stdTensor = data_std.reshape((1, 1, inputTensor.shape[2]))
    stdTensor = np.repeat(stdTensor, inputTensor.shape[0], axis=0)
    stdTensor = np.repeat(stdTensor, inputTensor.shape[1], axis=1)
    normalizedTensor = np.divide((inputTensor - meanTensor), stdTensor)
    return normalizedTensor



# prepare data for training from csv files.
# each csv file contain N x 45 parentToChildVect for one subject
# data prepared as list of length N, with each element a T x 42 matrix (without root) or T x 45 matrix (with root)
# T = len_sample, N = num_sample
# len_samples == chunk length

def sample_data(input_data, len_samples, bWithRoot, bDiffRootLoc):
    overall_sampled_data = []
    # i = 0;
    window_shift = 25
    tempStartFrame = 0
    while tempStartFrame + len_samples + 1 < input_data.shape[0]:
        # if (i+1)*(len_samples+1) >= input_data.shape[0]:
        # break
        s_t = tempStartFrame
        e_t = tempStartFrame + len_samples + 1
        if bWithRoot:
            sample_data = np.copy(input_data[s_t:e_t, :])  # should make a copy instead of aliasing
            # sample_data[:, 0:3] =  sample_data[:, 0:3]/100;
            # sample_data[:, 1:3] =  sample_data[:, 1:3]/100;
        else:
            sample_data = input_data[s_t:e_t, 3:]  # (W+1) x dim

        if bWithRoot and bDiffRootLoc:
            initRoot = sample_data[0, 0:3];
            initRoot = np.tile(initRoot, (sample_data.shape[0], 1))
            sample_data[:, 0:3] = sample_data[:, 0:3] - initRoot;
        # print np.max(sample_data[:, 1])
        if not np.isnan(np.max(sample_data)):
            overall_sampled_data.append(sample_data)
            # print 'Nan is detected, but ignored'
        # else:
        # overall_data.append([class_ids[x] fddor x in sample_text])

        # i = i + 1
        tempStartFrame = tempStartFrame + window_shift
    initLength = len(overall_sampled_data)
    finalLength = initLength

    if bAugRootLoc:
        overall_sampled_data = AugmentRootLocation(overall_sampled_data)
        finalLength = len(overall_sampled_data)
    print ('chunkNum: %d -> %d' % (initLength, finalLength))
    return overall_sampled_data


# with rotation normalization such that first frame of each chunk faces z direction (0,0,1)
'''
def sample_data_rotNom(input_data, len_samples, bWithRoot, bDiffRootLoc):
    overall_sampled_data = []
    # i = 0;
    window_shift = 25
    tempStartFrame = 0
    while tempStartFrame + len_samples + 1 < input_data.shape[0]:
        # if (i+1)*(len_samples+1) >= input_data.shape[0]:
        # break
        s_t = tempStartFrame
        e_t = tempStartFrame + len_samples + 1
        if bWithRoot:
            sample_data = np.copy(input_data[s_t:e_t, :])  # should make a copy instead of aliasing
            # sample_data[:, 0:3] =  sample_data[:, 0:3]/100;
            # sample_data[:, 1:3] =  sample_data[:, 1:3]/100;
        else:
            sample_data = input_data[s_t:e_t, 3:]  # (W+1) x dim

        if bWithRoot and bDiffRootLoc:
            initRoot = sample_data[0, 0:3];
            initRoot = np.tile(initRoot, (sample_data.shape[0], 1))
            sample_data[:, 0:3] = sample_data[:, 0:3] - initRoot;

        # Apply rotation normalization
        sample_data_3d = sample_data.reshape(sample_data.shape[0], sample_data.shape[1] / 3, 3);  # 101x15x3
        rot2z = input_data_rot2Z[s_t, :];  # 1x9
        rot2z = rot2z.reshape(3, 3);
        rot2z = rot2z.transpose();  # 3x3...Not sure transpose is needed.. should check
        for i in np.arange(0, sample_data_3d.shape[0]):
            tempPose = sample_data_3d[i, :, :];  # 15x3
            tempPose = tempPose.transpose()  # 3x15
            tempPose = np.dot(rot2z, tempPose);  # 3x15 , rotated
            sample_data_3d[i, :, :] = tempPose.transpose();
        sample_data = sample_data_3d.reshape(sample_data.shape[0], sample_data.shape[1]);
        # print np.max(sample_data[:, 1])
        if not np.isnan(np.max(sample_data)):
            overall_sampled_data.append(sample_data)
            # print 'Nan is detected, but ignored'
        # else:
        # overall_data.append([class_ids[x] fddor x in sample_text])

        # i = i + 1
        tempStartFrame = tempStartFrame + window_shift
    initLength = len(overall_sampled_data)
    finalLength = initLength

    if bAugRootLoc:
        overall_sampled_data = AugmentRootLocation(overall_sampled_data)
        finalLength = len(overall_sampled_data)
    print ('chunkNum: %d -> %d' % (initLength, finalLength))
    return overall_sampled_data
'''


def sample_data_rotNom(input_data, len_samples, bWithRoot, bDiffRootLoc):
    overall_data = []
    i = 0
    while (i + 1) * (len_samples + 1) < input_data.shape[0]:
        # if (i+1)*(len_samples+1) >= input_data.shape[0]:
        # break
        s_t = int(i * len_samples)
        e_t = int((i + 1) * len_samples + 1)
        sample = input_data[s_t:e_t, :]
        sample[1:, :3] = sample[1:, :3] - sample[:-1, :3]
        sample[0, :3] = 0
        '''update: keep root location but calculate the relative change. Always make the first frame of each chunk at center of the dome.'''
        '''sample data is of size len_sample x 42. Normalize the chunk so that the first frame is always facing the same direction
        reshape to len_sample x 15 x 3'''
        sample_reshape = sample.reshape((sample.shape[0], 15, 3))
        shoulder = sample_reshape[0,3,:] - sample_reshape[0,9,:]
        shoulder[1] = 0
        shoulder = shoulder/np.linalg.norm(shoulder)
        R = np.array([[shoulder[2], 0, -shoulder[0]], [0, 1, 0], [shoulder[0], 0, shoulder[2]]])
        #print 'R det:{0}'.format(np.linalg.det(R))
        #print 'R.T - inv(R)={0}'.format(R.T - np.linalg.inv(R))
        sample_reshape = np.dot(sample_reshape, R.T)
        sample_out = sample_reshape.reshape((sample_reshape.shape[0], 45))
        overall_data.append(sample_out)
        i = i + 1
    # overall_data = np.array(overall_data,dtype=np.float64)
    # train_data = overall_data[:,:-1,3:]
    # label_data = overall_data[:,1:,3:]
    return overall_data


def createTrain(datadir, num_samples=1000, len_samples=25,
                bWithRoot=0, data_mean=None, data_std=None, testingSeq=None,
                bDiffRootLoc=0, bNormRootRot=0):
    overall_data = []
    completeRawData = []
    for scene in os.listdir(datadir):
        #               if len(overall_data) > num_samples:
        #                       break
        subjects = os.listdir(os.path.join(datadir, scene))
        print('scene {0} has {1} subjects'.format(scene, len(subjects)))
        for sub in subjects:
            filename = os.path.join(datadir, scene, sub)
            if os.path.isdir(filename) == True or sub[-1] == 'z':
                continue
            if sub[-1] == 'z':  # rotation file
                continue
            bSkip = False
            if testingSeq is not None:
                testSeqNum = len(testingSeq) / 2
                for i in range(0, testSeqNum, 2):
                    if testingSeq[i] == scene and testingSeq[i + 1] == sub:
                        print ('Skip test data: %s' % filename)
                        bSkip = True
                        break
                    if bSkip == True:
                        break
            if bSkip == True:
                continue
            overall_data_sub= createTrainSubject(filename, len_samples, bWithRoot, bDiffRootLoc, bNormRootRot)
            if overall_data_sub is None:
                continue
            overall_data.extend(overall_data_sub)
            # completeRawData.extend(rawData_sub )   #do not use this anymore
            #                       if (len(overall_data)>num_samples):
            #                               overall_data = overall_data[:num_samples]
            #                               break
    # print('training_data: {0} distinct samples in total'.format(len(overall_data)))
    overall_data = np.array(overall_data, dtype=np.float32)  # (chunkNum, chunkLeng+1, dim)
    overall_data = np.swapaxes(overall_data, 0, 1)  # (chunkLeng+1, chunkNum, dim),  e.g.,(101, 5307, 45)
    train_data = overall_data[:-1, :, :]
    #label_data = overall_data[1:, :, :]

    overall_data_2d = overall_data.reshape(overall_data.shape[0] * overall_data.shape[1], overall_data.shape[2])
    if np.isnan(np.max(overall_data_2d)):
        print ('Warning!!!: completeRawData:: Nan is detected.')
    else:
        print ('Good!: completeRawData:: Nan is not detected.')
    if (data_mean is None) and (data_std is None):
        data_mean, data_std = normalizationStats(overall_data_2d)
    train_data = normalizeTensor(train_data, data_mean, data_std)
    #label_data = normalizeTensor(label_data, data_mean, data_std)
    #return train_data, label_data, data_mean, data_std
    return train_data, data_mean, data_std

def createTrainSubject(filename, len_samples, bWithRoot, bDiffRootLoc, bNormRootRot):
    raw_data = np.loadtxt(filename, delimiter=',')
    print ('frame_length: %d' % raw_data.shape[0])
    if raw_data.shape[0] < 100:
        return None, None

    if np.isnan(np.max(raw_data)):
        print ('Warning: Nan is detected: {0}'.format(filename))
    # input_data2 = np.genfromtxt(filename, delimiter=',')

    # print '{1} frames in subject {0}'.format(os.path.basename(filename), raw_data.shape[0])

    # [train_data,label_data] = sample_data(input_text,num_samples,len_samples,class_ids)
    #todo: what is this bNormRootRot doing? change to diff of root location and rotating according to first frame
    if bNormRootRot:
        #filename_rot2z = filename + '_rot2z'
        #rot2z_data = np.loadtxt(filename_rot2z, delimiter=',')
        #if (raw_data.shape[0] != rot2z_data.shape[0]):
        #    print ("Error: raw_data.shape[0] != rot2z_data.shape[0] (%d != %d)" % (
        #        raw_data.shape[0], rot2z_data.shape[0]))
        #    return None, None
        overall_sampled_data = sample_data_rotNom(raw_data, len_samples, bWithRoot, bDiffRootLoc)
    else:
        overall_sampled_data = sample_data(raw_data, len_samples, bWithRoot, bDiffRootLoc)
    # train_data = np.swapaxis(train_data, 0, 1)
    # label_data = np.swapaxis(label_data, 0, 1)
    # dim = T x N x 45
    return overall_sampled_data


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
        self.sample_placeholder = tf.placeholder(dtype=tf.float32, shape=(sample_size, 42))
        self.queue = tf.RandomShuffleQueue(capacity=queue_size,
                                           min_after_dequeue=10,
                                           dtypes=['float32'],
                                           shapes=[(sample_size, 42)])
        self.enqueue = self.queue.enqueue([self.sample_placeholder])

        # TODO Find a better way to check this.
        # Checking inside the AudioReader's thread makes it hard to terminate
        # the execution of the script, so we do it in the constructor for now.
        #if not find_files(skeleton_dir):
        #    raise ValueError("No skeleton files found in '{}'.".format(skeleton_dir))

    def dequeue(self, num_elements):
        output = self.queue.dequeue_many(num_elements)
        #symbolic dequeue... cannot check dimension before run
        #print('shape of output:{0}'.format(output.get_shape()))
        #print('content of output:{0}'.format(output))
        return output

    def thread_main(self, sess):
        #buffer_ = np.array([])
        stop = False
        #Go through the dataset multiple times
        #train_data: N x T x skeletonDim tensor. N: number of chunks. T: length of sample
        [train_data, data_mean, data_std] = createTrain(datadir, num_samples=1000, len_samples=self.sample_size,
                                 bWithRoot=1, data_mean=None, data_std=None, testingSeq=None,
                                bDiffRootLoc=1, bNormRootRot=1)

        while not stop:
            #todo: first aggregate all training samples into a tensor and do standardization
            #todo: then use while loop to iterate through all samples multiple times
            #output: a tensor of all training data
            #iterator = load_generic_skeleton(self.skeleton_dir)
            #for skeleton, filename in iterator:
            for index in range(train_data.shape[0]):
                if self.coord.should_stop():
                    stop = True
                    break

                #if self.sample_size:
                    # Cut samples into fixed size pieces
                    #buffer_ = np.append(buffer_, skeleton)
                    #while skeleton.shape[0] > self.sample_size:
                        #todo:  need to standardize the data and add root location
                        #piece = skeleton[:self.sample_size,3:]
                piece = train_data[index, :, :]
                        #print('shape of piece: {0}'.format(piece.shape))
                        #print('content of piece: \n{0}'.format(piece))
                        #'''this is where data is fed into the queue for future fetch'''
                sess.run(self.enqueue, feed_dict={self.sample_placeholder: piece})
                #skeleton = skeleton[self.sample_size:,:]
                #else:
                    #sess.run(self.enqueue,
                             #feed_dict={self.sample_placeholder: skeleton})

    def start_threads(self, sess, n_threads=1):
        for _ in range(n_threads):
            thread = threading.Thread(target=self.thread_main, args=(sess,))
            thread.daemon = True  # Thread will close when parent quits.
            thread.start()
            self.threads.append(thread)
        return self.threads
