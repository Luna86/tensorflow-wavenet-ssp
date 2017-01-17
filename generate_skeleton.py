from __future__ import division
from __future__ import print_function

import argparse
from datetime import datetime
import json
import os

import numpy as np
import tensorflow as tf
import scipy.io as sio

from wavenet_skeleton import WaveNetModel

SAMPLES = 16000
LOGDIR = './logdir'
WINDOW = 25 #1 second of past samples to take into account
WAVENET_PARAMS = './wavenet_params.json'
SAVE_EVERY = None

'''
This generation file takes as input a folder of seed motions and a directory of saved model. It will save generated results/ground truth under folder /logdir/skeleton_generate/model_name/seed_name
'''

def get_arguments():
    def _str_to_bool(s):
        """Convert string to bool (in argparse context)."""
        if s.lower() not in ['true', 'false']:
            raise ValueError('Argument needs to be a '
                             'boolean, got {}'.format(s))
        return {'true': True, 'false': False}[s.lower()]

    def _ensure_positive_float(f):
        """Ensure argument is a positive float."""
        if float(f) < 0:
            raise argparse.ArgumentTypeError('Argument must be greater than zero')
        return float(f)

    parser = argparse.ArgumentParser(description='WaveNet motion generation script')
    parser.add_argument(
        'checkpoint', type=str, help='Which model checkpoint to generate from')
    parser.add_argument(
        '--samples',
        type=int,
        default=SAMPLES,
        help='How many frames of motion samples to generate')
    parser.add_argument(
        '--logdir',
        type=str,
        default=LOGDIR,
        help='Directory in which to store the logging '
        'information for TensorBoard.')
    parser.add_argument(
        '--window',
        type=int,
        default=WINDOW,
        help='The number of past samples to take into '
        'account at each step')
    parser.add_argument(
        '--wavenet_params',
        type=str,
        default=WAVENET_PARAMS,
        help='JSON file with the network parameters')
    parser.add_argument(
        '--skeleton_out_path',
        type=str,
        default=None,
        help='Path to output skeleton file')
    parser.add_argument(
        '--fast_generation',
        type=_str_to_bool,
        default=True,
        help='Use fast generation')
    parser.add_argument(
        '--motion_seed',
        type=str,
        default=None,
        help='The skeleton file to start generation from')
    return parser.parse_args()


#todo create_seed of skeleton
def create_seed(filename,
                window_size=WINDOW):
    skeleton = np.loadtxt(filename, delimiter=',')
    cut_index = min(skeleton.shape[0], window_size)
    return skeleton, cut_index
    #return tf.constant(skeleton[:cut_index, 3:])

def main():
    args = get_arguments()
    started_datestring = "{0:%Y-%m-%dT%H-%M-%S}".format(datetime.now())
    #logdir is where logging file is saved. different from where generated mat is saved.
    logdir = os.path.join(args.logdir, 'generate', started_datestring)
    with open(args.wavenet_params, 'r') as config_file:
        wavenet_params = json.load(config_file)

    sess = tf.Session()

    net = WaveNetModel(
        batch_size=1,
        dilations=wavenet_params['dilations'],
        filter_width=wavenet_params['filter_width'],
        residual_channels=wavenet_params['residual_channels'],
        dilation_channels=wavenet_params['dilation_channels'],
        skip_channels=wavenet_params['skip_channels'],
        use_biases=wavenet_params['use_biases'],
        scalar_input=wavenet_params['scalar_input'],
        initial_filter_width=wavenet_params['initial_filter_width'])

    samples = tf.placeholder(dtype=tf.float32)
    #todo: Q: how does samples represent T x 42 data? does predict_proba_incremental memorize? A: samples can store multiple frames. T x 42 dim
    if args.fast_generation:
        next_sample = net.predict_proba_incremental(samples)
    else:
        next_sample = net.predict_proba(samples)

    if args.fast_generation:
        sess.run(tf.initialize_all_variables())
        sess.run(net.init_ops)

    variables_to_restore = {
        var.name[:-2]: var for var in tf.all_variables()
        if not ('state_buffer' in var.name or 'pointer' in var.name)}
    saver = tf.train.Saver(variables_to_restore)

    print('Restoring model from {}'.format(args.checkpoint))
    saver.restore(sess, args.checkpoint)

    #decode = mu_law_decode(samples, wavenet_params['quantization_channels'])

    #quantization_channels = wavenet_params['quantization_channels']

    if args.motion_seed:
        pass
    else:
        raise ValueError('motion seed not specified!')

    for sub in os.listdir(args.motion_seed):
        gt, cut_index = create_seed(os.path.join(args.motion_seed, sub), args.window)
        if np.isnan(np.sum(gt)):
            print('nan detected')
            continue
        seed = tf.constant(gt[:cut_index,3:])
        #seed: T x 42 tensor
        #tolist() converts a tf tensor to a list
        motion = sess.run(seed).tolist()
        #motion[i]: ith frame, list of 42 features
        if args.fast_generation and args.motion_seed:
            outputs = [next_sample]
            outputs.extend(net.push_ops)

            print('Priming generation...')
            for i, x in enumerate(motion[-args.window: -1]):
                if i % 10 == 0:
                    print('Priming sample {}'.format(i))
                sess.run(outputs, feed_dict={samples: np.reshape(x, (1,42))})
            print('Done.')

        last_sample_timestamp = datetime.now()
        for step in range(args.samples):
            if args.fast_generation:
                outputs = [next_sample]
                outputs.extend(net.push_ops)
                window = motion[-1]
            else:
                if len(motion) > args.window:
                    window = motion[-args.window:]
                else:
                    window = motion
                outputs = [next_sample]

            # Run the WaveNet to predict the next sample.
            prediction = sess.run(outputs, feed_dict={samples: np.reshape(window,(1,42))})[0]
            #prediction = sess.run(outputs, feed_dict={samples: window})[0]

            motion.append(prediction)
            # Show progress only once per second.
            current_sample_timestamp = datetime.now()
            time_since_print = current_sample_timestamp - last_sample_timestamp
            if time_since_print.total_seconds() > 1.:
                print('Sample {:3<d}/{:3<d}'.format(step + 1, args.samples),
                      end='\r')
                last_sample_timestamp = current_sample_timestamp

        print()

        #save result in .mat file
        if args.skeleton_out_path:
            #outdir = os.path.join('logdir','skeleton_generate', os.path.basename(os.path.dirname(args.checkpoint)) + os.path.basename(args.checkpoint)+'window'+str(args.window)+'sample'+str(args.samples))
            outdir = os.path.join(args.skeleton_out_path, os.path.basename(os.path.dirname(args.checkpoint)))
            if not os.path.exists(outdir):
                os.makedirs(outdir)
            filedir = os.path.join(outdir, str(sub)+'.mat')
            #filedir = os.path.join(outdir, (sub+args.skeleton_out_path))
            sio.savemat(filedir, {'wavenet_predict': motion, 'gt': gt})
            #out = sess.run(decode, feed_dict={samples: motion})
            #todo: write skeleton writer
            #write_skeleton(motion, args.wav_out_path)
            print(len(motion))
            print('generated filedir:{0}'.format(filedir))
        print('Finished generating. The result can be viewed in Matlab.')


if __name__ == '__main__':
    main()
