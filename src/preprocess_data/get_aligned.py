import os
import sys
import pickle
import argparse
import shutil


parser = argparse.ArgumentParser()
parser.add_argument('--data_root', type=str, default='../../data')
parser.add_argument('--out_root', type=str, default='../../data/smarts_frames_aligned_by_video')
args = parser.parse_args()

def make_dir(d):
    if not os.path.exists(d):
        os.makedirs(d)

if __name__ == "__main__":
    make_dir(args.out_root)

    # frames
    frames_dir = os.path.join(args.data_root, 'smarts_frames')
    actions = os.listdir(frames_dir)

    for act in actions:
        frames = os.listdir(os.path.join(frames_dir, act))

        for f in frames:
            # bbox
            frame_basename, ext = os.path.splitext(f)

            # make dir by video
            identity, no_frame = frame_basename.split('_')
            no_frame = no_frame.zfill(3)
            make_dir(os.path.join(args.out_root, identity, act)) 

            frame_path = os.path.join(frames_dir, act, f)
            shutil.copy(frame_path, os.path.join(args.out_root, identity, act, '{}.png'.format(no_frame)))

