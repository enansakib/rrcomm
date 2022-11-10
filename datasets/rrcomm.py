import torch.utils.data as data

import os
import sys
import random
import numpy as np
import cv2


def find_classes(dir):
    classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
    classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    return classes, class_to_idx


def make_dataset(root, source):

    if not os.path.exists(source):
        print("Setting file %s for rrcomm dataset doesn't exist." % (source))
        sys.exit()
    else:
        clips = []
        with open(source) as split_f:
            data = split_f.readlines()
            for line in data:
                line_info = line.split()
                clip_path = os.path.join(root, line_info[0])
                duration = int(line_info[1])
                target = int(line_info[2])
                item = (clip_path, duration, target)
                clips.append(item)
    return clips


def ReadSegmentRGB(path, offsets, new_height, new_width, new_length, is_color, name_pattern, duration, input_skip):
    if is_color:
        cv_read_flag = cv2.IMREAD_COLOR
    else:
        cv_read_flag = cv2.IMREAD_GRAYSCALE
    interpolation = cv2.INTER_LINEAR

    sampled_list = []
    for offset_id in range(len(offsets)):
        offset = offsets[offset_id]
        for length_id in range(1, new_length+1):
            
            if input_skip==0: 
                loaded_frame_index = length_id + offset
            elif input_skip==1:
                if length_id % 2 != 0:
                    loaded_frame_index = length_id + offset
                else: continue

            moded_loaded_frame_index = loaded_frame_index % (duration + 1)
            if moded_loaded_frame_index == 0:
                moded_loaded_frame_index = (duration + 1)
            frame_name = name_pattern % (moded_loaded_frame_index)
            frame_path = path + "/" + frame_name
            cv_img_origin = cv2.imread(frame_path, cv_read_flag)
            if cv_img_origin is None:
               print("Could not load file %s" % (frame_path))
               sys.exit()
            if new_width > 0 and new_height > 0:
                cv_img = cv2.resize(cv_img_origin, (new_width, new_height), interpolation)
            else:
                cv_img = cv_img_origin
            cv_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
            sampled_list.append(cv_img)
    clip_input = np.concatenate(sampled_list, axis=2)
    return clip_input


class rrcomm(data.Dataset):

    def __init__(self,
                 root,
                 source,
                 phase,
                 name_pattern=None,
                 is_color=True,
                 num_segments=1,
                 new_length=1,
                 new_width=0,
                 new_height=0,
                 transform=None,
                 target_transform=None,
                 video_transform=None,
                 ensemble_training = False,
                 input_skip=0):

        classes, class_to_idx = find_classes(root)
        clips = make_dataset(root, source)

        if len(clips) == 0:
            raise(RuntimeError("Found no clips in " + root))

        self.root = root
        self.source = source
        self.phase = phase

        self.classes = classes
        self.class_to_idx = class_to_idx
        self.clips = clips
        self.ensemble_training = ensemble_training

        if name_pattern:
            self.name_pattern = name_pattern
        else:
            self.name_pattern = "img_%05d.jpg"

        self.is_color = is_color
        self.num_segments = num_segments
        self.new_length = new_length
        self.length_for_offset = 64
        self.new_width = new_width
        self.new_height = new_height
        self.input_skip = input_skip

        self.transform = transform
        self.target_transform = target_transform
        self.video_transform = video_transform

    def __getitem__(self, index):
        path, duration, target = self.clips[index]
        duration = duration - 1
        average_duration = int(duration / self.num_segments)
        average_part_length = int(np.floor((duration-self.length_for_offset) / self.num_segments))
        offsets = []
        for seg_id in range(self.num_segments):
            if self.phase == "train":
                if average_duration >= self.length_for_offset:
                    offset = random.randint(0, average_duration - self.length_for_offset)
                    offsets.append(offset + seg_id * average_duration)
                elif duration >= self.length_for_offset:
                    offset = random.randint(0, average_part_length)
                    offsets.append(seg_id*average_part_length + offset)
                else:
                    increase = random.randint(0, duration)
                    offsets.append(0 + seg_id * increase)
            elif self.phase == "val":
                if average_duration >= self.length_for_offset:
                    offsets.append(int((average_duration - self.length_for_offset + 1)/2 + seg_id * average_duration))
                elif duration >= self.length_for_offset:
                    offsets.append(int((seg_id*average_part_length + (seg_id + 1) * average_part_length)/2))
                else:
                    increase = int(duration / self.num_segments)
                    offsets.append(0 + seg_id * increase)
            else:
                print("Only train and val are supported.")
        

        clip_input = ReadSegmentRGB(path,
                                    offsets,
                                    self.new_height,
                                    self.new_width,
                                    self.length_for_offset,
                                    self.is_color,
                                    self.name_pattern,
                                    duration,
                                    self.input_skip
                                    )

        if self.transform is not None:
            clip_input = self.transform(clip_input)
        if self.target_transform is not None:
            target = self.target_transform(target)
        if self.video_transform is not None:
            clip_input = self.video_transform(clip_input)   
        return clip_input, target
                
    def __len__(self):
        return len(self.clips)