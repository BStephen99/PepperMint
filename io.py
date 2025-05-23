import os
from PIL import Image
from scipy.io import wavfile
#from core.util import Logger
import numpy as np
import python_speech_features
import csv
import time
import json

import os
import torch


class Logger():
    def __init__(self, targetFile, separator=';'):
        self.targetFile = targetFile
        self.separator = separator

    def writeHeaders(self, headers):
        with open(self.targetFile, 'a') as fh:
            for aHeader in headers:
                fh.write(aHeader + self.separator)
            fh.write('\n')

    def writeDataLog(self, dataArray):
        with open(self.targetFile, 'a') as fh:
            for dataItem in dataArray:
                fh.write(str(dataItem) + self.separator)
            fh.write('\n')

def configure_backbone(backbone, size, pretrained_arg=True, num_classes_arg=2):
    return backbone(pretrained=pretrained_arg, rgb_stack_size=size,
                    num_classes=num_classes_arg)

def configure_backbone_forward_phase(backbone, pretrained_weights_path, size, pretrained_arg=True, num_classes_arg=2):
    return backbone(pretrained_weights_path, rgb_stack_size=size,
                    num_classes=num_classes_arg)

def load_train_video_set():
    files = os.listdir('.../AVA/csv/train')
    videos = [f[:-18] for f in files]
    videos.sort()
    return videos

def load_val_video_set():
    files = os.listdir('.../AVA/csv/val')
    videos = [f[:-18] for f in files]
    videos.sort()
    return videos

def preprocessRGBData(rgb_data):
    rgb_data = rgb_data.astype('float32')
    rgb_data = rgb_data/255.0
    rgb_data = rgb_data - np.asarray((0.485, 0.456, 0.406))

    return rgb_data


def _pil_loader(path, target_size):
    try:
        with open(path, 'rb') as f:
            img = Image.open(f)
            img = img.resize(target_size)
            return img.convert('RGB')
    except OSError as e:
        return Image.new('RGB', target_size)


def set_up_log_and_ws_out(models_out, opt_config, experiment_name, headers=None):
    target_logs = os.path.join(models_out, experiment_name + '_logs.csv')
    target_models = os.path.join(models_out, experiment_name)
    print('target_models', target_models)
    if not os.path.isdir(target_models):
        os.makedirs(target_models)
    log = Logger(target_logs, ';')

    if headers is None:
        log.writeHeaders(['epoch', 'train_loss', 'train_auc', 'train_map',
                          'val_loss', 'val_auc', 'val_map'])
    else:
        log.writeHeaders(headers)

    # Dump cfg to json
    dump_cfg = opt_config.copy()
    for key, value in dump_cfg.items():
        if callable(value):
            try:
                dump_cfg[key]=value.__name__
            except:
                dump_cfg[key]='CrossEntropyLoss'
    json_cfg = os.path.join(models_out, experiment_name+'_cfg.json')
    with open(json_cfg, 'w') as json_file:
      json.dump(dump_cfg, json_file)

    models_out = os.path.join(models_out, experiment_name)
    return log, models_out


def csv_to_list(csv_path):
    as_list = None
    with open(csv_path, 'r') as f:
        reader = csv.reader(f)
        as_list = list(reader)
    return as_list


def _generate_mel_spectrogram(audio_clip, sample_rate):
    mfcc = zip(*python_speech_features.mfcc(audio_clip, sample_rate))
    audio_features = np.stack([np.array(i) for i in mfcc])
    audio_features = np.expand_dims(audio_features, axis=0)
    return audio_features


def _fit_audio_clip(audio_clip, sample_rate, video_clip_lenght):
    target_audio_length = (1.0/27.0)*sample_rate*video_clip_lenght
    pad_required = int((target_audio_length-len(audio_clip))/2)
    if pad_required > 0:
        audio_clip = np.pad(audio_clip, pad_width=(pad_required, pad_required),
                            mode='reflect')
    if pad_required < 0:
        audio_clip = audio_clip[-1*pad_required:pad_required]

    return audio_clip


def load_av_clip_from_metadata(clip_meta_data, frames_source, audio_source,
                               audio_offset, target_size):
    ts_sequence = [str(meta[1]) for meta in clip_meta_data]

    min_ts = float(clip_meta_data[0][1])
    max_ts = float(clip_meta_data[-1][1])
    entity_id = clip_meta_data[0][0]

    selected_frames = [os.path.join(frames_source, entity_id, ts+'.jpg') for ts in ts_sequence]
    video_data = [_pil_loader(sf, target_size) for sf in selected_frames]
    audio_file = os.path.join(audio_source, entity_id+'.wav')

    try:
        sample_rate, audio_data = wavfile.read(audio_file)
    except:
        sample_rate, audio_data = 16000,  np.zeros((16000*10))

    audio_start = int((min_ts-audio_offset)*sample_rate)
    audio_end = int((max_ts-audio_offset)*sample_rate)
    audio_clip = audio_data[audio_start:audio_end]

    if len(audio_clip) < sample_rate*(2/25):
        audio_clip = np.zeros((int(sample_rate*(len(clip_meta_data)/25))))

    audio_clip = _fit_audio_clip(audio_clip, sample_rate, len(selected_frames))
    audio_features = _generate_mel_spectrogram(audio_clip, sample_rate)

    return video_data, audio_features
