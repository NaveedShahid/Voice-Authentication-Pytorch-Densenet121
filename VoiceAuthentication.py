import random

import torch
import librosa
import numpy as np
import os
import app_setup

N_MELS = 128
NUM_VOTES = 50
VOTING_THRESHOLD = 45   # 90%


class VoiceAuthentication:
    def __init__(self, model_path, print_info=False):
        self.print_info = print_info
        self.model_path = model_path
        self.model = None
        self.device = None
        self.__load_model()

    def authenticate(self, wav_1, wav_2):
        # Load wav files
        wav_1_samples, wav_1_sample_rate = librosa.core.load(wav_1)
        wav_2_samples, wav_2_sample_rate = librosa.core.load(wav_2)
        # Check if wav files are empty
        null_check = [self.__wav_null_check(wav) for wav in [wav_1_samples, wav_2_samples]]
        if any(null_check):
            for i, null in enumerate(null_check): print("WAV {} NULL".format(i+1))
            is_same_user, votes, vote_ratio = False, -1, -1
            print("AUTH: {}, VOTES: {} ({})".format(is_same_user, votes, vote_ratio))
            return is_same_user, votes, vote_ratio
        # Wav to spectrogram
        spectrograms = [self.__wav_to_spectrogram(wav, extend_copy=True) for wav in [wav_1_samples, wav_2_samples]]
        # create input tensor
        input_imgs_list = [[], []]
        # Majority voting
        for imgs, spectrogram in zip(input_imgs_list, spectrograms):
            for _ in range(NUM_VOTES):   # populate with random slices
                imgs.append(self.__get_random_spectrogram_slice(spectrogram))
        # input tensor
        input_imgs = torch.tensor(input_imgs_list)
        # predict
        outputs, preds = self.__model_predict(input_imgs)
        if self.print_info: print(outputs); print(preds)
        # aggregate votes
        votes_results = self.__preds_aggregator(preds)
        return votes_results
    
    def register(self, wav_1):  
        # Load wav files
        name = str(wav_1).split("\\")[-1]
        name = name.replace(".wav",".npy")
        out_file = os.path.join(app_setup.SAVED_AUDIO_FOLDER, name) 
        if os.path.exists(out_file)==False:
            wav_samples = self.__load_audio_check_null(wav_1)
            # Wav to spectrogram
            if wav_samples == "null":
                return False
            spectrogram = self.__wav_to_spectrogram(wav_samples, extend_copy=True)
            np.save(out_file, spectrogram)
            return True
        else:
            print("File already present in the directory")
            return False

    def recognize(self, wav_1):
        max_acc = VOTING_THRESHOLD*2
        name = "Unknown"
        recognized = False
        results =[]
        wav_1_samples = self.__load_audio_check_null(wav_1)
        if wav_1_samples == "null":
            return False
        spectrogram_1 = self.__wav_to_spectrogram(wav_1_samples, extend_copy=True)

        for user_file in os.listdir(app_setup.SAVED_AUDIO_FOLDER):
            # Majority voting
            spectrogram_2 = np.load(os.path.join(app_setup.SAVED_AUDIO_FOLDER, user_file))
            result = self.__compare_spectrograms(spectrogram_1, spectrogram_2)
            result_1 = list(result)
            result_1.append(user_file)
            results.append(result_1)
            if result_1[0]==True and max_acc < result_1[2]:
                max_vote = result_1[1]
                max_acc = result_1[2]
                recognized = result_1[0]
                name = user_file

        return recognized, max_vote, max_acc, name, results




    ### Private methods

    def __load_model(self):
        model = torch.jit.load(self.model_path)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # device = torch.device("cpu")
        print("DEVICE: {}".format(device))
        print("MODEL PATH: {}".format(self.model_path))
        if self.print_info: print(model.code)
        # set object attributes
        self.model = model.to(device)
        self.device = device
    
    def __load_audio_check_null(self, wav):
        wav_samples, wav_sample_rate = librosa.core.load(wav)
        # Check if wav files are empty
        null_check = [self.__wav_null_check(wav) for wav in [wav_samples]]
        if any(null_check):
            for i, null in enumerate(null_check): print("WAV {} NULL".format(i+1))
            print("Cannot register a null audio file")
            return "null"
        else:
            return wav_samples
            
    def __compare_spectrograms(self, wav_1_samples, wav_2_samples):
        input_imgs_list = [[], []]
        for imgs, spectrogram in zip(input_imgs_list, [wav_1_samples, wav_2_samples]):
            for _ in range(NUM_VOTES):   # populate with random slices
                imgs.append(self.__get_random_spectrogram_slice(spectrogram))
        # input tensor
        input_imgs = torch.tensor(input_imgs_list)
        # predict
        outputs, preds = self.__model_predict(input_imgs)
        if self.print_info: print(outputs); print(preds)
        # aggregate votes
        votes_results = self.__preds_aggregator(preds)  
        return votes_results
            
            
    def __wav_null_check(self, samples):
        is_null = np.sum(np.abs(samples[::10])) == 0.0
        return is_null

    def __wav_to_spectrogram(self, samples, extend_copy=False):
        if extend_copy: samples = np.append(samples, samples)   # add copy to increase sample size
        S = librosa.feature.melspectrogram(samples, n_mels=N_MELS)
        S_dB = librosa.power_to_db(S, ref=1.0)
        spectrogram = S_dB
        return spectrogram

    def __get_random_spectrogram_slice(self, spectrogram, depth=3, sliding_ratio=2):
        ### Combine multiple sliding greyscale img slices into an n-depth image
        height = spectrogram.shape[0]
        slide_step = height // sliding_ratio
        img_slice = np.zeros((depth, height, height))  # initialize empty img (pytorch style)
        # Get random start idx
        slice_start = random.randint(0, spectrogram.shape[1] - (slide_step * (depth + 1)) - 1)
        for i in range(depth):
            img_slice[i, :, :] = spectrogram[:, slice_start:slice_start + height]  # get slice (pytorch style)
            slice_start += slide_step  # slide
        img_slice = img_slice.astype("float32")
        img_slice = img_slice / np.amax(np.absolute(img_slice))  # normalize to range [-1, 1]
        return img_slice

    def __model_predict(self, input_imgs):
        # prep inputs
        inputs = [img.to(self.device) for img in input_imgs]
        # predict
        with torch.no_grad():
            self.model.eval()   # eval mode
            outputs = self.model(inputs)
            _, preds = torch.max(outputs, 1)
        return outputs, preds

    def __preds_aggregator(self, preds):
        preds = preds.flatten().tolist()
        votes = sum(preds)
        is_same_user = votes >= VOTING_THRESHOLD
        vote_ratio = votes / NUM_VOTES * 100.0
        print("AUTH: {}, VOTES: {} ({})".format(is_same_user, votes, vote_ratio))
        return is_same_user, votes, vote_ratio








