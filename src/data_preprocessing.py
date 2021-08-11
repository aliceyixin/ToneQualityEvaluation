import librosa
import numpy as np
import pandas as pd
import scipy
import warnings
warnings.filterwarnings('ignore')


input_length = 143999  # =48000*3
sr = 48000
n_mels = 80


def pre_process_audio_melspec(audio, sr):
    mel_spec = librosa.feature.melspectrogram(y=audio, sr=sr, n_fft=2048, hop_length=480, n_mels=n_mels)
    mel_db = (librosa.power_to_db(mel_spec, ref=np.max) + 40) / 40
    return mel_db.T


def load_audio_file(file_path, input_length=input_length):
    try:
        data, _ = librosa.load(str(file_path), sr=sr, mono=True)
    except ZeroDivisionError:
        data = []

    if len(data) > input_length:

        max_offset = len(data) - input_length

        offset = np.random.randint(max_offset)

        data = data[offset: (input_length + offset)]

    else:
        if input_length > len(data):
            max_offset = input_length - len(data)

            offset = np.random.randint(max_offset)
        else:
            offset = 0

        data = np.pad(data, (offset, input_length - len(data) - offset), "constant")

    return data

def Residual_estimation(y, p):
    a = librosa.lpc(y, p)
    b = np.hstack([[0], -1 * a[1:]])
    y_lp = scipy.signal.lfilter(b, [1], y)
    # LP residual
    y_r = y - y_lp
    # HE envelope
    # r_a = scipy.signal.hilbert(y_r)
    # h_e = np.sqrt(abs(y_r)**2 + abs(r_a)**2)
    # r_a = scipy.signal.hilbert(y_lp)
    # h_e = np.sqrt(abs(y_lp)**2 + abs(r_a)**2)
    return y_r


def save_features(filename):
    file_path = Path(args.wav_path+filename)
    audio = load_audio_file(file_path)

    # 1. pre-emphasis
    audio = scipy.signal.lfilter([1.0, -0.97], 1, audio)

    # 2. calculate features:
    # change pitch
    # Shift up by a major third
    pitch_up = librosa.effects.pitch_shift(audio, sr, n_steps=4)
    # Shift down by a major third
    pitch_down = librosa.effects.pitch_shift(audio, sr, n_steps=-4)
    # change loudness
    loudness_up = audio*1.5
    loudness_down = audio*0.5

    # 3. source estimation
    order = 16
    source = Residual_estimation(audio, p=order)
    source_pu = Residual_estimation(pitch_up, p=order)
    source_pd = Residual_estimation(pitch_down, p=order)
    source_lu = Residual_estimation(loudness_up, p=order)
    source_ld = Residual_estimation(loudness_down, p=order)

    # 4. calculate RPMelspec OR Melspec
    # RPMelspec
    melspec = pre_process_audio_melspec(source, sr)
    melspec_pu = pre_process_audio_melspec(source_pu, sr)
    melspec_pd = pre_process_audio_melspec(source_pd, sr)
    melspec_lu = pre_process_audio_melspec(source_lu, sr)
    melspec_ld = pre_process_audio_melspec(source_ld, sr)

    # Melspec
    # melspec = pre_process_audio_melspec(audio, sr)
    # melspec_pu = pre_process_audio_melspec(pitch_up, sr)
    # melspec_pd = pre_process_audio_melspec(pitch_down, sr)
    # melspec_lu = pre_process_audio_melspec(loudness_up, sr)
    # melspec_ld = pre_process_audio_melspec(loudness_down, sr)

    # 5. save featuremap
    melspec_path = Path(args.feature_path+filename[:-4] + ".npy")
    melspec_pu_path = Path(args.feature_path+filename[:-4] + "_pu.npy")
    melspec_pd_path = Path(args.feature_path+filename[:-4] + "_pd.npy")
    melspec_lu_path = Path(args.feature_path+filename[:-4] + "_lu.npy")
    melspec_ld_path = Path(args.feature_path+filename[:-4] + "_ld.npy")

    np.save(melspec_path, melspec)
    np.save(melspec_pu_path, melspec_pu)
    np.save(melspec_pd_path, melspec_pd)
    np.save(melspec_lu_path, melspec_lu)
    np.save(melspec_ld_path, melspec_ld)
    return True


if __name__ == "__main__":
    from tqdm import tqdm
    # from glob import glob
    from multiprocessing import Pool
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser()
    parser.add_argument("--wav_path", default="../saxData/audio/")
    parser.add_argument("--feature_path", default="../saxData/RPMelspec/")
    parser.add_argument("--train_path", default="../saxData/evaluation_setup/sax_train.csv")
    parser.add_argument("--val_path", default="../saxData/evaluation_setup/sax_val.csv")
    parser.add_argument("--test_path", default="../saxData/evaluation_setup/sax_test.csv")

    args = parser.parse_args()

    base_path = Path(args.wav_path)
    train_csv = pd.read_csv(args.train_path, delimiter=',')
    val_csv = pd.read_csv(args.val_path, delimiter=',')
    test_csv = pd.read_csv(args.test_path, delimiter=',')

    files_csv = pd.concat([train_csv, val_csv, test_csv])
    files = files_csv.name.tolist()

    p = Pool(8)
    for i, _ in tqdm(enumerate(p.imap(save_features, files)), total=len(files)):
        if i % 1000 == 0:
            print(i)
