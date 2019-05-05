# Music Genre Classification using Python

We need to build a classifier to classify songs into different genres. Given an audio file our task is to sort them according to the music genre into different folders such as jazz, classical, country, pop, rock, and metal.

# Audio Processing with Python
Sound is represented in the form of an audio signal having parameters such as frequency, bandwidth, decibel etc. A typical audio signal can be expressed as a function of Amplitude and Time.
These sounds are available in many formats which makes it possible for the computer to read and analyse them. Some examples are:

1. mp3 format
2. WMA (Windows Media Audio) format
3. wav (Waveform Audio File) format

# Audio Libraries in Python
Python has some great libraries for audio processing like Librosa and PyAudio.There are also built-in modules for some basic audio functionalities.

We will mainly use two libraries for audio acquisition and playback:

1. Librosa
2. IPython.display.Audio

# Installation

pip install librosa
or
conda install -c conda-forge librosa

We will be using Librosa library for our purpose.

# Dataset
We have used the famous GITZAN dataset for our case study. This dataset was used for the well-known paper in genre classification “ Musical genre classification of audio signals “ by G. Tzanetakis and P. Cook in IEEE Transactions on Audio and Speech Processing 2002.

The dataset consists of 1000 audio tracks each 30 seconds long. It contains 10 genres namely, blues, classical, country, disco, hiphop, jazz, reggae, rock, metal and pop. Each genre consists of 100 sound clips.

# Spectrogram
A spectrogram is a visual representation of the spectrum of frequencies of sound or other signals as they vary with time. Spectrograms are sometimes called sonographs, voiceprints, or voicegrams. You can either use the spectrogram images directly for classification or can extract the features and use the classification models on them.

# Feature Extraction
We then need to extract meaningful features from audio files. To classify our audio clips, we will choose 5 features, i.e. Mel-Frequency Cepstral Coefficients, Spectral Centroid, Zero Crossing Rate, Chroma Frequencies, Spectral Roll-off. All the features are then appended into a .csv file so that classification algorithms can be used.

1. Zero Crossing Rate

The zero crossing rate is the rate of sign-changes along a signal, i.e., the rate at which the signal changes from positive to negative or back. This feature has been used heavily in both speech recognition and music information retrieval. It usually has higher values for highly percussive sounds like those in metal and rock.

2. Spectral Centroid

It indicates where the ”centre of mass” for a sound is located and is calculated as the weighted mean of the frequencies present in the sound. Consider two songs, one from a blues genre and the other belonging to metal. Now as compared to the blues genre song which is the same throughout its length, the metal song has more frequencies towards the end. So spectral centroid for blues song will lie somewhere near the middle of its spectrum while that for a metal song would be towards its end.

3. Spectral Rolloff

It is a measure of the shape of the signal. It represents the frequency below which a specified percentage of the total spectral energy, e.g. 85%, lies.

4. Mel-Frequency Cepstral Coefficients

The Mel frequency cepstral coefficients (MFCCs) of a signal are a small set of features (usually about 10–20) which concisely describe the overall shape of a spectral envelope. It models the characteristics of the human voice.

5. Chroma Frequencies
Chroma features are an interesting and powerful representation for music audio in which the entire spectrum is projected onto 12 bins representing the 12 distinct semitones (or chroma) of the musical octave.

We have in-built Librosa library API's to extract each of the features directly.

# Preprocessing the Data
Before training the classification model, we have to transform raw data from audio samples into more meaningful representations. The audio clips need to be converted from .au format to .wav format to make it compatible with python’s wave module for reading audio files. Python's Librosa supports only mp3, wav (Waveform Audio File) format and WMA (Windows Media Audio) formats only. So, I used the open source SoX module for the conversion. 

sox input.au output.wav

The different models I have built are as follows:

1. Simple Feed forward neural network with all the features combined
2. LSTM model by taking only mfcc features which are 20 in number.
3. CNN using Spectrogram images directly.
4. A hybrid model containing lstm(for mfcc features) + feed forward neural network(other features except mfcc)

I have saved the model's architecture and weights for future reference as well.
