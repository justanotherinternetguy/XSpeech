# XSpeech: A Novel Deep Learning Approach to Classifying Stutters

https://huggingface.co/coqui/XTTS-v2
whisper + annotations --> tts --> e2e train

/home/alien/Git/DATA/ml-stuttering-events-dataset/CLIP_DIR/IStutterSoWhat/1/IStutterSoWhat_1_100.wav

/home/alien/Git/DATA/ml-stuttering-events-dataset/CLIP_DIR/IStutterSoWhat/1/IStutterSoWhat_1_100.wav  and where I cut, cut, cut, cut, cut, cut, cut, cut, cut.


- we have transcriptions
TTS that 

Tensorflow: create an end to end audio deep learning model which has:
The input data is a 2D numpy array called X. X is 16271 elements long, with each element being a 1D vector that is 132300 long. Each 1D vector represents audio signals from librosa.load()
The output data is a 2D list called y. y is also 16271, but each element VARIES in length.

First, normalize the lengths of every element in y by padding the end of each element so that all elements are the same length. They should all be the equal to the length of the longest element.

Then, do a 85%-15% split on the data. Create a tensorflow model to train and test the data.