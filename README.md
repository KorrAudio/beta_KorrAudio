KorrAudio
======================
Simple, lightweight program in Python for obtaining information about an audio file.

Who to install ?
---------------------
There are two steps to running the program:

1. Install the necessary libraries
```
pip install librosa soundfile numpy PySimpleGUI matplotlib tinytag
```
2. Download and run the file KorrAudio.py
```
python3 KorrAudio.py
```

Features
---------------------
Audio analysis:
```
    File information:
    Name, Format, Last Modified, File Hash
    
    Metadata:
    Artist, Title, Album, Year, Genre

    Audio analysis:
    File Duration, Sample Rate, Sampling Frequency, Number of Channels, Maximum Amplitude, 
    Average Amplitude, Minimum Frequency, Maximum Frequency, Tempo, Average Loudness, Chroma
```
Graphical display:
```
    Waveform
    Spectrogram
    Frequency Spectrum
    Spectral Envelope
```

Screenshot
---------------------
![scr](https://github.com/KorrAudio/beta_KorrAudio/assets/139574456/a1a9fce7-2623-4780-bf52-400c5b2cd515)

Libraries Informations
---------------------
```
librosa: This library is used for loading and analyzing audio files.
soundfile: This library is used to read and write audio files. 
numpy: This library is used to perform numerical calculations on audio data. 
PySimpleGUI: This library is used to create the graphical interface. 
matplotlib: This library is used to display audio graphics.
tinytag: This library is used for reading music meta data 
```
<p align="center"><sup>* The program assumes the availability of necessary dependencies such as librosa, matplotlib, numpy, PySimpleGUI, soundfile, and tinytag.</sup></p>
<p align="center">KorrAudio - GPLv3 license</p>
