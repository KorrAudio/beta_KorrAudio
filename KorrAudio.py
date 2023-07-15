import datetime
import hashlib
import os

import librosa
import matplotlib.pyplot as plt
import numpy as np
import PySimpleGUI as sg
import soundfile as sf
from tinytag import TinyTag
from librosa.feature import chroma_stft

# Constants
SUPPORTED_FORMATS = ["mp3", "wav", "ogg", "flac", "aiff"]

# Audio analysis functions
def analyze_audio(file_path):
    # Get the audio file format
    file_format = os.path.splitext(file_path)[1][1:].strip().lower()

    if file_format not in SUPPORTED_FORMATS:
        return "The selected file is not a valid audio file."

    # Get file information
    file_info = os.stat(file_path)
    modification_time = datetime.datetime.fromtimestamp(file_info.st_mtime)
    file_hash = calculate_file_hash(file_path)

    # Load the audio file
    audio, sample_rate = librosa.load(file_path)

    # Calculate the sample rate
    bitrate = sf.info(file_path).samplerate

    # Calculate the audio file duration
    duration = len(audio) / sample_rate

    # Calculate the number of audio channels
    num_channels = audio.shape[1] if len(audio.shape) > 1 else 1

    # Calculate the maximum and average amplitude
    max_amplitude = np.max(np.abs(audio))
    mean_amplitude = np.mean(np.abs(audio))

    # Calculate the minimum and maximum frequencies
    min_frequency = 0
    max_frequency = sample_rate / 2

    # Extract metadata
    audio_file = TinyTag.get(file_path)
    artist = audio_file.artist or "Unknown"
    title = audio_file.title or "Unknown"
    album = audio_file.album or "Unknown"
    year = audio_file.year or "Unknown"
    genre = audio_file.genre or "Unknown"

    # Calculate the tempo
    tempo, beat_frames = librosa.beat.beat_track(y=audio, sr=sample_rate)

    # Calculate Chroma Features
    chroma = chroma_stft(y=audio, sr=sample_rate)
    mean_chroma = np.mean(chroma, axis=1)

    # Create the analysis results text
    file_info_text1 = f"File Name: {os.path.basename(file_path)}\n" \
                     f"Audio File Format: {file_format}\n" \
                     f"Last Modified: {modification_time}\n" \
                     f"File Hash: {file_hash}\n"

    file_info_text2 = f"\n" \
                     f"File Duration: {duration:.2f} seconds\n" \
                     f"Sample Rate: {bitrate} Hz\n" \
                     f"Sampling Frequency: {sample_rate} Hz\n" \
                     f"Number of Channels: {num_channels}\n" \
                     f"Maximum Amplitude: {max_amplitude:.2f} (scaled value)\n" \
                     f"Average Amplitude: {mean_amplitude:.2f} (scaled value)\n" \
                     f"Minimum Frequency: {min_frequency} Hz\n" \
                     f"Maximum Frequency: {max_frequency:.2f} Hz\n"

    tempo_text = f"Tempo: {tempo:.2f} BPM\n"
                    
    chroma_text = f"\nChroma Features:\n"
    for note, value in zip(["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"], mean_chroma):
        chroma_text += f"{note}: {value:.3f}\n"

    metadata_text = f"\n" \
                    f"Artist: {artist}\n" \
                    f"Title: {title}\n" \
                    f"Album: {album}\n" \
                    f"Year: {year}\n" \
                    f"Genre: {genre}\n"

    results = file_info_text1 + metadata_text + file_info_text2 + tempo_text + chroma_text

    return results

def calculate_file_hash(file_path):
    with open(file_path, 'rb') as f:
        hasher = hashlib.md5()

        for chunk in iter(lambda: f.read(4096), b''):
            hasher.update(chunk)

        return hasher.hexdigest()

def get_frequency_spectrum(audio, sample_rate):
    frequencies = np.fft.fftfreq(len(audio), 1/sample_rate)
    magnitudes = np.abs(np.fft.fft(audio))
    return frequencies, magnitudes

def get_spectral_envelope(audio, sample_rate):
    spectrogram = librosa.stft(audio)
    envelope = librosa.amplitude_to_db(np.abs(spectrogram)).max(axis=0)
    return envelope

# Graphical display functions
def show_waveform(audio, sample_rate):
    plt.figure(figsize=(12, 4))
    plt.plot(audio)
    plt.title('Waveform')
    plt.xlabel('Time')
    plt.ylabel('Amplitude')
    plt.show()

def show_spectrogram(audio, sample_rate):
    plt.figure(figsize=(12, 8))
    plt.specgram(audio, Fs=sample_rate)
    plt.title('Spectrogram')
    plt.xlabel('Time')
    plt.ylabel('Frequency')
    plt.colorbar(format='%+2.0f dB')
    plt.show()

def show_frequency_spectrum(audio, sample_rate):
    frequencies, amplitudes = get_frequency_spectrum(audio, sample_rate)
    plt.figure(figsize=(12, 6))
    plt.plot(frequencies, amplitudes)
    plt.title('Frequency Spectrum')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Amplitude')
    plt.xlim(0, sample_rate/2)
    plt.show()

def show_spectral_envelope(audio, sample_rate):
    envelope = get_spectral_envelope(audio, sample_rate)
    plt.figure(figsize=(12, 6))
    plt.plot(envelope)
    plt.title('Spectral Envelope')
    plt.xlabel('Time')
    plt.ylabel('Amplitude')
    plt.show()

# List of available plots
plots = [
    ("Waveform", show_waveform),
    ("Spectrogram", show_spectrogram),
    ("Frequency Spectrum", show_frequency_spectrum),
    ("Spectral Envelope", show_spectral_envelope),
]

# GUI code
layout = [
    [
        sg.TabGroup([
            [
                sg.Tab('Analysis', [
                    [sg.Text("Select an audio file", justification='center')],
                    [sg.InputCombo([], key="-FILE-", enable_events=True, size=(40, 1)), sg.FileBrowse()],
                    [sg.Text("")],
                    [sg.HorizontalSeparator()],
                    [sg.Column([[sg.Button("Analyze", size=(10, 1))]], justification='center')],
                    [sg.Text("Audio File Information", justification='center')],
                    [sg.Multiline(size=(60, 10), key="-OUTPUT-", disabled=True)],
                    [sg.Text("")],
                    [sg.HorizontalSeparator()],
                    [sg.Text("Graphical Representations of the Audio File", font=("Helvetica", 10))],
                    [sg.Button(plot_title) for plot_title, _ in plots]
                ]),
                sg.Tab('Help', [
                    [sg.Text("Supported Audio Formats:")],
                    [sg.Text(", ".join(SUPPORTED_FORMATS))],
                    [sg.Text("")],
                    [sg.Text("")],
                    [sg.Text("@Copyleft all wrongs reserved")],
                    [sg.Text("https://github.com/KorrAudio/beta_KorrAudio")],
                ])
            ]
        ])
    ]
]

window = sg.Window("KorrAudio", layout)

while True:
    event, values = window.read()

    if event == sg.WINDOW_CLOSED:
        break

    if event == "Analyze":
        file_path = values["-FILE-"]
        if os.path.isfile(file_path):
            file_extension = os.path.splitext(file_path)[1][1:].strip().lower()
            if file_extension in SUPPORTED_FORMATS:
                results = analyze_audio(file_path)
                window["-OUTPUT-"].update(results)
            else:
                window["-OUTPUT-"].update("The selected file is not a valid audio file.")
        else:
            window["-OUTPUT-"].update("The selected file does not exist.")

    for plot_title, plot_func in plots:
        if event == plot_title:
            if os.path.isfile(file_path):
                audio, sample_rate = librosa.load(file_path)
                plot_func(audio, sample_rate)
if __name__ == "__main__":
    window.close()
