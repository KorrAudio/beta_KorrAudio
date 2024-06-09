#!/usr/bin/env python
# -*- coding: utf-8 -*-

# This file is distributed under the terms of the GPLv3 license
# For more information, refer to the LICENSE file or visit
# http://www.gnu.org/licenses/gpl-3.0.html

import datetime
import hashlib
import os

import librosa
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
from tinytag import TinyTag
from librosa.feature import chroma_stft

import tkinter as tk
from tkinter import filedialog, scrolledtext, messagebox, ttk

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

    # Calculate average loudness
    loudness = librosa.amplitude_to_db(audio)
    average_loudness = np.mean(loudness)

    # Create the analysis results text
    file_info_text = f"File Name: {os.path.basename(file_path)}\n" \
                     f"Audio File Format: {file_format}\n" \
                     f"Last Modified: {modification_time}\n" \
                     f"File Hash: {file_hash}\n"
    
    metadata_text = f"\n" \
                    f"Artist: {artist}\n" \
                    f"Title: {title}\n" \
                    f"Album: {album}\n" \
                    f"Year: {year}\n" \
                    f"Genre: {genre}\n"

    file_analyze_text = f"\n" \
                     f"File Duration: {duration:.2f} seconds\n" \
                     f"Sample Rate: {bitrate} Hz\n" \
                     f"Sampling Frequency: {sample_rate} Hz\n" \
                     f"Number of Channels: {num_channels}\n" \
                     f"Maximum Amplitude: {max_amplitude:.2f} (scaled value)\n" \
                     f"Average Amplitude: {mean_amplitude:.2f} (scaled value)\n" \
                     f"Minimum Frequency: {min_frequency} Hz\n" \
                     f"Maximum Frequency: {max_frequency:.2f} Hz\n"

    tempo_text = f"\n" \
                f"Tempo: {tempo:.2f} BPM\n"

    loudness_text = f"Average Loudness: {average_loudness:.2f} dB\n"
                    
    chroma_text = f"\nChroma Features:\n"
    for note, value in zip(["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"], mean_chroma):
        chroma_text += f"{note}: {value:.3f}\n"

    results = file_info_text + metadata_text + file_analyze_text + tempo_text + loudness_text + chroma_text
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

# Function to update file information and analysis
def update_file_info(file_path):
    if file_path and os.path.isfile(file_path):
        file_extension = os.path.splitext(file_path)[1][1:].strip().lower()
        if file_extension in SUPPORTED_FORMATS:
            results = analyze_audio(file_path)
            output_text.configure(state='normal')
            output_text.delete("1.0", tk.END)
            output_text.insert(tk.END, results)
            output_text.configure(state='disabled')
        else:
            messagebox.showerror("Invalid File", "The selected file is not a valid audio file.")
    else:
        messagebox.showerror("File Not Found", "The selected file does not exist.")

# Function to plot audio analysis
def plot_audio_analysis(file_path, plot_func):
    if file_path and os.path.isfile(file_path):
        audio, sample_rate = librosa.load(file_path)
        plot_func(audio, sample_rate)
    else:
        messagebox.showerror("File Not Found", "The selected file does not exist.")

# Tkinter GUI setup
root = tk.Tk()
root.title("KorrAudio")

# Notebook for tabs
notebook = ttk.Notebook(root)
notebook.pack(pady=10, expand=True)

# Analysis Tab
frame_analysis = ttk.Frame(notebook, width=800, height=600)
frame_analysis.pack(fill='both', expand=True)

file_label = tk.Label(frame_analysis, text="Select an audio file", font=('Helvetica', 12))
file_label.pack(pady=10)

file_entry = tk.Entry(frame_analysis, width=50)
file_entry.pack(side='left', padx=(20, 10))

file_button = tk.Button(frame_analysis, text="Browse", command=lambda: file_entry.insert(0, filedialog.askopenfilename(filetypes=[("Audio files", "*.mp3 *.wav *.ogg *.flac *.aiff")])))
file_button.pack(side='left', padx=10)

analyze_button = tk.Button(frame_analysis, text="Analyze", command=lambda: update_file_info(file_entry.get()))
analyze_button.pack(side='left', padx=10)

output_text = scrolledtext.ScrolledText(frame_analysis, width=100, height=20, state='disabled')
output_text.pack(pady=20)

plot_label = tk.Label(frame_analysis, text="Graphical Representations of the Audio File", font=('Helvetica', 10))
plot_label.pack(pady=10)

plot_buttons_frame = tk.Frame(frame_analysis)
plot_buttons_frame.pack(pady=10)

for plot_title, plot_func in [("Waveform", show_waveform), ("Spectrogram", show_spectrogram), ("Frequency Spectrum", show_frequency_spectrum), ("Spectral Envelope", show_spectral_envelope)]:
    tk.Button(plot_buttons_frame, text=plot_title, command=lambda func=plot_func: plot_audio_analysis(file_entry.get(), func)).pack(side='left', padx=10)

# Help Tab
frame_help = ttk.Frame(notebook, width=800, height=600)
frame_help.pack(fill='both', expand=True)

help_text = tk.Text(frame_help, width=100, height=20, state='normal')
help_text.insert(tk.END, "Supported Audio Formats:\n")
help_text.insert(tk.END, ", ".join(SUPPORTED_FORMATS) + "\n\n")
help_text.insert(tk.END, "KorrAudio - GPLv3 license\n")
help_text.insert(tk.END, "https://github.com/KorrAudio/beta_KorrAudio\n")
help_text.configure(state='disabled')
help_text.pack(pady=20)

notebook.add(frame_analysis, text="Analysis")
notebook.add(frame_help, text="Help")

root.mainloop()
