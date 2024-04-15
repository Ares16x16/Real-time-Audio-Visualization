import numpy as np
import cmath
import tkinter as tk
import pyaudio
from scipy import signal
from scipy.fftpack import dct
import librosa

# from scipy.fft import fft

"""
Average Horizontal Rectangle
"""


def downsample_mean(audio_data, bar_width, window_width):
    new_length = window_width // bar_width - 5
    segment_length = len(audio_data) // new_length
    downsampled_audio = np.zeros(new_length)

    for i in range(new_length):
        start_index = i * segment_length
        end_index = start_index + segment_length
        segment = audio_data[start_index:end_index]
        downsampled_audio[i] = np.mean(segment)

    return downsampled_audio


def average_horizontal_rectangle(visualizer, window, canvas, audio_data):
    bar_width = 3
    window_width = window.winfo_width()
    window_height = window.winfo_height()

    downsample_audio = downsample_mean(audio_data, bar_width, window_width)

    bar_heights = np.abs(downsample_audio) // 75
    audio_length = len(downsample_audio)

    for i in range(audio_length):
        bar_x = i * bar_width + (window_width / 2) - (audio_length * bar_width / 2)
        bar_y = window_height / 2
        canvas.create_rectangle(
            bar_x,
            bar_y,
            bar_x + bar_width,
            bar_y - bar_heights[i],
            fill=visualizer.bar_color,
        )


"""
Anti-Aliasing Filter Horizontal Rectangle
"""


def downsample_with_filter(audio_data, factor):
    # downsampling by an integer factor
    downsampled_audio = signal.decimate(audio_data, factor, zero_phase=True)

    # anti-aliasing filter
    cutoff_freq = 0.5 / factor
    b, a = signal.butter(4, cutoff_freq, analog=False)
    filtered_audio = signal.lfilter(b, a, downsampled_audio)

    return filtered_audio


def anti_aliasing_filter_horizontal_rectangle(visualizer, window, canvas, audio_data):
    bar_width = 3
    window_width = window.winfo_width()
    window_height = window.winfo_height()

    audio_length = window_width // bar_width - 5
    segment_length = len(audio_data) // audio_length
    downsampled_audio = downsample_with_filter(audio_data, segment_length)
    bar_height = np.abs(downsampled_audio) // 75

    for i in range(audio_length):
        start_index = i * segment_length
        end_index = start_index + segment_length
        segment = downsampled_audio[start_index:end_index]

        bar_x = i * bar_width + (window_width / 2) - (audio_length * bar_width / 2)
        bar_y = window_height / 2
        canvas.create_rectangle(
            bar_x,
            bar_y,
            bar_x + bar_width,
            bar_y - bar_height[i],
            fill=visualizer.bar_color,
        )


"""
Fast Fourier Transform Horizontal Rectangle
"""


def fft(data):
    N = len(data)
    if N <= 1:
        return data
    even = fft(data[0::2])
    odd = fft(data[1::2])

    twiddle_factors = [cmath.exp(-2j * cmath.pi * k / N) for k in range(N // 2)]
    transformed_data = [0] * N

    for k in range(N // 2):
        transformed_data[k] = even[k] + twiddle_factors[k] * odd[k]
        transformed_data[k + N // 2] = even[k] - twiddle_factors[k] * odd[k]

    return transformed_data


def fft_horizontal_rectangle(visualizer, window, canvas, audio_data):
    bar_width = 3
    window_width = window.winfo_width()
    window_height = window.winfo_height()

    audio_length = window_width // bar_width - 5
    segment_length = len(audio_data) // audio_length
    fft_data = fft(audio_data)

    for i in range(audio_length):
        start_index = i * segment_length
        end_index = start_index + segment_length
        segment = fft_data[start_index:end_index]
        magnitudes = np.abs(segment)
        bar_height = np.mean(magnitudes) // 1000

        bar_x = i * bar_width + (window_width / 2) - (audio_length * bar_width / 2)
        bar_y = window_height - 10
        canvas.create_rectangle(
            bar_x,
            bar_y,
            bar_x + bar_width,
            bar_y - bar_height,
            fill=visualizer.bar_color,
        )


"""
Anti-Aliasing Filter Circle
"""


def anti_aliasing_filter_vertical_circle(visualizer, window, canvas, audio_data):
    bar_width = 3
    window_width = window.winfo_width()
    window_height = window.winfo_height()

    audio_length = window_width // bar_width - 5
    segment_length = len(audio_data) // audio_length
    downsampled_audio = downsample_with_filter(audio_data, segment_length)
    bar_height = np.abs(downsampled_audio) // 75

    circle_center_x = window_width / 2
    circle_center_y = window_height / 2

    circle_radius = min(circle_center_x, circle_center_y) - 40

    for i in range(audio_length):
        start_index = i * segment_length
        end_index = start_index + segment_length
        segment = downsampled_audio[start_index:end_index]

        angle = 2 * np.pi * i / audio_length

        bar_x = circle_center_x + circle_radius * np.cos(angle) - bar_width / 2
        bar_y = circle_center_y + circle_radius * np.sin(angle) - bar_height[i] / 2

        canvas.create_rectangle(
            bar_x,
            bar_y,
            bar_x + bar_width,
            bar_y + bar_height[i],
            fill=visualizer.bar_color,
        )


def anti_aliasing_filter_vertical_inner_circle(visualizer, window, canvas, audio_data):
    bar_width = 3
    window_width = window.winfo_width()
    window_height = window.winfo_height()

    audio_length = window_width // bar_width - 5
    segment_length = len(audio_data) // audio_length
    downsampled_audio = downsample_with_filter(audio_data, segment_length)
    bar_height = np.abs(downsampled_audio) // 75

    circle_center_x = window_width / 2
    circle_center_y = window_height / 2

    circle_radius = min(circle_center_x, circle_center_y) - 40

    for i in range(audio_length):
        start_index = i * segment_length
        end_index = start_index + segment_length
        segment = downsampled_audio[start_index:end_index]

        angle = 2 * np.pi * i / audio_length
        bar_x = circle_center_x + circle_radius * np.cos(angle) - bar_width / 2
        bar_y = circle_center_y + circle_radius * np.sin(angle) - bar_height[i] / 2
        angle_to_center = np.arctan2(circle_center_y - bar_y, circle_center_x - bar_x)

        # Align
        adjusted_bar_x = bar_x + bar_width / 2 * np.cos(angle_to_center)
        adjusted_bar_y = bar_y + bar_height[i] / 2 * np.sin(angle_to_center)

        canvas.create_rectangle(
            adjusted_bar_x,
            adjusted_bar_y,
            adjusted_bar_x + bar_width,
            adjusted_bar_y + bar_height[i],
            fill=visualizer.bar_color,
        )


""" Circle (but something went wrong)"""


def rotate_point(x, y, angle, center_x, center_y):
    x -= center_x
    y -= center_y
    new_x = x * np.cos(angle) - y * np.sin(angle)
    new_y = x * np.sin(angle) + y * np.cos(angle)
    return new_x + center_x, new_y + center_y


def draw_rotated_rectangle(canvas, x, y, width, height, angle, fill_color):
    rect_center_x = x + width / 2
    rect_center_y = y + height / 2

    points = [(x, y), (x + width, y), (x + width, y + height), (x, y + height)]

    rotated_points = [
        rotate_point(px, py, angle, rect_center_x, rect_center_y) for px, py in points
    ]
    flattened_rotated_points = [coord for point in rotated_points for coord in point]
    canvas.create_polygon(flattened_rotated_points, fill=fill_color)


def anti_aliasing_filter_circle(visualizer, window, canvas, audio_data):
    bar_width = 3
    window_width = window.winfo_width()
    window_height = window.winfo_height()

    audio_length = window_width // bar_width - 5
    segment_length = len(audio_data) // audio_length
    downsampled_audio = downsample_with_filter(audio_data, segment_length)
    bar_height = np.abs(downsampled_audio) // 75

    circle_center_x = window_width / 2
    circle_center_y = window_height / 2

    circle_radius = min(circle_center_x, circle_center_y) - 40

    for i in range(audio_length):
        start_index = i * segment_length
        end_index = start_index + segment_length
        segment = downsampled_audio[start_index:end_index]

        angle = 2 * np.pi * i / audio_length

        bar_x = circle_center_x + circle_radius * np.cos(angle) - bar_width / 2
        bar_y = circle_center_y + circle_radius * np.sin(angle) - bar_height[i]

        angle_to_center = np.arctan2(circle_center_y - bar_y, circle_center_x - bar_x)

        draw_rotated_rectangle(
            canvas,
            bar_x,
            bar_y,
            bar_width,
            bar_height[i],
            angle_to_center,
            visualizer.bar_color,
        )


""" mel frequency cepstral coefficients """


def mel_frequency_cepstral_coefficients(visualizer, window, canvas, audio_data):

    window_width = window.winfo_width()
    window_height = window.winfo_height()
    bar_width = 20
    bar_y = window_height // 2

    n_mfcc = window_width // bar_width - 1
    audio_data = np.array(audio_data, dtype=np.float32)
    # audio_data = np.where(audio_data < 5 , 0.0, audio_data)
    # print(audio_data)
    mfccs = librosa.feature.mfcc(y=audio_data, sr=44100, n_mfcc=n_mfcc)
    # mfccs = librosa.power_to_db(mfccs, ref=np.max)
    # np.delete(mfccs, 0, axis=0)
    # print(mfccs)

    # Add up the channels for each frame
    summed_mfccs = np.sum(mfccs, axis=1)
    num_frames = summed_mfccs.shape[0]

    for i in range(num_frames):
        height = summed_mfccs[i] / 3
        bar_x = bar_width / 2 + i * bar_width
        canvas.create_rectangle(
            bar_x,
            bar_y,
            bar_x + bar_width,
            bar_y + height,
            fill=visualizer.bar_color,
        )

        bar_x += bar_width
