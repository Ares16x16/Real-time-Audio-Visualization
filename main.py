import numpy as np
import tkinter as tk
from tkinter import colorchooser
import pyaudio
import threading

import mode

# Constants
CHUNK_SIZE = 2048  # Number of audio frames per buffer
FORMAT = pyaudio.paInt16  # Audio format (16-bit)
CHANNELS = 1  # Mono audio
RATE = 44100  # Sample rate (Hz)

# Initialize Tkinter
window = tk.Tk()
window.title("Real-time Audio Visualizer")
window.resizable(False, False)
# Set up the Tkinter canvas
WINDOW_WIDTH = 800
WINDOW_HEIGHT = 400
canvas = tk.Canvas(window, width=WINDOW_WIDTH, height=WINDOW_HEIGHT)
canvas.pack()


class AudioVisualizer:
    def __init__(self, input_device_index, output_device_index):
        self.input_device_index = input_device_index
        self.output_device_index = output_device_index

        # Open audio stream
        self.p = pyaudio.PyAudio()

        # Open input stream
        self.input_stream = self.p.open(
            format=FORMAT,
            channels=CHANNELS,
            rate=RATE,
            input=True,
            input_device_index=self.input_device_index,
            frames_per_buffer=CHUNK_SIZE,
        )

        # Open output stream
        self.output_stream = self.p.open(
            format=FORMAT,
            channels=CHANNELS,
            rate=RATE,
            output=True,
            output_device_index=self.output_device_index,
            frames_per_buffer=CHUNK_SIZE,
        )

        # Initialize settings
        self.bar_color = "#ffffff"  # Default color: white
        self.visualization_mode = {
            "Average Horizontal Rectangle": mode.average_horizontal_rectangle,  # Default
            "Anti-Aliasing Filter Horizontal Rectangle": mode.anti_aliasing_filter_horizontal_rectangle,
            "Fast Fourier Transform Horizontal Rectangle": mode.fft_horizontal_rectangle,
            "Anti-Aliasing Filter Vertical Circle": mode.anti_aliasing_filter_vertical_circle,
            "Anti-Aliasing Filter Vertical Inner Circle": mode.anti_aliasing_filter_vertical_inner_circle,
            "Anti-Aliasing Filter Circle": mode.anti_aliasing_filter_circle,
            "Mel-Frequency Cepstral Coefficients": mode.mel_frequency_cepstral_coefficients,
        }
        self.current_mode_name = "Average Horizontal Rectangle"

        # Create menu bar
        menubar = tk.Menu(window)
        window.config(menu=menubar)

        """
        Create color submenu
        """
        color_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Color", menu=color_menu)
        # Add circle palette to color submenu
        color_menu.add_command(label="Set Color", command=self.open_color_palette)

        """
        Create input and output devices submenu
        """
        input_devices_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Input Device", menu=input_devices_menu)
        output_devices_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Output Device", menu=output_devices_menu)

        # Populate devices submenu
        input_devices, output_devices = self.get_devices()
        for i, device in enumerate(input_devices):
            input_devices_menu.add_command(
                label=device["name"],
                command=lambda index=device["index"]: self.set_input_device(index),
            )
        for i, device in enumerate(output_devices):
            output_devices_menu.add_command(
                label=device["name"],
                command=lambda index=device["index"]: self.set_output_device(index),
            )

        """
        Create mode submenu
        """
        mode_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Visualization Mode", menu=mode_menu)
        mode_menu.add_command(
            label="Average - Horizontal Rectangle",
            command=lambda: self.set_mode("Average Horizontal Rectangle"),
        )
        mode_menu.add_command(
            label="Anti-Aliasing Filter - Horizontal Rectangle",
            command=lambda: self.set_mode("Anti-Aliasing Filter Horizontal Rectangle"),
        )
        mode_menu.add_command(
            label="Anti-Aliasing Filter - Vertical Circle",
            command=lambda: self.set_mode("Anti-Aliasing Filter Vertical Circle"),
        )
        mode_menu.add_command(
            label="Anti-Aliasing Filter - Vertical Inner Circle",
            command=lambda: self.set_mode("Anti-Aliasing Filter Vertical Inner Circle"),
        )
        mode_menu.add_command(
            label="Anti-Aliasing Filter - Circle",
            command=lambda: self.set_mode("Anti-Aliasing Filter Circle"),
        )
        mode_menu.add_command(
            label="Fast Fourier Transform - Horizontal Rectangle",
            command=lambda: self.set_mode(
                "Fast Fourier Transform Horizontal Rectangle"
            ),
        )
        mode_menu.add_command(
            label="Mel-Frequency Cepstral Coefficients",
            command=lambda: self.set_mode("Mel-Frequency Cepstral Coefficients"),
        )

        self.audio_event = threading.Event()

    def get_devices(self):
        input_devices, output_devices = [], []
        info = self.p.get_host_api_info_by_index(0)
        device_count = info.get("deviceCount")
        for i in range(device_count):
            device_info = self.p.get_device_info_by_index(i)
            if device_info["maxInputChannels"] > 0:
                input_devices.append({"index": i, "name": device_info["name"]})
            if device_info["maxOutputChannels"] > 0:
                output_devices.append({"index": i, "name": device_info["name"]})
        return input_devices, output_devices

    def set_input_device(self, index):
        self.input_device_index = index
        self.input_stream = self.p.open(
            format=FORMAT,
            channels=CHANNELS,
            rate=RATE,
            input=True,
            input_device_index=self.input_device_index,
            frames_per_buffer=CHUNK_SIZE,
        )
        window.update()

    def set_output_device(self, index):
        self.output_device_index = index
        self.output_stream = self.p.open(
            format=FORMAT,
            channels=CHANNELS,
            rate=RATE,
            output=True,
            output_device_index=self.output_device_index,
            frames_per_buffer=CHUNK_SIZE,
        )
        window.update()

    def open_color_palette(self):
        color = colorchooser.askcolor(title="Choose Color")
        self.bar_color = color[1] if color[1] else self.bar_color

    def set_mode(self, selected_mode):
        self.current_mode_name = selected_mode

    def process_audio(self):
        while True:
            audio_buffer = self.input_stream.read(CHUNK_SIZE)
            audio_data = np.frombuffer(audio_buffer, dtype=np.int16)

            # drawing function
            window.after(0, self.visualize, audio_data)
            self.output_stream.write(audio_buffer)

    def visualize(self, audio_data):
        canvas.delete("all")
        # print(self.current_mode_name)
        self.visualization_mode[self.current_mode_name](
            visualizer, window, canvas, audio_data
        )

    def start_visualization(self):
        self.process_audio_thread = threading.Thread(target=self.process_audio)
        self.process_audio_thread.daemon = True
        self.process_audio_thread.start()

    def run(self):
        self.start_visualization()

        window.mainloop()

        # Cleanup
        self.input_stream.stop_stream()
        self.input_stream.close()
        self.output_stream.stop_stream()
        self.output_stream.close()
        self.p.terminate()


if __name__ == "__main__":
    visualizer = AudioVisualizer(input_device_index=7, output_device_index=12)
    visualizer.run()
