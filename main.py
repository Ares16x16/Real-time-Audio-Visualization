import numpy as np
import tkinter as tk
from tkinter import colorchooser
import pyaudio
import threading

# Constants
CHUNK_SIZE = 1024  # Number of audio frames per buffer
FORMAT = pyaudio.paInt16  # Audio format (16-bit)
CHANNELS = 1  # Mono audio
RATE = 44100  # Sample rate (Hz)

# Initialize Tkinter
window = tk.Tk()
window.title("Real-time Audio Visualizer")

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

        # Create menu bar
        menubar = tk.Menu(window)
        window.config(menu=menubar)

        # Create color submenu
        color_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Color", menu=color_menu)

        # Add circle palette to color submenu
        color_menu.add_command(label="Set Color", command=self.open_color_palette)

        # Create a threading Event for synchronization
        self.audio_event = threading.Event()

    def open_color_palette(self):
        color = colorchooser.askcolor(title="Choose Color")
        self.bar_color = color[1] if color[1] else self.bar_color

    def process_audio(self):
        while True:
            audio_buffer = self.input_stream.read(CHUNK_SIZE)
            audio_data = np.frombuffer(audio_buffer, dtype=np.int16)

            # Notify the drawing function
            window.after(0, self.draw_bars, audio_data)

            self.output_stream.write(audio_buffer)

    def draw_bars(self, audio_data):
        canvas.delete("all")
        bar_width = 5
        bar_heights = np.abs(audio_data) // 100

        for i in range(CHUNK_SIZE):
            bar_x = i * bar_width
            bar_y = WINDOW_HEIGHT / 2
            canvas.create_rectangle(
                bar_x,
                bar_y,
                bar_x + bar_width,
                bar_y - bar_heights[i],
                fill=self.bar_color,
            )

    def start_visualization(self):
        # Start the audio processing thread
        self.process_audio_thread = threading.Thread(target=self.process_audio)
        self.process_audio_thread.daemon = True
        self.process_audio_thread.start()

    def run(self):
        self.start_visualization()

        # Start the Tkinter main loop
        window.mainloop()

        # Cleanup
        self.input_stream.stop_stream()
        self.input_stream.close()
        self.output_stream.stop_stream()
        self.output_stream.close()
        self.p.terminate()


# Usage
visualizer = AudioVisualizer(input_device_index=7, output_device_index=12)
visualizer.run()
