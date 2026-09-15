"""
Live EEG streaming via BrainFlow. Defaults to BrainFlow's built-in
SYNTHETIC_BOARD so this file is fully testable without any hardware
(verified working below) -- swap board_id to a real device for the
actual demo:

    BoardIds.MUSE_S_BOARD           (Muse S, needs BLE + brainflow's muse bridge)
    BoardIds.CYTON_BOARD             (OpenBCI Cyton, USB dongle)
    BoardIds.SYNTHETIC_BOARD         (no hardware -- generates realistic fake EEG, used here)

Usage:
    streamer = EEGStreamer(board_id=BoardIds.SYNTHETIC_BOARD.value)
    streamer.start()
    epoch = streamer.get_epoch(epoch_sec=4)   # shape (n_channels, n_samples)
    streamer.stop()
"""
import time
import numpy as np
from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds


class EEGStreamer:
    def __init__(self, board_id=BoardIds.SYNTHETIC_BOARD.value, serial_port=""):
        params = BrainFlowInputParams()
        if serial_port:
            params.serial_port = serial_port
        self.board_id = board_id
        self.board = BoardShim(board_id, params)
        self.sampling_rate = BoardShim.get_sampling_rate(board_id)
        self.eeg_channels = BoardShim.get_eeg_channels(board_id)

    def start(self):
        self.board.prepare_session()
        self.board.start_stream()
        time.sleep(1)  # let the buffer fill briefly

    def get_epoch(self, epoch_sec=4):
        """Blocks until epoch_sec worth of new samples are available, returns
        (n_channels, n_samples) float array -- same shape eeg_features.py expects."""
        n_needed = int(epoch_sec * self.sampling_rate)
        time.sleep(epoch_sec)
        data = self.board.get_current_board_data(n_needed)
        eeg_data = data[self.eeg_channels, :]
        return eeg_data

    def stop(self):
        self.board.stop_stream()
        self.board.release_session()


if __name__ == "__main__":
    # Sanity test using the synthetic board -- runs anywhere, no hardware needed.
    streamer = EEGStreamer(board_id=BoardIds.SYNTHETIC_BOARD.value)
    print(f"Sampling rate: {streamer.sampling_rate} Hz | EEG channels: {streamer.eeg_channels}")
    streamer.start()
    epoch = streamer.get_epoch(epoch_sec=2)
    streamer.stop()
    print(f"Captured epoch shape: {epoch.shape}")
    print("Sample values (channel 0, first 5 samples):", epoch[0, :5])
