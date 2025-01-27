import h5py
import numpy as np
import threading
import time
from torch.utils.data import Dataset


class HDF5ReplayBuffer:
    def __init__(self, file_name, state_dim, action_dim, max_size):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.max_size = max_size
        self.current_size = 0

        # Open the HDF5 file and create datasets
        self.file = h5py.File(file_name, "w")
        self.file.create_dataset("states", shape=(max_size, state_dim), dtype='float32')
        self.file.create_dataset("actions", shape=(max_size, action_dim), dtype='float32')
        self.file.create_dataset("rewards", shape=(max_size,), dtype='float32')
        self.file.create_dataset("next_states", shape=(max_size, state_dim), dtype='float32')
        self.file.create_dataset("dones", shape=(max_size,), dtype='bool')

        self.buffer = []

        self.lock = threading.Lock()

        # Start the background writer thread
        self.writer_thread = HDF5WriterThread(self)
        self.writer_thread.start()

    def add_interaction(self, state, action, reward, next_state, done):
        """Add an interaction to the buffer."""
        with self.lock:  # Use lock to prevent race conditions
            self.buffer.append((state, action, reward, next_state, done))

    def flush(self):
        """Flush the buffer to the HDF5 file."""
        if not self.buffer:
            return

        # Convert buffer to NumPy arrays
        with self.lock:  # Ensure thread safety when accessing the buffer
            buffer_array = np.array(self.buffer, dtype=object)
            states, actions, rewards, next_states, dones = zip(*buffer_array)

            # Write to datasets
            start_idx = self.current_size
            end_idx = start_idx + len(states)

            self.file["states"][start_idx:end_idx] = states
            self.file["actions"][start_idx:end_idx] = actions
            self.file["rewards"][start_idx:end_idx] = rewards
            self.file["next_states"][start_idx:end_idx] = next_states
            self.file["dones"][start_idx:end_idx] = dones

            self.current_size += len(states)
            self.buffer = []

    def __len__(self):
        """Return the current size of the buffer."""
        return self.current_size

    def close(self):
        """Stop the writer thread and close the HDF5 file."""
        # Stop the background writer thread
        self.writer_thread.stop()
        self.writer_thread.join()  # Wait for the thread to finish

        # Flush any remaining data and close the file
        self.flush()
        self.file.close()


class HDF5WriterThread(threading.Thread):
    def __init__(self, buffer):
        super().__init__()
        self.buffer = buffer
        self.running = True

    def run(self):
        """Run the background thread to periodically flush the buffer."""
        while self.running:
            time.sleep(0.1)
            self.buffer.flush()

    def stop(self):
        """Stop the background thread."""
        self.running = False
