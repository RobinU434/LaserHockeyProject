from hdf5_replay_buffer import HDF5ReplayBuffer
import numpy as np

def test_no_duplicates():
    print("Running Test: No Duplicates Added")
    state_dim = 5
    action_dim = 2

    buffer = HDF5ReplayBuffer("test_no_duplicates.h5", state_dim=state_dim, action_dim=action_dim, max_size=1000)

    state = np.ones(state_dim) * 0.1
    action = np.ones(action_dim) * 0.2
    buffer.add_interaction(state, action, 1.0, state, False)
    buffer.add_interaction(state, action, 1.0, state, False)

    buffer.flush()
    assert len(buffer) == 2, f"Expected 2 interactions, found {len(buffer)}"
    print("Test Passed: No Duplicates Added")
    buffer.close()

def run_all_tests():
    print("Starting HDF5 Replay Buffer Tests...\n")
    test_no_duplicates()
    print("\nAll Tests Passed!")

if __name__ == "__main__":
    run_all_tests()
