"""
Entry point to run the validator.
"""
from neurons.validator import Validator
import time

if __name__ == "__main__":
    with Validator() as validator:
        print("Validator started. Press Ctrl+C to stop.")
        while True:
            time.sleep(5)