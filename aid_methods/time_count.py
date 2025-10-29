import time

class EpochTimer:
    def __init__(self):
        self.start_time = None

    def start(self):
        """Start the timer for an epoch."""
        self.start_time = time.time()

    def stop(self, epoch):
        """Stop the timer and print the duration of the epoch."""
        if self.start_time is None:
            raise ValueError("Timer was not started. Please call 'start()' before 'stop()'.")
        end_time = time.time()
        elapsed_time = end_time - self.start_time
        print(f"Epoch {epoch} took {elapsed_time:.2f} seconds.")
        # Optionally, log to wandb or other systems here if needed.
        return elapsed_time
