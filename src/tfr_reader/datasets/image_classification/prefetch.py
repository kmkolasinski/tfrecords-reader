"""
PrefetchBuffer: Asynchronous batch preparation with background threading.
"""

import queue
import threading
from collections.abc import Callable

import numpy as np


class PrefetchBuffer:
    """
    Asynchronous batch prefetcher using background threads.

    Maintains a buffer of pre-loaded batches to reduce waiting time
    during training.
    """

    def __init__(self, batch_generator: Callable, buffer_size: int = 3, num_workers: int = 1):
        """
        Initialize the PrefetchBuffer.

        Args:
            batch_generator: Callable that generates batches (should be thread-safe)
            buffer_size: Number of batches to prefetch
            num_workers: Number of worker threads (usually 1 is sufficient)
        """
        self.batch_generator = batch_generator
        self.buffer_size = buffer_size
        self.num_workers = num_workers

        self.queue: queue.Queue[tuple[np.ndarray, np.ndarray] | None] = queue.Queue(
            maxsize=buffer_size
        )
        self.workers = []
        self.stop_event = threading.Event()
        self.exception: RuntimeError | OSError | None = None

        # Start worker threads
        for _ in range(num_workers):
            worker = threading.Thread(target=self._worker_loop, daemon=True)
            worker.start()
            self.workers.append(worker)

    def _worker_loop(self):
        """Worker thread loop that prefetches batches."""
        try:
            while not self.stop_event.is_set():
                # Generate a batch
                batch = self.batch_generator()

                if batch is None:
                    # End of data - put sentinel and stop
                    self.queue.put(None)
                    break

                # Put in queue (blocks if queue is full)
                if not self.stop_event.is_set():
                    self.queue.put(batch, timeout=1.0)

        except (RuntimeError, OSError) as e:
            self.exception = e
            self.queue.put(None)  # Sentinel to unblock consumer

    def get(self, timeout: float | None = None) -> tuple[np.ndarray, np.ndarray] | None:
        """
        Get the next prefetched batch.

        Args:
            timeout: Maximum time to wait for a batch (None = wait forever)

        Returns:
            Batch tuple (images, labels) or None if no more data

        Raises:
            Exception: If an error occurred in the worker thread
        """
        if self.exception is not None:
            raise self.exception

        try:
            batch = self.queue.get(timeout=timeout)

            if self.exception is not None:
                raise self.exception
        except queue.Empty:
            return None
        else:
            return batch

    def stop(self):
        """Stop all worker threads and cleanup."""
        self.stop_event.set()

        # Drain the queue to unblock workers
        try:
            while True:
                self.queue.get_nowait()
        except queue.Empty:
            pass

        # Wait for workers to finish
        for worker in self.workers:
            worker.join(timeout=2.0)

    def __del__(self):
        """Cleanup on deletion."""
        self.stop()
