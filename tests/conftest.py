"""Shared test doubles."""

import threading

import pytest


class LiveRecording:
    """What the live viewers read from a streamed recording: arrived acquisitions, completed datasets, memory and
    unload. Scans are encoded with quantem.gpu's LiveEncoder, as a receiver would; the receiving loop itself
    (queues, threads) belongs to the streaming package and is not needed to test the viewers.
    """

    def __init__(self, *, receiving: bool = True):
        self.lock = threading.RLock()
        self.acquisitions, self.datasets = [], []
        self.error, self.receiving = None, receiving

    def add(self, counts, scan_shape, series: str):
        """Encode one complete scan of ``counts`` (n, det_row, det_col) on the GPU and keep it."""
        from quantem.gpu.streaming import LiveEncoder

        encoder = LiveEncoder(scan_shape, counts.shape[1:], metadata={"series": series})
        with self.lock:
            self.acquisitions.append(encoder)
        encoder.add(0, counts)
        dataset = encoder.finish()
        with self.lock:
            self.datasets.append(dataset)
        return dataset

    def memory_snapshot(self) -> dict:
        with self.lock:
            scans = [{**encoder.memory_snapshot(), "state": "Ready", "can_unload": True, "unload_reason": ""}
                     for encoder in self.acquisitions]
        return {"scans": scans, "releasing": [], "receiving": self.receiving, "error": None}

    def unload(self, scan_id: str) -> None:
        with self.lock:
            encoder = next(item for item in self.acquisitions if item.identifier == scan_id)
            dataset = next(item for item in self.datasets if item.data is encoder.source)
        self.discard(dataset)

    def discard(self, dataset) -> None:
        """Drop a completed scan; its counts go once the viewers still reading it close."""
        with self.lock:
            self.datasets[:] = [item for item in self.datasets if item is not dataset]
            retired = [item for item in self.acquisitions if item.source is dataset.data]
            self.acquisitions[:] = [item for item in self.acquisitions if item.source is not dataset.data]
        for encoder in retired:
            encoder._retire()

    def release(self) -> None:
        self.receiving = False
        for encoder in self.acquisitions:
            encoder.release()
        for dataset in self.datasets:
            dataset.close()


@pytest.fixture
def live_recording():
    """A factory: ``live_recording(receiving=...)`` returns an empty LiveRecording."""
    return LiveRecording
