"""The bounded query interface for an already prepared detector session."""

from typing import Protocol, runtime_checkable


@runtime_checkable
class DetectorSessionSource(Protocol):
    """A borrowed session that computes on its resident acquisition.

    Shapes use (row, col). Query results are host arrays or CPU tensors;
    counts are reduced on the source device before transport. ``metadata``
    carries the same calibration keys as a loaded acquisition. Closing the
    session releases its query resources, not the acquisition it borrows.
    """

    scan_shape: tuple[int, int]
    detector_shape: tuple[int, int]
    device: object
    metadata: dict

    def frame(self, index: int):
        """Return one pattern at a row-major scan index."""
        ...

    def masked_sum(self, mask):
        """Return a float32 virtual image for a boolean detector mask.

        Each returned host array or CPU tensor must retain its values while
        borrowed by the caller, including after later queries and ``close()``.
        A backend that reuses scratch storage must return a copy. The widget
        publishes a borrowed view of this result without copying its bytes.
        """
        ...

    def mean_pattern(self):
        """Return the mean of every diffraction pattern."""
        ...

    def reduce_frames(self, indices, mode: str = "mean"):
        """Reduce selected row-major positions by mean, sum, or max."""
        ...

    def close(self) -> None:
        """Release session resources without closing the source acquisition."""
        ...


class SessionSeries:
    """Prepared query sessions of equal geometry, including scans still arriving."""

    def __init__(self, sources):
        self.frames = []
        for source in sources:
            self.append(source)

    @property
    def shape(self):
        first = self.frames[0]
        return (len(self.frames), *first.scan_shape, *first.detector_shape)

    @property
    def device(self):
        return self.frames[0].device

    def append(self, source):
        session = source.detector_session() if callable(getattr(source, "detector_session", None)) else source
        if not isinstance(session, DetectorSessionSource):
            raise TypeError("A session series requires prepared detector queries.")
        if self.frames and (session.scan_shape != self.frames[0].scan_shape
                            or session.detector_shape != self.frames[0].detector_shape
                            or str(session.device) != str(self.device)):
            session.close()
            raise ValueError("Live sessions must share scan/detector shape and device.")
        self.frames.append(session)
