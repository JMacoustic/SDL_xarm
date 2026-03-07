from dataclasses import dataclass
import numpy as np
from mathutils import Orientation


@dataclass(frozen=True)
class MarkerData:
    index: int #= "Empty"
    corner_pos: np.ndarray #= np.zeros((4, 2))
    orientation: Orientation #= Orientation(np.identity(3), np.zeros((3, 1)))

    def __str__(self) -> str:
        def fmt(arr: np.ndarray) -> str:
            return np.array2string(
                arr,
                precision=4,
                suppress_small=True,
                separator=", "
            )

        return (
            "\n"
            f"========== {self.index} ==========\n"
            f"  corners :\n{fmt(self.corner_pos)},\n"
            f"  rotation :\n{fmt(self.orientation.rot)},\n"
            f"  translation :\n{fmt(self.orientation.trans)}\n"
        )
