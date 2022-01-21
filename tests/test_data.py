import numpy as np
import typicle

import graphicle


types_ = typicle.Types()


def test_pdgs():
    pdg_vals = np.arange(1, 7, dtype=types_.int)
    pdgs = graphicle.PdgArray(pdg_vals)
    assert list(pdgs.name) == ["d", "u", "s", "c", "b", "t"]
