"""Architecture-preserving baseline snapshot; callers only receive deep clones."""
import copy


class ModelSnapshot:
    def __init__(self, model):
        self._baseline = copy.deepcopy(model)

    def restore(self):
        return copy.deepcopy(self._baseline)
