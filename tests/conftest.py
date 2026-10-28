import numpy as np
from sklearn.metrics import _scorer

# deepchecks 0.19.1 usa np.Inf y el scorer max_error, que numpy 2 y scikit-learn 1.9 ya no traen
np.Inf = np.inf
_scorer._SCORERS["max_error"] = _scorer._SCORERS["neg_max_error"]
