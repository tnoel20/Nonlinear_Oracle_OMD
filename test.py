import numpy as np
from sklearn.metrics.cluster import normalized_mutual_info_score as nmi_score

print(nmi_score([0,0,1,1],[0,0,1,1]))
