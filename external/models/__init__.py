def import_model_by_backend(tensorflow_cmd, pytorch_cmd):
    import sys
    for _backend in sys.modules["external"].backend:
        if _backend == "tensorflow":
            exec(tensorflow_cmd)
        elif _backend == "pytorch":
            exec(pytorch_cmd)
            break


from .psge import PSGE
from .sgmc import SGMC
from .gfcf import GFCF
from .svd_ae import SVD_AE
from .bspm import BSPM
from .chebycf import ChebyCF

import sys
for _backend in sys.modules["external"].backend:
    if _backend == "tensorflow":
        pass
    elif _backend == "pytorch":
        from .lightgcn import LightGCN
        from .bism import BISM
        from .svd_gcn import SVDGCN
        from .svd_gcn_s import SVDGCNS
        from .gde import GDE
        from .sgde import SGDE
        from .rsgde import RSGDE
        from .csgde import CSGDE
        from .fpsr import FPSR
        from .fpsr_plus import FPSRplus
        from .fpsr_plus_f import FPSRplusF
        from .turbocf import TurboCF
        from .sgfcf import SGFCF

        # FPSR variants
        from .fpsr_knn import FPSR_KNN
        from .fpsr_rp3beta import FPSR_RP3beta
        from .fpsr_easer import FPSR_EASEr
        from .fpsr_svd import FPSR_PureSVD
        from .fpsr_psge import FPSR_PSGE
        from .fpsr_svdae import FPSR_SVDAE
        from .fpsr_gfcf import FPSR_GFCF