def import_model_by_backend(tensorflow_cmd, pytorch_cmd):
    import sys
    for _backend in sys.modules["external"].backend:
        if _backend == "tensorflow":
            exec(tensorflow_cmd)
        elif _backend == "pytorch":
            exec(pytorch_cmd)
            break


from .psge import PSGE
from .gfcf import GFCF
from .svd_ae import SVD_AE
from .bspm import BSPM
from .chebycf import ChebyCF

import sys
for _backend in sys.modules["external"].backend:
    if _backend == "tensorflow":
        pass
    elif _backend == "pytorch":
        from .svd_gcn import SVDGCN
        from .svd_gcn_s import SVDGCNS
        from .gde import GDE
        from .fpsr import FPSR

        # FPSR variants
        from .fpsr_knn import FPSR_KNN
        from .fpsr_rp3beta import FPSR_RP3beta
        from .fpsr_easer import FPSR_EASEr
        from .fpsr_svd import FPSR_PureSVD
        from .fpsr_psge import FPSR_PSGE
        from .fpsr_svdae import FPSR_SVDAE
        from .fpsr_gfcf import FPSR_GFCF