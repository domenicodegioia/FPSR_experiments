import torch
from tqdm import tqdm


def batch_sparse_matmul_sparse_output(A, B, device, batch_size=1000):
    """
    Calcola A @ B a blocchi e restituisce una matrice sparse COO come output.

    Args:
        A (torch.sparse_coo_tensor): Primo tensore sparso.
        B (torch.sparse_coo_tensor): Secondo tensore sparso.
        batch_size (int): Dimensione del batch per l'elaborazione.

    Returns:
        torch.sparse_coo_tensor: Risultato della moltiplicazione come tensore sparso.
    """

    A = A.coalesce()
    B = B.coalesce()

    A_indices = A.indices()
    A_values = A.values()
    B_dense = B.to_dense()

    n = A.size(0)
    m = B.size(1)

    result_rows = []

    for start in tqdm(range(0, n, batch_size), disable=True):
        end = min(start + batch_size, n)

        # Seleziona solo le righe del batch per A
        mask = (A_indices[0] >= start) & (A_indices[0] < end)
        batch_indices = A_indices[:, mask].clone()
        batch_indices[0] -= start
        batch_values = A_values[mask]

        # Costruisci blocco sparso per A
        A_block = torch.sparse_coo_tensor(
            batch_indices,
            batch_values,
            size=(end - start, A.size(1)),
            device=A.device,
            dtype=A.dtype
        ).coalesce()

        # Sparse @ Dense (eseguito su GPU)
        block_result = torch.matmul(A_block, B_dense)  # (batch_size, m), denso

        result_rows.append(block_result.cpu())

        del A_block, block_result, batch_indices, batch_values
        torch.cuda.empty_cache()

    # Ricostruisci tutta la matrice densa su CPU
    full_dense_cpu = torch.cat(result_rows, dim=0)  # shape: (n, m)

    # Pulizia della GPU per il tensore B
    del B_dense
    torch.cuda.empty_cache()

    # Applica soglia per costruire sparse
    mask = full_dense_cpu != 0

    indices = mask.nonzero(as_tuple=False).T  # shape: (2, nnz)
    values = full_dense_cpu[mask]

    del full_dense_cpu
    torch.cuda.empty_cache()

    # Costruisci tensore sparso sul device finale
    return torch.sparse_coo_tensor(
            indices.to(device),
            values.to(device),
            size=(n, m),
            device=device,
            dtype=A.dtype
    ).coalesce()
