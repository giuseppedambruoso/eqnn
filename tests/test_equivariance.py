import pytest
import torch
from qnn import create_qnn
from data_encoding import embedding_unitary

# Configurazione globale del test
DEVICE_NAME = "default.qubit"
P_ERR = 0.0
NUM_IMAGES = 10
REPS = 2

@pytest.fixture(scope="module")
def device_and_tensors():
    """Prepara i dati casuali e i parametri del modello."""
    torch.manual_seed(42)
    torch_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Inizializzazione parametri
    params = torch.empty(8*REPS, device=torch_device).uniform_(-0.1, 0.1)
    params.requires_grad_()
    phi = torch.empty(1, device=torch_device).uniform_(-0.1, 0.1)
    phi.requires_grad_()
    
    # Generazione immagini di test
    test_images = torch.rand(NUM_IMAGES, 16, 16, device=torch_device)
    
    return torch_device, params, phi, test_images

@pytest.mark.parametrize("non_equivariance", [0, 4])
@pytest.mark.parametrize("img_idx", range(NUM_IMAGES))
def test_p4m_equivariance(device_and_tensors, non_equivariance, img_idx):
    """Verifica l'invarianza dell'output per riflessione p4m."""
    torch_device, params, phi, test_images = device_and_tensors
    
    # Creazione del nodo QNN
    qnn_node = create_qnn(device=DEVICE_NAME, non_equivariance=non_equivariance, p_err=P_ERR, reps=REPS)
    
    img = test_images[img_idx]
    # Applica flip lungo l'asse y (riflessione p4m)
    img_flip = torch.flip(img, dims=[-1])

    # Embedding ed esecuzione
    out_orig = qnn_node(embedding_unitary(img), params, phi)
    out_flip = qnn_node(embedding_unitary(img_flip), params, phi)

    # L'output deve essere identico per costruzione (Twirling/EquivAnsatz)
    assert torch.allclose(out_orig, out_flip, atol=1e-2), \
        f"Equivariance failed: Mode {non_equivariance}, Image Index {img_idx}"
