import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
import matplotlib.pyplot as plt
import logging

# Configurazione logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# ==========================================
# 1. Definizione dell'Autoencoder
# ==========================================
class ResizeAutoencoder(nn.Module):
    def __init__(self):
        super(ResizeAutoencoder, self).__init__()
        
        # ENCODER: Da 256x256 -> 16x16
        self.encoder = nn.Sequential(
            # Input: 1 x 256 x 256 -> Output: 16 x 128 x 128
            nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            # Output: 32 x 64 x 64
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            # Output: 16 x 32 x 32
            nn.Conv2d(32, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            # Bottleneck: 1 x 16 x 16 (Usiamo Sigmoid per visualizzarlo come immagine 0-1)
            nn.Conv2d(16, 1, kernel_size=3, stride=2, padding=1),
            nn.Sigmoid() 
        )
        
        # DECODER: Da 16x16 -> 256x256
        self.decoder = nn.Sequential(
            # Input: 1 x 16 x 16 -> Output: 16 x 32 x 32
            nn.ConvTranspose2d(1, 16, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            # Output: 32 x 64 x 64
            nn.ConvTranspose2d(16, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            # Output: 16 x 128 x 128
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            # Output finale: 1 x 256 x 256
            nn.ConvTranspose2d(16, 1, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        bottleneck = self.encoder(x)
        reconstructed = self.decoder(bottleneck)
        return bottleneck, reconstructed

# ==========================================
# 2. Caricamento Dati NWPU
# ==========================================
def load_nwpu_train_data(data_dir="data/NWPU-RESISC45", batch_size=32):
    train_dir = os.path.join(data_dir, "train", "train")
    
    # Trasformazioni base: Originale 256x256, Grayscale
    base_transforms = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
    ])
    
    temp_dataset = torchvision.datasets.ImageFolder(root=train_dir)
    airplane_idx = temp_dataset.class_to_idx.get('airplane')
    ship_idx = temp_dataset.class_to_idx.get('ship')
    
    if airplane_idx is None or ship_idx is None:
        raise ValueError(f"Classi 'airplane' o 'ship' non trovate in {train_dir}.")
        
    dataset = torchvision.datasets.ImageFolder(root=train_dir, transform=base_transforms)
    
    # Filtriamo solo aerei e navi
    targets = torch.as_tensor(dataset.targets)
    valid_idx = torch.isin(targets, torch.tensor([airplane_idx, ship_idx])).nonzero(as_tuple=True)[0]
    
    final_dataset = Subset(dataset, valid_idx.tolist())
    dataloader = DataLoader(final_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    
    return final_dataset, dataloader, dataset.classes

# ==========================================
# 3. Training Loop (Rapido)
# ==========================================
def train_autoencoder(model, dataloader, epochs=5, device='cpu'):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    model.to(device)
    model.train()
    
    logger.info(f"Inizio addestramento Autoencoder per {epochs} epoche su {device}...")
    for epoch in range(epochs):
        running_loss = 0.0
        for images, _ in dataloader:
            images = images.to(device)
            
            optimizer.zero_grad()
            # Vogliamo che la ricostruzione sia uguale all'immagine originale 256x256
            _, reconstructed = model(images)
            loss = criterion(reconstructed, images)
            
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * images.size(0)
            
        epoch_loss = running_loss / len(dataloader.dataset)
        logger.info(f"Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss:.4f}")
        
    logger.info("Addestramento completato.")
    return model

# ==========================================
# 4. Generazione PDF di Confronto
# ==========================================
def generate_comparison_pdf(dataset, model, class_names, num_samples=15, output_pdf="resize_comparison.pdf", device='cpu'):
    model.eval()
    
    # Tool per il resize brutale
    brutal_resizer = transforms.Resize((16, 16))
    
    # Selezioniamo 15 indici casuali
    indices = random.sample(range(len(dataset)), num_samples)
    
    # Prepariamo la griglia: 15 righe x 3 colonne
    fig, axes = plt.subplots(num_samples, 3, figsize=(9, 3 * num_samples))
    
    logger.info(f"Generazione del PDF con {num_samples} esempi...")
    
    with torch.no_grad():
        for row, idx in enumerate(indices):
            # 1. Immagine Originale (256x256)
            original_img, label_idx = dataset[idx]
            original_img_batch = original_img.unsqueeze(0).to(device)
            
            # 2. Resize Brutale (16x16)
            brutal_img = brutal_resizer(original_img)
            
            # 3. Resize Autoencoder (Il Bottleneck 16x16)
            bottleneck, _ = model(original_img_batch)
            ae_img = bottleneck.squeeze(0).cpu() # Rimuoviamo la dimensione del batch
            
            class_name = class_names[dataset.dataset.targets[dataset.indices[idx]]]
            
            # PLOT ORIGINAL
            ax = axes[row, 0]
            ax.imshow(original_img.squeeze(), cmap="gray")
            if row == 0: ax.set_title("Original (256x256)", fontsize=12, fontweight='bold')
            ax.set_ylabel(class_name, fontsize=10, rotation=0, labelpad=30, ha='right', va='center')
            ax.set_xticks([]); ax.set_yticks([])
            
            # PLOT BRUTAL RESIZE
            ax = axes[row, 1]
            ax.imshow(brutal_img.squeeze(), cmap="gray")
            if row == 0: ax.set_title("Brutal Resize Torch (16x16)", fontsize=12, fontweight='bold')
            ax.axis("off")
            
            # PLOT AUTOENCODER RESIZE
            ax = axes[row, 2]
            ax.imshow(ae_img.squeeze(), cmap="gray")
            if row == 0: ax.set_title("Autoencoder Bottleneck (16x16)", fontsize=12, fontweight='bold')
            ax.axis("off")
            
    plt.tight_layout()
    plt.savefig(output_pdf, format="pdf", bbox_inches="tight")
    plt.close()
    
    logger.info(f"✅ PDF salvato con successo come: {output_pdf}")

# ==========================================
# Esecuzione
# ==========================================
if __name__ == "__main__":
    # Setup Device
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    
    # 1. Carica i dati
    # Assicurati che il path 'data/NWPU-RESISC45' esista e contenga le cartelle corrette
    dataset, dataloader, class_names = load_nwpu_train_data()
    
    # 2. Inizializza il modello
    ae_model = ResizeAutoencoder()
    
    # 3. Addestra il modello (5 epoche sono sufficienti per vedere una differenza di pattern, 
    # ma puoi aumentarle a 10-20 per una qualità migliore)
    ae_model = train_autoencoder(ae_model, dataloader, epochs=15, device=device)
    
    # 4. Genera il confronto
    generate_comparison_pdf(dataset, ae_model, class_names, num_samples=15, output_pdf="resize_comparison.pdf", device=device)
