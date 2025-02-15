import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import time
import os
import torchvision.utils as vutils

def check_device():
    print("\nDevice Information:")
    print("-" * 50)
    if torch.cuda.is_available():
        print("PyTorch is built with CUDA")
    else:
        print("PyTorch is NOT built with CUDA")

    print("\nAvailable devices:")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("No GPU devices available")
    print("-" * 50)

# Call device check before training
check_device()

# GPU Configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# Constants
LATENT_DIM = 100
IMG_SIZE = 48
NUM_CLASSES = 7
BATCH_SIZE = 64
EPOCHS = 100

# Generator model
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.label_embedding = nn.Embedding(NUM_CLASSES, LATENT_DIM)
        self.model = nn.Sequential(
            nn.Linear(LATENT_DIM, 512 * 6 * 6),
            nn.BatchNorm1d(512 * 6 * 6),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Unflatten(1, (512, 6, 6)),
            
            # Adjusted ConvTranspose2d layers for proper upsampling
            nn.ConvTranspose2d(512, 256, 4, 2, 1),  # 12x12
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.ConvTranspose2d(256, 128, 4, 2, 1),  # 24x24
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.ConvTranspose2d(128, 64, 4, 2, 1),   # 48x48
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(64, 1, 3, 1, 1),              # 48x48
            nn.Tanh()
        )

    def forward(self, noise, labels):
        gen_input = noise * self.label_embedding(labels)
        return self.model(gen_input)

# Discriminator model
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.label_embedding = nn.Embedding(NUM_CLASSES, IMG_SIZE * IMG_SIZE)
        
        # Process the label embedding
        self.label_processor = nn.Sequential(
            nn.Linear(IMG_SIZE * IMG_SIZE, IMG_SIZE * IMG_SIZE),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        # Main discriminator model
        self.model = nn.Sequential(
            # Input: 2 channels (1 for image, 1 for label)
            nn.Conv2d(2, 64, 4, 2, 1),              # 24x24
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(64, 128, 4, 2, 1),            # 12x12
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(128, 256, 4, 2, 1),           # 6x6
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(256, 512, 4, 2, 1),           # 3x3
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Flatten(),
            nn.Linear(512 * 3 * 3, 1),
            nn.Sigmoid()
        )

    def forward(self, img, labels):
        # Process label embedding
        label_embed = self.label_embedding(labels)
        label_embed = self.label_processor(label_embed)
        # Reshape label to match image dimensions
        label_embed = label_embed.view(-1, 1, IMG_SIZE, IMG_SIZE)
        # Concatenate image and label
        d_in = torch.cat((img, label_embed), dim=1)
        return self.model(d_in)

# Initialize models
generator = Generator().to(device)
discriminator = Discriminator().to(device)

# Loss functions and optimizers
adversarial_loss = nn.BCELoss()
g_optimizer = optim.Adam(generator.parameters(), lr=0.0001, betas=(0.5, 0.999))
d_optimizer = optim.Adam(discriminator.parameters(), lr=0.0001, betas=(0.5, 0.999))

# Check and create directories if needed
def setup_dataset_path():
    dataset_path = 'dataset/train'
    if not os.path.exists(dataset_path):
        print(f"Dataset not found at {dataset_path}")
        print("Please ensure your dataset is organized as follows:")
        print("dataset/")
        print("  └── train/")
        print("      ├── angry/")
        print("      ├── disgust/")
        print("      ├── fear/")
        print("      ├── happy/")
        print("      ├── neutral/")
        print("      ├── sad/")
        print("      └── surprise/")
        raise FileNotFoundError(f"Dataset not found at {dataset_path}")
    return dataset_path

# Load and preprocess data
try:
    dataset_path = setup_dataset_path()
    transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomRotation(20),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
    dataset = datasets.ImageFolder(root=dataset_path, transform=transform)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
except Exception as e:
    print(f"Error loading dataset: {str(e)}")
    raise

# Print model summaries
print("\nGenerator Summary:")
print(generator)
print("\nDiscriminator Summary:")
print(discriminator)

# Training loop
print("\nStarting training...")
for epoch in range(EPOCHS):
    start = time.time()
    gen_losses = []
    disc_losses = []

    print(f"\nEpoch {epoch+1}/{EPOCHS}")
    for batch_idx, (real_imgs, labels) in enumerate(dataloader):
        batch_size = real_imgs.size(0)
        real_imgs = real_imgs.to(device)
        labels = labels.to(device)
        valid = torch.ones(batch_size, 1, device=device)
        fake = torch.zeros(batch_size, 1, device=device)

        # Train Generator
        generator.zero_grad()
        z = torch.randn(batch_size, LATENT_DIM, device=device)
        gen_labels = labels
        gen_imgs = generator(z, gen_labels)
        validity = discriminator(gen_imgs, gen_labels)
        g_loss = adversarial_loss(validity, valid)
        g_loss.backward()
        g_optimizer.step()
        gen_losses.append(g_loss.item())

        # Train Discriminator
        discriminator.zero_grad()
        real_pred = discriminator(real_imgs, labels)
        real_loss = adversarial_loss(real_pred, valid)
        fake_pred = discriminator(gen_imgs.detach(), gen_labels)
        fake_loss = adversarial_loss(fake_pred, fake)
        d_loss = (real_loss + fake_loss) / 2
        d_loss.backward()
        d_optimizer.step()
        disc_losses.append(d_loss.item())

        if batch_idx % 50 == 0:
            print(f"Batch {batch_idx}/{len(dataloader)} - "
                  f"G_loss: {g_loss.item():.4f}, D_loss: {d_loss.item():.4f}")

    print(f"Time for epoch {epoch+1}: {time.time()-start:.2f} sec")
    print(f"Generator loss: {np.mean(gen_losses):.4f}")
    print(f"Discriminator loss: {np.mean(disc_losses):.4f}")

    # Save generated images periodically
    if (epoch + 1) % 10 == 0:
        with torch.no_grad():
            z = torch.randn(BATCH_SIZE, LATENT_DIM, device=device)
            labels = torch.arange(NUM_CLASSES).repeat(BATCH_SIZE // NUM_CLASSES + 1)[:BATCH_SIZE].to(device)
            gen_imgs = generator(z, labels)
            gen_imgs = gen_imgs * 0.5 + 0.5  # Rescale to [0,1]
            save_path = f'generated_samples_epoch_{epoch+1}.png'
            vutils.save_image(gen_imgs, save_path, nrow=NUM_CLASSES, normalize=True)
            print(f"Generated images saved to {save_path}")

    # Save model periodically
    if (epoch + 1) % 10 == 0:
        torch.save(generator.state_dict(), f'generator_epoch_{epoch+1}.pth')

# Save final model
torch.save(generator.state_dict(), 'generator_final.pth')
