import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import time
import os
import torchvision.utils as vutils
import numpy as np

# Constants
LATENT_DIM = 100
IMG_SIZE = 48
NUM_CLASSES = 7
BATCH_SIZE = 64
ADDITIONAL_EPOCHS = 200  # Just need this one

# Use GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# Import the model architectures from train.py
from train import Generator, Discriminator, setup_dataset_path

def continue_training():
    # Initialize models
    generator = Generator().to(device)
    discriminator = Discriminator().to(device)
    
    # Load the latest checkpoint
    generator.load_state_dict(torch.load('generator_continued_final.pth', map_location=device))
    
    # Initialize optimizers
    g_optimizer = optim.Adam(generator.parameters(), lr=0.0001, betas=(0.5, 0.999))
    d_optimizer = optim.Adam(discriminator.parameters(), lr=0.0001, betas=(0.5, 0.999))
    adversarial_loss = nn.BCELoss()
    
    # Load dataset
    transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomRotation(20),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
    
    dataset = datasets.ImageFolder(root=setup_dataset_path(), transform=transform)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    print("\nContinuing training for 200 additional epochs...")
    
    for epoch in range(ADDITIONAL_EPOCHS):
        start = time.time()
        gen_losses = []
        disc_losses = []
        
        print(f"\nEpoch {epoch+1}/{ADDITIONAL_EPOCHS}")
        for batch_idx, (real_imgs, labels) in enumerate(dataloader):
            batch_size = real_imgs.size(0)
            real_imgs = real_imgs.to(device)
            labels = labels.to(device)
            valid = torch.ones(batch_size, 1, device=device)
            fake = torch.zeros(batch_size, 1, device=device)

            # Train Generator
            generator.zero_grad()
            z = torch.randn(batch_size, LATENT_DIM, device=device)
            gen_imgs = generator(z, labels)
            validity = discriminator(gen_imgs, labels)
            g_loss = adversarial_loss(validity, valid)
            g_loss.backward()
            g_optimizer.step()
            gen_losses.append(g_loss.item())

            # Train Discriminator
            discriminator.zero_grad()
            real_pred = discriminator(real_imgs, labels)
            fake_pred = discriminator(gen_imgs.detach(), labels)
            real_loss = adversarial_loss(real_pred, valid)
            fake_loss = adversarial_loss(fake_pred, fake)
            d_loss = (real_loss + fake_loss) / 2
            d_loss.backward()
            d_optimizer.step()
            disc_losses.append(d_loss.item())

            if batch_idx % 50 == 0:
                print(f"Batch {batch_idx}/{len(dataloader)} - "
                      f"G_loss: {g_loss.item():.4f}, D_loss: {d_loss.item():.4f}")

        print(f"Time for epoch: {time.time()-start:.2f} sec")
        print(f"Generator loss: {np.mean(gen_losses):.4f}")
        print(f"Discriminator loss: {np.mean(disc_losses):.4f}")

        # Save samples periodically
        if (epoch + 1) % 10 == 0:
            with torch.no_grad():
                z = torch.randn(BATCH_SIZE, LATENT_DIM, device=device)
                labels = torch.arange(NUM_CLASSES).repeat(BATCH_SIZE // NUM_CLASSES + 1)[:BATCH_SIZE].to(device)
                gen_imgs = generator(z, labels)
                save_path = f'continued_samples_epoch_{epoch + 1}.png'
                vutils.save_image(gen_imgs, save_path, nrow=NUM_CLASSES, normalize=True)
                print(f"Generated images saved to {save_path}")
            
            # Save model checkpoint
            torch.save(generator.state_dict(), f'generator_continued_epoch_{epoch + 1}.pth')

    # Save final model
    torch.save(generator.state_dict(), 'generator_continued_final.pth')
    print("Training completed. Final model saved as 'generator_continued_final.pth'")

if __name__ == "__main__":
    continue_training()