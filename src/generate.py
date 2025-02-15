import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import os

def check_device():
    print("\nDevice Information:")
    print("-" * 50)
    if torch.cuda.is_available():
        print("PyTorch is built with CUDA")
    else:
        print("PyTorch is NOT built with CUDA")

    print("\nAvailable devices:")
    devices = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    for device in devices:
        print(f"- GPU: {device}")
    
    print("\nCurrent device strategy:", "Single GPU" if torch.cuda.device_count() == 1 else "Multiple GPUs")
    
    if torch.cuda.is_available():
        gpu_device = torch.cuda.current_device()
        print(f"GPU Device: {gpu_device}")
        print("GPU Memory info:", torch.cuda.get_device_properties(gpu_device))
    else:
        print("No GPU devices available")
    print("-" * 50)

# Call device check at start
check_device()

# GPU Configuration
if torch.cuda.is_available():
    print('GPU is available')
else:
    print('No GPU available, using CPU')

# Constants (ensure these match the values in train.py)
LATENT_DIM = 100
NUM_CLASSES = 7

# Generator model (same as defined in train.py)
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.label_embedding = nn.Embedding(NUM_CLASSES, LATENT_DIM)
        self.model = nn.Sequential(
            nn.Linear(LATENT_DIM, 512 * 6 * 6),
            nn.BatchNorm1d(512 * 6 * 6),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Unflatten(1, (512, 6, 6)),
            
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

# Load the trained generator
def load_generator(model_path='generator_continued_final.pth'):  # This default is correct
    generator = Generator()
    generator.load_state_dict(torch.load(model_path, map_location=torch.device('cpu'), weights_only=True))
    #generator.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    generator.eval()
    return generator

def generate_face(emotion_label, generator, save_path=None):
    # Create face_emotions directory if it doesn't exist
    if not os.path.exists('face_emotions'):
        os.makedirs('face_emotions')
    
    # Modify save_path to be in face_emotions folder
    if save_path:
        save_path = os.path.join('face_emotions', save_path)
        
    label_dict = {
        'angry': 0, 'disgust': 1, 'fear': 2, 
        'happy': 3, 'neutral': 4, 'sad': 5, 'surprise': 6
    }
    if isinstance(emotion_label, str):
        emotion_label = label_dict[emotion_label.lower()]
    noise = torch.randn(1, 100)
    label = torch.tensor([emotion_label], dtype=torch.long)
    with torch.no_grad():
        generated_image = generator(noise, label)
    generated_image = generated_image * 0.5 + 0.5  # Rescale to [0,1]
    image_np = generated_image.squeeze().numpy()
    if save_path:
        plt.imsave(save_path, image_np, cmap='gray')
        print(f"Generated face saved as {save_path}")
    else:
        plt.imshow(image_np, cmap='gray')
        plt.axis('off')
        plt.show()

if __name__ == "__main__":
    generator = load_generator()
    emotion = input("Enter emotion (angry/disgust/fear/happy/neutral/sad/surprise): ")
    generate_face(emotion, generator, 'face.png')
