import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
from loss import wasserstein_loss, GradientPenalty
from stackGAN_model import StageIGenerator, StageIDiscriminator, StageIIGenerator, StageIIDiscriminator
from dataloader import COCO_Dataset, collate_fn, transform, train_annotations, train_dir, val_annotations, val_dir
from text_embedding import TextEmbeddingModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


train_dataset = COCO_Dataset(train_annotations, train_dir, transform)
val_dataset = COCO_Dataset(val_annotations, val_dir, transform)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn, num_workers=2)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn, num_workers=2)

embedding_dim = 128
hidden_dim = 256
noise_dim = 100
image_size = 64
channels = 3
epochs = 100
lr = 0.0001
beta1 = 0.5
beta2 = 0.999
lambda_gp = 10

stage1_gen = StageIGenerator(hidden_dim, noise_dim, image_size, channels).to(device)
stage1_disc = StageIDiscriminator(image_size, channels, hidden_dim).to(device)
stage2_gen = StageIIGenerator(hidden_dim, noise_dim, image_size * 4, channels).to(device)
stage2_disc = StageIIDiscriminator(image_size * 4, channels, hidden_dim).to(device)
text_embedding_model = TextEmbeddingModel(len(vocab), embedding_dim, hidden_dim).to(device)

optimizer_g1 = optim.Adam(stage1_gen.parameters(), lr=lr, betas=(beta1, beta2))
optimizer_d1 = optim.Adam(stage1_disc.parameters(), lr=lr, betas=(beta1, beta2))
optimizer_g2 = optim.Adam(stage2_gen.parameters(), lr=lr, betas=(beta1, beta2))
optimizer_d2 = optim.Adam(stage2_disc.parameters(), lr=lr, betas=(beta1, beta2))

train_losses = {"d1_loss": [], "g1_loss": [], "d2_loss": [], "g2_loss": []}
val_losses = {"d1_loss": [], "g1_loss": [], "d2_loss": [], "g2_loss": []}

def validate(loader, models, text_embedding_model, device):
    stage1_gen, stage1_disc, stage2_gen, stage2_disc = models
    stage1_gen.eval()
    stage1_disc.eval()
    stage2_gen.eval()
    stage2_disc.eval()

    d1_loss_val, g1_loss_val, d2_loss_val, g2_loss_val = 0, 0, 0, 0
    with torch.no_grad():
        for images, captions in loader:
            batch_size = images.size(0)
            images, captions = images.to(device), captions.to(device)

            text_embedding = text_embedding_model(captions).to(device)
            noise = torch.randn(batch_size, noise_dim, device=device)

            fake_images = stage1_gen(text_embedding, noise)
            real_output = stage1_disc(images, text_embedding).view(-1)
            fake_output = stage1_disc(fake_images, text_embedding).view(-1)
            gradient_penalty_1 = GradientPenalty(stage1_disc, images, fake_images, text_embedding, device)
            d1_loss = wasserstein_loss(real_output, fake_output) + lambda_gp * gradient_penalty_1
            fake_output = stage1_disc(fake_images, text_embedding).view(-1)
            g1_loss = -torch.mean(fake_output)

            fake_images_2 = stage2_gen(text_embedding, noise)
            real_output_2 = stage2_disc(images, text_embedding).view(-1)
            fake_output_2 = stage2_disc(fake_images_2, text_embedding).view(-1)
            gradient_penalty_2 = GradientPenalty(stage2_disc, images, fake_images_2, text_embedding, device)
            d2_loss = wasserstein_loss(real_output_2, fake_output_2) + lambda_gp * gradient_penalty_2
            fake_output_2 = stage2_disc(fake_images_2, text_embedding).view(-1)
            g2_loss = -torch.mean(fake_output_2)

            d1_loss_val += d1_loss.item()
            g1_loss_val += g1_loss.item()
            d2_loss_val += d2_loss.item()
            g2_loss_val += g2_loss.item()

    return d1_loss_val / len(loader), g1_loss_val / len(loader), d2_loss_val / len(loader), g2_loss_val / len(loader)

for epoch in range(epochs):
    loop = tqdm(train_loader, leave=True)
    for images, captions in loop:
        batch_size = images.size(0)
        images, captions = images.to(device), captions.to(device)

        real_labels = torch.ones(batch_size, 1).to(device)
        fake_labels = torch.ones(batch_size, 1).to(device)

        ### Train Stage-1 Discriminator ###
        optimizer_d1.zero_grad()
        text_embedding = text_embedding_model(captions).to(device)
        real_output = stage1_disc(images, text_embedding).view(-1)
        noise = torch.randn(batch_size, noise_dim, device=device)
        fake_images = stage1_gen(text_embedding, noise)
        fake_output = stage1_disc(fake_images.detach(), text_embedding).view(-1)
        gradient_penalty_1 = GradientPenalty(stage1_disc, images, fake_images, text_embedding, device)
        d1_loss = wasserstein_loss(real_output, fake_output) + lambda_gp * gradient_penalty_1
        d1_loss.backward()
        optimizer_d1.step()

        ### Train Stage-1 Generator ###
        optimizer_g1.zero_grad()
        fake_output = stage1_disc(fake_images, text_embedding).view(-1)
        g1_loss = -torch.mean(fake_output)
        g1_loss.backward()
        optimizer_g1.step()

        ### Train Stage-2 Discriminator ###
        optimizer_d2.zero_grad()
        fake_images_2 = stage2_gen(text_embedding, noise)
        real_output_2 = stage2_disc(images, text_embedding).view(-1)
        fake_output_2 = stage2_disc(fake_images_2.detach(), text_embedding).view(-1)
        gradient_penalty_2 = GradientPenalty(stage2_disc, images, fake_images_2, text_embedding, device)
        d2_loss = wasserstein_loss(real_output_2, fake_output_2) + lambda_gp * gradient_penalty_2
        d2_loss.backward()
        optimizer_d2.step()

        ### Train Stage-2 Generator ###
        optimizer_g2.zero_grad()
        fake_output_2 = stage2_disc(fake_images_2, text_embedding).view(-1)
        g2_loss = -torch.mean(fake_output_2)
        g2_loss.backward()
        optimizer_g2.step()

        ### Update progress bar ###
        loop.set_description(f"Epoch [{epoch+1}/{epochs}]")
        loop.set_postfix(d1_loss=d1_loss.item(), g1_loss=g1_loss.item(), d2_loss=d2_loss.item(), g2_loss=g2_loss.item())

    train_losses["d1_loss"].append(d1_loss.item())
    train_losses["g1_loss"].append(g1_loss.item())
    train_losses["d2_loss"].append(d2_loss.item())
    train_losses["g2_loss"].append(g2_loss.item())

    d1_loss_val, g1_loss_val, d2_loss_val, g2_loss_val = validate(val_loader, 
        (stage1_gen, stage1_disc, stage2_gen, stage2_disc), text_embedding_model, device)

    val_losses["d1_loss"].append(d1_loss_val)
    val_losses["g1_loss"].append(g1_loss_val)
    val_losses["d2_loss"].append(d2_loss_val)
    val_losses["g2_loss"].append(g2_loss_val)

    print(f"Validation losses: d1_loss={d1_loss_val}, g1_loss={g1_loss_val}, d2_loss={d2_loss_val}, g2_loss={g2_loss_val}")

### Plotting the training and validation losses ###
plt.figure(figsize=(10, 5))
plt.plot(train_losses["d1_loss"], label="Train D1 Loss")
plt.plot(val_losses["d1_loss"], label="Val D1 Loss")
plt.plot(train_losses["g1_loss"], label="Train G1 Loss")
plt.plot(val_losses["g1_loss"], label="Val G1 Loss")
plt.plot(train_losses["d2_loss"], label="Train D2 Loss")
plt.plot(val_losses["d2_loss"], label="Val D2 Loss")
plt.plot(train_losses["g2_loss"], label="Train G2 Loss")
plt.plot(val_losses["g2_loss"], label="Val G2 Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.title("Training and Validation Losses")
plt.show()
