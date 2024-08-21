import torch

from tqdm import tqdm
from torchvision import transforms

from vae.dataset import get_cifar10_dataloaders
from vae.model import VAE


EPOCHS = 30
LR = 0.001
BATCH_SIZE = 16


def train_epoch(model, train_loader, optimizer, device):
    train_loss = 0.0
    train_reconstruction_loss = 0.0
    train_kld = 0.0

    model.train()
    for imgs, targets in tqdm(train_loader, desc="train phase"):
        num = imgs.size(0)
        imgs = imgs.to(device)

        outputs, mu, log_var = model(imgs)
        loss_results = model.loss_function(outputs, imgs, mu, log_var, num)
        loss = loss_results["loss"]
        reconstruction_loss = loss_results["reconstruction_loss"]
        kld = loss_results["kld"]

        train_loss += loss.item() * num
        train_reconstruction_loss += (reconstruction_loss.item() * num)
        train_kld += (kld.item() * num)

        model.zero_grad()

        loss.backward()
        optimizer.step()
    train_loss = train_loss / len(train_loader.sampler)
    train_reconstruction_loss = train_reconstruction_loss / len(train_loader.sampler)
    train_kld = train_kld / len(train_loader.sampler)

    return {
        "train_loss": train_loss,
        "train_reconstruction_loss": train_reconstruction_loss,
        "train_kld": train_kld,
    }


def val_epoch(model, val_loader, optimizer, device):
    val_loss = 0.0
    val_reconstruction_loss = 0.0
    val_kld = 0.0

    model.eval()
    for imgs, targets in tqdm(val_loader, desc="val phase"):
        num = imgs.size(0)
        imgs = imgs.to(device)

        with torch.no_grad():
            outputs, mu, log_var = model(imgs)
            loss_results = model.loss_function(outputs, imgs, mu, log_var, num)
            loss = loss_results["loss"]
            reconstruction_loss = loss_results["reconstruction_loss"]
            kld = loss_results["kld"]

            val_loss += (loss.item() * num)
            val_reconstruction_loss += (reconstruction_loss.item() * num)
            val_kld += (kld.item() * num)

    val_loss = val_loss / len(val_loader.sampler)
    val_reconstruction_loss = val_reconstruction_loss / len(val_loader.sampler)
    val_kld = val_kld / len(val_loader.sampler)

    return {
        "val_loss": val_loss,
        "val_reconstruction_loss": val_reconstruction_loss,
        "val_kld": val_kld,
    }


def main():
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.has_mps or torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print("Using device:", device)

    train_loader, val_loader = get_cifar10_dataloaders(BATCH_SIZE)
    model = VAE(3, 512).to(device)
    params = list(model.parameters())
    optimizer = torch.optim.Adam(params, lr=LR)

    best_val_loss = 1e19
    for i in range(EPOCHS):
        print(f"Train epoch: {i+1}/{EPOCHS}")
        train_results = train_epoch(model, train_loader, optimizer, device)
        print(train_results)
        val_results = val_epoch(model, val_loader, optimizer, device)
        print(val_results)

        val_loss = val_results["loss"]

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(
                {
                    "epoch": i,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict()
                },
                f"checkpoints/vae-epoch-{i}.pt"
            )
            print("Checkpoint was saved")

    z = torch.randn(16, 512)
    z = z.to(device)
    output = model.sample(z)
    output = 0.5 * (output + 1) * 255
    output = output.to(torch.uint8).cpu()
    for i, img_tensor in enumerate(output):
        img = transforms.ToPILImage()(img_tensor)
        img.save(f'output_image_{i}.png')


if __name__ == "__main__":
    main()
