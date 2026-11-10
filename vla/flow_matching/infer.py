"""Sample images from the trained flow matching UNet."""

import torch
import matplotlib.pyplot as plt

from train import UNet


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

steps = 250
label_id = 3

model = UNet().to(device)
checkpoint = torch.load("./model.pt", map_location="cpu")
model.load_state_dict(checkpoint)
model.eval()

label = torch.tensor([label_id], dtype=torch.long, device=device)

with torch.no_grad():
    x = torch.randn(1, 1, 28, 28, device=device)
    for i in range(steps):
        t = torch.tensor([i / steps], device=device)
        pred_vt = model(x, t, label)
        x = x + pred_vt * (1.0 / steps)

x = (x + 1) / 2

# Save to file (works in headless environments).
plt.figure(figsize=(2, 2))
plt.axis("off")
plt.imshow(x[0, 0].cpu().numpy(), cmap="gray")
plt.savefig("sample.png", bbox_inches="tight", dpi=150)
print("Saved sample to sample.png")

# Try interactive display (no-op on headless backends).
plt.show()
