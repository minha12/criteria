import hydra
import torch
from criteria import load_model
from criteria.id_loss import IdLost
from criteria.clip_loss import CLIPLoss
from criteria.lpips.lpips import LPIPS

def create_test_data():
    # Create random face images (batch_size=2, channels=3, height=256, width=256)
    y_hat = torch.randn(2, 3, 256, 256).to("cuda") 
    y = torch.randn(2, 3, 256, 256).to("cuda")
    
    # Normalize to [0,1] range to simulate real images
    y_hat = (y_hat - y_hat.min()) / (y_hat.max() - y_hat.min())
    y = (y - y.min()) / (y.max() - y.min())
    
    return y_hat, y

@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg):
    model = load_model("ir152", cfg)
    print("Model loaded.")

    y_hat, y = create_test_data()

    # Initialize ID loss
    id_loss = IdLost("ir152")
    clip_loss = CLIPLoss()
    lpips_loss = LPIPS()

    # Calculate loss
    loss = id_loss(y_hat, y)
    print(f"ID Loss: {loss.item():.4f}")
    clip_dist = clip_loss.compute_distance(y_hat, y)
    print(f"CLIP Distance: {clip_dist.item():.4f}")
    lpips_dist = lpips_loss(y_hat, y)
    print(f"LPIPS Distance: {lpips_dist.item():.4f}")


if __name__ == "__main__":
    main()
