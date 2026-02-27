from transformers import VitPoseForPoseEstimation
import torch
from FeatureExtractor import FeatureExtractor

teacher = VitPoseForPoseEstimation.from_pretrained("usyd-community/vitpose-base-simple")
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
teacher = teacher.to(device) #type: ignore
teacher.eval()

extractor = FeatureExtractor(teacher, [0, 5, 11])

x = torch.rand(1, 3, 256, 192).to(device)
with torch.no_grad():
    out = teacher(x)

for idx, feat in extractor.features.items():
    print(f"Layer {idx}: {feat.shape}")
