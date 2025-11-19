import torch
from torchvision import models, transforms
from torchvision.utils import draw_bounding_boxes
import torchvision.transforms.functional as F
from PIL import Image
import matplotlib.pyplot as plt

# Choose device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load pretrained model + category names
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights
weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
model = models.detection.fasterrcnn_resnet50_fpn(weights=weights).to(device)
model.eval()

# Get category names (COCO 91 → 80 used)
categories = weights.meta["categories"]

# Load and preprocess image
image_path = "test_images/cat1.png"  # change this
image_path = "test_images/clock1.png"  # change this
image_path = "test_images/book_lamp.png"  # change this
image_path = "test_images/leaf1.png"  # change this
image_path = "test_images/leaf2.png"  # change this
image_path = "test_images/tree1.png"  # change this
img = Image.open(image_path).convert("RGB")
transform = transforms.Compose([transforms.ToTensor()])
img_tensor = transform(img).to(device)

# Run prediction
with torch.no_grad():
    outputs = model([img_tensor])

boxes = outputs[0]["boxes"]
labels = outputs[0]["labels"]
scores = outputs[0]["scores"]

# Filter by confidence threshold
confidence_threshold = 0.6
keep = scores > confidence_threshold
boxes = boxes[keep].cpu()
labels = labels[keep].cpu()
scores = scores[keep].cpu()

# Build labels with object names and confidence
label_texts = [
    f"{categories[l]}: {s:.2f}" for l, s in zip(labels, scores)
]

print(label_texts)

# Draw boxes on image
img_uint8 = (img_tensor.cpu() * 255).byte()
img_with_boxes = draw_bounding_boxes(
    img_uint8, boxes, labels=label_texts, colors="red", width=3
)

# Convert to PIL for saving or display
img_out = F.to_pil_image(img_with_boxes)
#img_out.save("output_with_boxes.jpg")

plt.figure(figsize=(10,10))
plt.imshow(img_out)
plt.axis("off")
plt.title("Detected Objects with Labels")
plt.show()

