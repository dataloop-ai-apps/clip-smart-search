import torch
import clip
from PIL import Image

from sklearn.metrics.pairwise import cosine_similarity


device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

image = preprocess(Image.open(r"E:\TypesExamples\389b35df-25e8-4290-899c-4aad9d4e266f-NUP_185453_0043.jpg")).unsqueeze(0).to(device)
# image = preprocess(Image.open(r"C:\Users\Shabtay\Downloads\ec4Ao4s.png")).unsqueeze(0).to(device)
text = clip.tokenize(["a photo of a man",
                      "a photo of a woman",
                      "a photo of an ear",
                      "alien running in a field",
                      "a portrait of an astronaut with the American flag"]).to(device)

with torch.no_grad():
    image_features = model.encode_image(image)
    text_features = model.encode_text(text)

    logits_per_image, logits_per_text = model(image, text)
    probs = logits_per_image.softmax(dim=-1).cpu().numpy()

# Pick the top 5 most similar labels for the image
image_features /= image_features.norm(dim=-1, keepdim=True)
text_features /= text_features.norm(dim=-1, keepdim=True)
similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
values, indices = similarity[0].topk(5)


result = cosine_similarity(image_features.cpu().numpy(), text_features.cpu().numpy())

print("Label probs:", probs)  # prints: [[0.9927937  0.00421068 0.00299572]]
