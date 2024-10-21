import torch
import clip
from PIL import Image


def use1():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)

    image = preprocess(Image.open("/data5/lailihao/projects/Text-IF/dataset/MSRS_Val/Visible/1.png")).unsqueeze(0).to(
        device)
    text = clip.tokenize(["a diagram", "a dog", "a cat", "a person"]).to(device)

    with torch.no_grad():
        image_features = model.encode_image(image)
        text_features = model.encode_text(text)

        logits_per_image, logits_per_text = model(image, text)
        probs = logits_per_image.softmax(dim=-1).cpu().numpy()

    print("Label probs:", probs)  # prints: [[0.9927937  0.00421068 0.00299572]]


def use2():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load('ViT-B/32', device)

    # Download the dataset
    image_input = preprocess(Image.open("/data5/lailihao/projects/Text-IF/dataset/MSRS_Val/Visible/1.png")).unsqueeze(
        0).to(
        device)
    print(image_input.shape)  # torch.Size([1, 3, 224, 224])

    classes = ["dog", "person", "house", "car"]
    text_inputs = torch.cat([clip.tokenize(f"a photo of a {c}") for c in classes]).to(device)
    print(text_inputs.shape)  # 每段文字被转为长度为77的token,torch.Size([4, 77])

    # Calculate features
    with torch.no_grad():
        image_features = model.encode_image(image_input)
        text_features = model.encode_text(text_inputs)
        print(image_features.shape)  # torch.Size([1, 512])
        print(text_features.shape)  # torch.Size([4, 512]),第一个维度batch

    # Pick the top 5 most similar labels for the image
    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)

    # [1,512] @ [512,4]即计算图像与每一段文字的相似度,然后softmax得到图像与每段文字的相识概率,即可作为分类概率
    similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)

    values, indices = similarity[0].topk(3)

    # Print the result
    print("\nTop predictions:\n")
    for value, index in zip(values, indices):
        print(f"{classes[index]:>16s}: {100 * value.item():.2f}%")


if __name__ == '__main__':
    use1()
    # use2()
