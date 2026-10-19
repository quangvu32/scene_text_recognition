from PIL import ImageDraw
import torch
import torch.nn as nn
from torchvision import transforms
import timm
from ultralytics import YOLO
import gradio as gr

text_det_model_path = './best results/scene detection/best.pt'
text_rec_model_path = './best results/text recognition/best_3.pt'

yolo_model = YOLO(text_det_model_path)

class CRNN(nn.Module):
    def __init__(
        self, vocab_size, hidden_size, n_layers, dropout=0.2, unfreeze_layers=3
    ):
        super(CRNN, self).__init__()

        backbone = timm.create_model("resnet152", in_chans=1, pretrained=True)
        modules = list(backbone.children())[:-2]
        modules.append(nn.AdaptiveAvgPool2d((1, None)))
        self.backbone = nn.Sequential(*modules)

        # Unfreeze the last few layers
        for parameter in self.backbone[-unfreeze_layers:].parameters():
            parameter.requires_grad = True

        self.mapSeq = nn.Sequential(
            nn.Linear(2048, 512), nn.ReLU(), nn.Dropout(dropout)
        )

        self.gru = nn.GRU(
            512,
            hidden_size,
            n_layers,
            bidirectional=True,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0,
        )
        self.layer_norm = nn.LayerNorm(hidden_size * 2)

        self.out = nn.Sequential(
            nn.Linear(hidden_size * 2, vocab_size), nn.LogSoftmax(dim=2)
        )

    @torch.autocast(device_type="cuda")
    def forward(self, x):
        x = self.backbone(x)
        x = x.permute(0, 3, 1, 2)
        x = x.view(x.size(0), x.size(1), -1)  # Flatten the feature map
        x = self.mapSeq(x)
        x, _ = self.gru(x)
        x = self.layer_norm(x)
        x = self.out(x)
        x = x.permute(1, 0, 2)  # Based on CTC

        return x
        
chars = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz²³µ¹¼½¾ÀÁÂÃÄÅÆÇÈÉÊËÌÍÎÏÑÒÓÔÕÖØÙÚÛÜÝßàáâãäåæçèéêëìíîïñòóôõöøùúûüýÿĀāĂăĄąĆćċČčĐĒēĖėęĚěġĩĪİıĺĽŁŃńňŉŌōŎőŒŘřŚśşŠšŢţŤũŪūŮůŰźŻżŽžƃƉƠơƬƯưǏǒǪǵȂȇȊȍȎȏȓȕȗțȮəɮʃʰˉˢˣΔΡΣΦΩίαβελνπστϋабвгдежиклмнопрстуъыьэײبرل٢٣ڮڽۊᴐᴰᴱᴴᴵᴸᴹᴺᵀᵃᵈᵉᵐᵒᵗᵛᶜᶻᶾḂḎḕḢḻṁṅṫẏẢẤẦầẬẮẶẾỀềỆọỐốồỘỢό⁰⁵⁷⁸⁹ⁿ₁₃₄ₓₘₙₜℇℒΩℱ⅓⅔ⅠⅡⅢⅣⅤⅥⅦⅧⅨⅩⅪⅫ①②❶➀➁➂➃➄➅➋➌➍➎ⱽ卐ꝋ𝕵-'
vocab_size = len(chars)
char_to_idx = {char: idx + 1 for idx , char in enumerate(sorted(chars))}
idx_to_char = {idx: char for char, idx in char_to_idx.items()}

hidden_size = 256
n_layers = 3
dropout_prob = 0.2
unfreeze_layers=3
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model_path = 'checkpoint.pt'

crnn_model = CRNN(
    vocab_size = vocab_size,
    hidden_size = hidden_size,
    n_layers = n_layers,
    dropout = dropout_prob,
    unfreeze_layers = unfreeze_layers
).to(device)

checkpoint = torch.load(text_rec_model_path) # Text Recognition model
crnn_model.load_state_dict(checkpoint['model_state_dict'])

def decode(encoded_sequences, idx_to_char, blank_char='-'):
    decoded_sequences = []

    for seq in encoded_sequences:
        decoded_label = []
        prev_char = None # To track the previous character

        for token in seq:
            if token != 0: # Ignore padding (token = 0)
                char = idx_to_char[token.item()]
                # Append the character if it ’s not a blank or the same as the previous character

                if char != blank_char:
                    if char != prev_char or prev_char == blank_char:
                        decoded_label.append(char)
                prev_char = char # Update previous character

        decoded_sequences.append("".join(decoded_label))
    print(f"From {encoded_sequences} to {decoded_sequences}")
    return decoded_sequences

def text_detection(img_path, text_det_model) :
    text_det_results = text_det_model(img_path, verbose = False)[0]
    
    bboxes = text_det_results.boxes.xyxy.tolist()
    classes = text_det_results.boxes.cls.tolist()
    names = text_det_results.names
    confs = text_det_results.boxes.conf.tolist()
    
    return bboxes, classes, names, confs

def text_recognition(img , data_transforms , text_reg_model , idx_to_char ,device):
    transformed_image = data_transforms(img)
    transformed_image = transformed_image.unsqueeze(0).to(device)
    text_reg_model.eval()
    with torch.no_grad() :
        logits = text_reg_model(transformed_image).detach().cpu().log_softmax(2)
    text = decode(logits.permute(1, 0, 2).argmax(2), idx_to_char)
    
    return text

val_transform = transforms.Compose([
    transforms.Resize((100, 420)),
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

def sort_boxes(bboxes, classes, confs, y_threshold=20):
    """
    Sort bounding boxes top-to-bottom, left-to-right.
    y_threshold: Maximum vertical difference (in pixels) to group boxes into the same line.
    """
    if not bboxes:
        return [], [], []

    combined = list(zip(bboxes, classes, confs))
    combined.sort(key=lambda item: item[0][1])

    rows = []
    current_row = [combined[0]]

    for item in combined[1:]:
        if abs(item[0][1] - current_row[0][0][1]) <= y_threshold:
            current_row.append(item)
        else:
            rows.append(current_row)
            current_row = [item]
    if current_row:
        rows.append(current_row)
    sorted_combined = []
    for row in rows:
        row.sort(key=lambda item: item[0][0])
        sorted_combined.extend(row)
    sorted_bboxes = [item[0] for item in sorted_combined]
    sorted_classes = [item[1] for item in sorted_combined]
    sorted_confs = [item[2] for item in sorted_combined]

    return sorted_bboxes, sorted_classes, sorted_confs

def predict(pil_img):
    if pil_img is None:
        return None, "Please update an image"

    # 1. Detection via YOLO
    results = yolo_model(pil_img, verbose=False)[0]
    
    bboxes = results.boxes.xyxy.tolist()
    classes = results.boxes.cls.tolist()
    confs = results.boxes.conf.tolist()
    names = results.names

    bboxes, classes, confs = sort_boxes(
        bboxes, classes, confs, y_threshold=20
    )

    annotated_img = pil_img.copy()
    draw = ImageDraw.Draw(annotated_img)
    
    details_str = []

    # 2. Loop through detections & Recognize text
    for idx, (bbox, cls, conf) in enumerate(zip(bboxes, classes, confs), 1):
        x1, y1, x2, y2 = bbox
        class_name = names[int(cls)]

        cropped_crop = pil_img.crop((x1, y1, x2, y2))
        recognized_text = text_recognition(cropped_crop, val_transform, crnn_model, idx_to_char, device)

        draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
        label_text = f"#{idx} {class_name}: {recognized_text}"
        draw.text((x1, max(0, y1 - 15)), label_text, fill="red")

        details_str.append(f"[{idx}] {recognized_text}")

    output_text = "\n".join(details_str) if details_str else "Text undetected."
    return annotated_img, output_text

# ================= Gradio app =================
def main():
    with gr.Blocks(title="Scene Text Detection & OCR") as demo:
        gr.Markdown("## Scene Text Detection & Recognition (YOLO + CRNN)")

        with gr.Row():
            with gr.Column():
                image_input = gr.Image(type="pil", label="Upload Image")
                submit_btn = gr.Button("Detect & Recognize", variant="primary")

            with gr.Column():
                image_output = gr.Image(type="pil", label="Detected Image")
                text_output = gr.Textbox(
                    label="Detection Details",
                    lines=12
                )

        submit_btn.click(
            fn=predict,
            inputs=[image_input],
            outputs=[image_output, text_output]
        )

    demo.launch(share=True)

if __name__ == "__main__":
    main()


