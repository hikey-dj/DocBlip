# app.py — Iterative Document QA with InstructBLIP + Gradio
import torch, math
from transformers import (
    InstructBlipProcessor, 
    InstructBlipForConditionalGeneration,
    OwlViTProcessor,
    OwlViTForObjectDetection
)
from PIL import Image
import pytesseract
import gradio as gr
import torch.nn.functional as F
import re

# Point pytesseract to the Tesseract-OCR installation
pytesseract.pytesseract.tesseract_cmd = r"/usr/bin/tesseract"

# ---------- CONFIG ----------
# Choose model: "Salesforce/instructblip-flan-t5-xl" is small and simple.
MODEL_NAME = "Salesforce/instructblip-flan-t5-xl"
# or use "Salesforce/instructblip-vicuna-7b" if you already handle device_map/accelerate.
MAX_NEW_TOKENS = 256
NUM_BEAMS = 3

# ---------- LOAD MODEL ----------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

processor = InstructBlipProcessor.from_pretrained(MODEL_NAME)
model = InstructBlipForConditionalGeneration.from_pretrained(
    MODEL_NAME,
    device_map={"": "cuda:0"},              # auto places layers on GPU(s)
    load_in_8bit=True,
    dtype=torch.float16,      # half precision
)

# Load OWL-ViT for region detection
print("Loading OWL-ViT for region detection...")
owl_processor = OwlViTProcessor.from_pretrained("google/owlvit-base-patch32")
owl_model = OwlViTForObjectDetection.from_pretrained("google/owlvit-base-patch32")
if device.type == "cuda":
    owl_model = owl_model.to(device)
owl_model.eval()

# ---------- GENERATION KW ----------
GEN_KW = dict(max_new_tokens=MAX_NEW_TOKENS, num_beams=NUM_BEAMS, do_sample=False)

# ---------- HELPERS ----------
def decode_generated(outputs):
    return processor.batch_decode(outputs, skip_special_tokens=True)[0].strip()

def avg_logprob_from_generate(out):
    # out: return_dict_in_generate=True, output_scores=True
    if not getattr(out, "scores", None):
        return None
    scores = out.scores
    full_sequence = out.sequences[0]  # full generated sequence
    # The generated tokens are the last len(scores) tokens
    generated_tokens = full_sequence[-len(scores):]
    probs = []
    for i, logits in enumerate(scores):
        try:
            # Take the max probability across beams for the token
            softmax_probs = F.softmax(logits, dim=-1)
            max_probs = softmax_probs.max(dim=0)[0]  # max over batch/beam dimension
            p = max_probs[generated_tokens[i].item()].item()
        except IndexError:
            p = 1e-12  # fallback if token id out of bounds
        probs.append(max(p, 1e-12))
    return sum(math.log(p) for p in probs) / max(len(probs), 1)

def ocr_boxes(image: Image.Image, conf_thresh=5):
    data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
    boxes=[]
    for i, txt in enumerate(data['text']):
        if not txt.strip():
            continue
        try:
            conf = float(data['conf'][i])
        except:
            conf = -1.0
        if conf < conf_thresh:
            continue
        boxes.append({
            "text": txt,
            "left": int(data['left'][i]),
            "top": int(data['top'][i]),
            "width": int(data['width'][i]),
            "height": int(data['height'][i]),
            "conf": conf
        })
    return boxes

def crop_image(image, box, pad=6):
    left = max(0, box['left']-pad)
    top  = max(0, box['top']-pad)
    right = min(image.width, box['left']+box['width']+pad)
    bottom = min(image.height, box['top']+box['height']+pad)
    return image.crop((left, top, right, bottom))

def make_sliding_line_candidates(ocr_boxes_list, image, max_crops=3,
                                 line_tol=None, stretch_to_right=True):
    """
    Build sliding-window candidates: for each detected line group, create boxes for
    windows: [i], [i-1,i], [i,i+1], [i-1,i,i+1] (centered on i). 
    Optionally stretch boxes to right edge to catch right-aligned amounts.
    Returns list of boxes: {'left','top','width','height','text','conf'} sorted by conf desc.
    """
    if not ocr_boxes_list:
        return []

    H, W = image.height, image.width
    if line_tol is None:
        line_tol = max(8, int(0.01 * H))

    # 1) sort words and group into rough lines
    ocr_sorted = sorted(ocr_boxes_list, key=lambda b: (b['top'], b['left']))
    lines = []
    current = []
    last_top = None
    for b in ocr_sorted:
        if last_top is None or abs(b['top'] - last_top) <= line_tol:
            current.append(b)
            if last_top is None:
                last_top = b['top']
        else:
            lines.append(current)
            current = [b]
            last_top = b['top']
    if current:
        lines.append(current)

    # 2) build compact line objects
    line_objs = []
    for grp in lines:
        left = min(g['left'] for g in grp)
        top = min(g['top'] for g in grp)
        right = max(g['left'] + g['width'] for g in grp)
        bottom = max(g['top'] + g['height'] for g in grp)
        text = " ".join(g['text'] for g in grp).strip()
        conf = sum(g.get('conf', 0) for g in grp) / max(len(grp), 1)
        line_objs.append({
            'left': left, 'top': top, 'right': right, 'bottom': bottom,
            'width': right - left, 'height': bottom - top, 'text': text, 'conf': conf
        })

    # 3) sliding windows centered on each line
    candidate_boxes = []
    n = len(line_objs)
    for i in range(n):
        windows = [
            (i, i),        # just line i
            (max(0, i-1), i),  # i-1 .. i
            (i, min(n-1, i+1)),# i .. i+1
            (max(0, i-1), min(n-1, i+1))  # i-1 .. i+1
        ]
        for a, b in windows:
            # merge lines a..b into one bbox
            left = min(line_objs[j]['left'] for j in range(a, b+1))
            top = min(line_objs[j]['top'] for j in range(a, b+1))
            right = max(line_objs[j]['right'] for j in range(a, b+1))
            bottom = max(line_objs[j]['bottom'] for j in range(a, b+1))
            text = " ".join(line_objs[j]['text'] for j in range(a, b+1)).strip()
            conf = sum(line_objs[j]['conf'] for j in range(a, b+1)) / (b - a + 1)

            # optional: stretch to the right edge to catch right-aligned amounts
            if stretch_to_right:
                right = W

            # small expansion so numbers just outside line are captured
            pad_up = int((bottom - top) * 4)
            pad_down = int((bottom - top) * 2)
            y1 = max(0, top - pad_up)
            y2 = min(H, bottom + pad_down)
            x1 = max(0, left)
            x2 = min(W, right)

            # normalize and save
            candidate_boxes.append({
                'left': int(x1), 'top': int(y1),
                'width': int(x2 - x1), 'height': int(y2 - y1),
                'text': text, 'conf': conf
            })

    # 4) deduplicate almost-identical boxes (by intersection-over-union)
    kept = []
    def iou(a,b):
        xa1, ya1, xa2, ya2 = a['left'], a['top'], a['left']+a['width'], a['top']+a['height']
        xb1, yb1, xb2, yb2 = b['left'], b['top'], b['left']+b['width'], b['top']+b['height']
        ix1, iy1 = max(xa1, xb1), max(ya1, yb1)
        ix2, iy2 = min(xa2, xb2), min(ya2, yb2)
        if ix2 <= ix1 or iy2 <= iy1:
            return 0.0
        inter = (ix2-ix1) * (iy2-iy1)
        union = (xa2-xa1)*(ya2-ya1) + (xb2-xb1)*(yb2-yb1) - inter
        return inter / (union + 1e-9)
    candidate_boxes = sorted(candidate_boxes, key=lambda x: -x['conf'])
    for c in candidate_boxes:
        too_similar = False
        for k in kept:
            if iou(c,k) > 0.9:
                too_similar = True
                break
        if not too_similar:
            kept.append(c)
        if len(kept) >= max_crops:
            break

    return kept

# small helper: simple token overlap score between question and OCR text
def ocr_relevance_score(question, ocr_text):
    qtokens = set(re.findall(r"[A-Za-z0-9]+", question.lower()))
    otokens = set(re.findall(r"[A-Za-z0-9]+", ocr_text.lower()))
    if not qtokens or not otokens:
        return 0.0
    inter = qtokens.intersection(otokens)
    return len(inter) / len(qtokens)   # fraction of question tokens covered

# ---------- AGENT (compact) ----------
def iterative_doc_agent(image: Image.Image, question: str, max_crops=3):
    # 0) initial observation
    print("Generating initial observation...")

    obs_prompt = ("Observe the image and write a detailed paragraph describing the complete image.")
    inputs = processor(images=image, text=obs_prompt, return_tensors="pt")
    if device.type=="cuda":
        inputs = {k:v.to(device) for k,v in inputs.items()}
    with torch.no_grad():
        out_obs = model.generate(**inputs, **GEN_KW, return_dict_in_generate=True, output_scores=True)
    observation = decode_generated(out_obs.sequences)
    observation = observation.replace(obs_prompt, "").strip()
    print("Observation:", observation)


     # --- OCR-driven region selection (replacement for model keyword generation + OWL) ---
    print("Extracting OCR boxes for region candidates...")
    ocr_all = ocr_boxes(image, conf_thresh=30)  # uses your existing function

    candidates = make_sliding_line_candidates(ocr_all, image, max_crops=max_crops, stretch_to_right=True)

    # ensure candidates is sliced to max_crops later (compatible with downstream code)
    # (existing code sorts and truncates candidates:)
    candidates = sorted(candidates, key=lambda x: -x['conf'])[:max_crops]
    print(f"Selected {len(candidates)} OCR-based candidate regions.")

    # 2) crop & query candidates
    answers = []
    if not candidates:
        # fallback: use whole image (existing behavior)
        crops = [("whole-image", image)]
    else:
        # prepare crops for all candidates
        crops = []
        for i, c in enumerate(candidates):
            crop_img = image.crop((c['left'], c['top'], c['left']+c['width'], c['top']+c['height']))
            crops.append((f"cand_{i}", crop_img, c))

    # Weights (tunable)
    W_MODEL = 0.6
    W_OCR   = 0.25

    for item in crops:
        if item[0] == "whole-image":
            crop = item[1]
            cmeta = None
        else:
            _, crop, cmeta = item

        # save crop for debug (optional)
        crop.save(f"debug_crop_{item[0]}.png")

        # 2) OCR the crop and compute OCR-based signals
        crop_ocr_boxes = ocr_boxes(crop, conf_thresh=20)   # lower thresh for small crops
        crop_ocr_text = " ".join([b['text'] for b in crop_ocr_boxes]).strip()

        ocr_rel = ocr_relevance_score(question, crop_ocr_text)    # 0..1

        # 1) query model on this crop
        crop_prompt = question
        inputs = processor(images=crop, text=crop_prompt+ "\nOCR: " + crop_ocr_text, return_tensors="pt")
        if device.type=="cuda":
            inputs = {k:v.to(device) for k,v in inputs.items()}
        with torch.no_grad():
            out = model.generate(**inputs, **GEN_KW, return_dict_in_generate=True, output_scores=True)
        decoded = decode_generated(out.sequences)
        model_conf = avg_logprob_from_generate(out)
        if model_conf is None:
            model_conf = -10.0  # fallback small value

        combined_score = W_MODEL * model_conf + W_OCR * ocr_rel

        answers.append({
            "text": decoded,
            "conf": model_conf,
            "model_score": model_conf,
            "ocr_text": crop_ocr_text,
            "ocr_rel": ocr_rel,
            "combined_score": combined_score,
            "cand_meta": cmeta
        })

    # select best by combined_score (tie-breaker: model_conf)
    if answers:
        best = max(answers, key=lambda x: (x['combined_score'], x['conf']))
    else:
        # ultimate fallback: run on whole image
        inputs = processor(images=image, text=question, return_tensors="pt")
        if device.type=="cuda":
            inputs = {k:v.to(device) for k,v in inputs.items()}
        with torch.no_grad():
            out = model.generate(**inputs, **GEN_KW, return_dict_in_generate=True, output_scores=True)
        best = {"text": decode_generated(out.sequences), "conf": avg_logprob_from_generate(out),
                "ocr_text": " ".join([b['text'] for b in ocr_boxes(image, conf_thresh=20)]),
                "combined_score": 0.0, "cand_meta": None}
        
    print("Best answer OCR text:", best['ocr_text'])

    # return best answer (and optionally debug info)
    return f"{best['text']}\n\n{best['ocr_text']}"

# ---------- GRADIO UI ----------
def ask_entry(image, question, crops=3):
    if image is None or question.strip()=="":
        return "Upload an image and enter a question."
    pil = Image.fromarray(image) if not isinstance(image, Image.Image) else image
    return iterative_doc_agent(pil, question, max_crops=crops)

demo = gr.Interface(
    fn=ask_entry,
    inputs=[
        gr.Image(type="numpy", label="Document Image"),
        gr.Textbox(lines=1, label="Question (e.g., 'What is the total amount?')"),
        gr.Slider(minimum=1, maximum=5, value=3, step=1, label="Candidate crops to try")
    ],
    outputs="text",
    title="Iterative Document QA (InstructBLIP)",
    description="Prototype: observation → OCR/locate → crop & re-query. Prefer OCR for numeric answers."
)

if __name__ == "__main__":
    demo.launch(share=True)