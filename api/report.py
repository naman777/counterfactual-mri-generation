"""Render a deterministic research report card from one completed result."""
import base64
import io
from PIL import Image, ImageDraw, ImageFont


def font(size, bold=False):
    name = "C:/Windows/Fonts/seguisb.ttf" if bold else "C:/Windows/Fonts/segoeui.ttf"
    try:
        return ImageFont.truetype(name, size)
    except OSError:
        return ImageFont.load_default(size=size)


def report_card(result, kind="png"):
    canvas = Image.new("RGB", (1400, 1430), "#f4f6f8")
    draw = ImageDraw.Draw(canvas)
    draw.text((50, 40), "MR-one / Counterfactual evidence", fill="#192b39", font=font(32, True))
    draw.text((50, 93), "RESEARCH USE ONLY. NOT FOR DIAGNOSIS.", fill="#087f73", font=font(18, True))
    req, metrics = result["request"], result["metrics"]
    source = result["source"]
    draw.text((50, 135), f"{source['patient']} | slice {source['instance']} | {req['model']} ({result['weights']})", fill="#536b78", font=font(21))
    draw.text((50, 169), f"t={req['timestep']} | w={req['guidance']} | seed={req['seed']} | EchoTime fixed: {source['echo_ms']:.2f} ms", fill="#536b78", font=font(19))
    target = result["target"]
    mfg = lambda n: "GE" if n == 0 else "Siemens"
    fs = lambda n: "1.5 T" if n == 0 else "3.0 T"
    draw.text((50, 206), f"{mfg(source['manufacturer'])} / {fs(source['field_strength'])} -> {mfg(target['manufacturer'])} / {fs(target['field_strength'])}", fill="#192b39", font=font(22, True))
    for i, (key, label) in enumerate([("source", "Source"), ("edited", "Counterfactual"), ("difference", "Absolute difference (0-1)"), ("local_ssim", "Local SSIM (display clipped)")]):
        x = 50 + i * 330
        image = Image.open(io.BytesIO(base64.b64decode(metrics["images"][key].split(",")[1]))).convert("RGB")
        canvas.paste(image.resize((310, 310)), (x, 288))
        draw.text((x, 252), label, fill="#536b78", font=font(17))
    draw.text((50, 625), f"SSIM {metrics['ssim']:.4f}     MAD {metrics['mad']:.5f}     HF energy {metrics['hf_after']:.6f}", fill="#192b39", font=font(25, True))
    draw.text((50, 681), "Target probability: source -> edited", fill="#536b78", font=font(20))
    for j, judge in enumerate(result["judges"]):
        y = 728 + j * 95
        draw.text((50, y), judge["name"], fill="#192b39", font=font(20, True))
        for a, (axis, target_index) in enumerate([("mfg", target["manufacturer"]), ("fs", target["field_strength"])]):
            x = 440 + a * 450
            before, after = (judge[axis][i][target_index] for i in (0, 1))
            draw.text((x, y), f"{axis}: {before*100:.1f}% -> {after*100:.1f}%", fill="#536b78", font=font(19))
            draw.rectangle((x, y+34, x+350, y+40), fill="#dfe7e9")
            draw.rectangle((x, y+34, x+350*after, y+40), fill="#087f73")
    y = 1120
    words, line = result["verdict"]["text"].split(), ""
    for word in words:
        if len(line + word) > 100:
            draw.text((50, y), line, fill="#775e2e", font=font(21)); y += 31; line = ""
        line += word + " "
    draw.text((50, y), line, fill="#775e2e", font=font(21))
    draw.text((50, 1260), "Source: " + result["provenance"], fill="#637c87", font=font(15))
    draw.text((50, 1290), "Result: " + result["id"], fill="#637c87", font=font(15))
    draw.text((50, 1330), "Independent agreement and SSIM do not establish clinical validity. Export JSON for complete metrics.", fill="#637c87", font=font(17))
    output = io.BytesIO()
    canvas.save(output, format="PDF" if kind == "pdf" else "PNG", resolution=140.)
    return output.getvalue()
