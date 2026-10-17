import numpy as np
from scipy.ndimage import laplace
from skimage.metrics import structural_similarity

from api.data import image_url


def verdict(ssim, judges, target_mfg, target_fs, changed):
    def hit(j):
        return all(int(np.argmax(j[k][1])) == (target_mfg if k == "mfg" else target_fs) for k in changed)
    independent = [j for j in judges if j["independent"]]
    count = sum(hit(j) for j in independent) if changed else 0
    if ssim < .5:
        text = "Low structural similarity: treat as a regeneration, not an anatomy-preserving edit."
    elif not changed:
        text = "Null edit: source conditions preserved. Compare with the target edit using the same seed."
    elif count == 3:
        text = "Confirmed by 3/3 independent judges. Classifier agreement does not establish clinical validity."
    elif count == 0 and hit(judges[0]):
        text = "Only the original predictor confirms the target: possible shortcut cue."
    else:
        text = f"Confirmed by {count}/3 independent judges. The requested shift is not consistently verified."
    return dict(text=text, confirmed=count, total=len(independent), low_ssim=ssim < .5)


def analyse(before, after):
    score, local = structural_similarity(before, after, data_range=1., win_size=7, full=True)
    diff = np.abs(after - before)
    edges = np.linspace(0, 1, 33)
    # A shared source mask excludes black background without hiding new changes.
    foreground = before > .01
    hb, _ = np.histogram(before[foreground], bins=edges)
    ha, _ = np.histogram(after[foreground], bins=edges)
    return dict(ssim=float(score), mad=float(diff.mean()),
                hf_before=float(np.mean(laplace(before) ** 2)), hf_after=float(np.mean(laplace(after) ** 2)),
                histogram_note="Source foreground only (source intensity > 0.01); same mask for both images.",
                histogram=[dict(intensity=float((edges[i] + edges[i+1])/2), source=int(hb[i]), edited=int(ha[i])) for i in range(32)],
                images=dict(source=image_url(before), edited=image_url(after), difference=image_url(diff, True),
                            local_ssim=image_url(np.clip(local, 0, 1))))
