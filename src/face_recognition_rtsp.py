import cv2
import time
import datetime
import threading
import numpy as np
from pathlib import Path
from insightface.app import FaceAnalysis
from insightface.utils import face_align

# ════════════════════════════════════════════════════════════
# CW ENERJİ RENK PALETİ
# ════════════════════════════════════════════════════════════
CW_MAVI      = (180, 100, 20)
CW_ACIK_MAVI = (220, 180, 80)
CW_BEYAZ     = (255, 255, 255)
CW_YESIL     = (120, 200, 60)
CW_KIRMIZI   = (60,  60,  200)
CW_KOYU      = (18,  18,  24)

# ════════════════════════════════════════════════════════════
# SABİTLER
# ════════════════════════════════════════════════════════════
DET_SCORE_ESIK  = 0.45
SIMILARITY_ESIK = 0.30
LOG_ARALIK      = 5.0

# ════════════════════════════════════════════════════════════
# ROI — orijinal 2560x1440 koordinatları
# ════════════════════════════════════════════════════════════
ROI_CAM1 = np.array([
    [1360, 440], [42, 284], [18, 1422], [2442, 1424], [1788, 22], [1390, 16]
], np.int32)

ROI_CAM2 = np.array([
    [2542, 1006], [1306, 244], [1104, 240], [1070, 20],
    [480, 14], [14, 858], [12, 1426], [2522, 1418]
], np.int32)

# ════════════════════════════════════════════════════════════
# KAYIT KLASÖRÜ
# ════════════════════════════════════════════════════════════
KAYIT_DIR = Path("recordings")
KAYIT_DIR.mkdir(exist_ok=True)

# ════════════════════════════════════════════════════════════
# LOG DOSYASI
# ════════════════════════════════════════════════════════════
LOG_DOSYA = Path("face_log.txt")
log_file  = open(LOG_DOSYA, "a", encoding="utf-8")
log_file.write(f"\n{'='*50}\n")
log_file.write(f"Oturum basladi: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
log_file.write(f"{'='*50}\n")
log_file.flush()

# ════════════════════════════════════════════════════════════
# MODEL
# ════════════════════════════════════════════════════════════
app = FaceAnalysis(name="buffalo_l", providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
app.prepare(ctx_id=0, det_size=(640, 640))
model_lock = threading.Lock()

# ════════════════════════════════════════════════════════════
# REGISTRY YÜKLEME
# ════════════════════════════════════════════════════════════
registry = {}
faces_db = Path("..\\faces_db")

if faces_db.exists():
    for person_dir in sorted(faces_db.iterdir()):
        if not person_dir.is_dir():
            continue
        name = person_dir.name
        embeddings = []
        for img_path in person_dir.glob("*"):
            if img_path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                continue
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            h, w = img.shape[:2]
            if h <= 150 and w <= 150:
                crop_rgb     = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                crop_resized = cv2.resize(crop_rgb, (112, 112))
                feat = app.models['recognition'].get_feat(crop_resized)
                feat = feat.flatten()
                feat = feat / np.linalg.norm(feat)
                embeddings.append(feat)
            else:
                for f in app.get(img):
                    embeddings.append(f.normed_embedding)

        if embeddings:
            mean_emb = np.mean(embeddings, axis=0)
            registry[name] = mean_emb / np.linalg.norm(mean_emb)
            print(f"Yuklendi: {name} ({len(embeddings)} fotograf)")

print(f"\nToplam {len(registry)} kisi yuklendi. Baslaniyor...\n")


# ════════════════════════════════════════════════════════════
# YARDIMCI FONKSİYONLAR
# ════════════════════════════════════════════════════════════
def roi_scaled(roi_orig, orig_w, orig_h, yeni_w, yeni_h):
    s = roi_orig.astype(np.float32).copy()
    s[:, 0] = s[:, 0] * yeni_w / orig_w
    s[:, 1] = s[:, 1] * yeni_h / orig_h
    return s.astype(np.int32)


def koseli_cerceve(frame, x1, y1, x2, y2, color, kalinlik=2, uzunluk=18):
    for p, q in [((x1,y1),(x1+uzunluk,y1)), ((x1,y1),(x1,y1+uzunluk)),
                 ((x2,y1),(x2-uzunluk,y1)), ((x2,y1),(x2,y1+uzunluk)),
                 ((x1,y2),(x1+uzunluk,y2)), ((x1,y2),(x1,y2-uzunluk)),
                 ((x2,y2),(x2-uzunluk,y2)), ((x2,y2),(x2,y2-uzunluk))]:
        cv2.line(frame, p, q, color, kalinlik)


def roi_ciz(frame, roi_poly, aktif=False):
    t = time.time()
    if aktif:
        alpha = 0.10 + 0.10 * abs(np.sin(t * 3.0))
        renk  = (60, 220, 120)
        kenar = (80, 240, 140)
        kk    = 2
    else:
        alpha = 0.20          # 0.06 → 0.12 yaptık, daha belirgin dolgu
        renk  = CW_MAVI       # CW_ACIK_MAVI yerine koyu mavi — daha az göz yakan
        kenar = CW_ACIK_MAVI
        kk    = 1

    overlay = frame.copy()
    cv2.fillPoly(overlay, [roi_poly], renk)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
    cv2.polylines(frame, [roi_poly], True, kenar, kk)

    for pt in roi_poly:
        cv2.rectangle(frame, (pt[0]-4, pt[1]-4), (pt[0]+4, pt[1]+4), CW_MAVI, -1)
        cv2.rectangle(frame, (pt[0]-4, pt[1]-4), (pt[0]+4, pt[1]+4), kenar, 1)

    if aktif:
        parlak = int(180 + 75 * abs(np.sin(t * 3.0)))
        for pt in roi_poly:
            cv2.circle(frame, tuple(pt), 6, (parlak, 255, parlak), -1)
            cv2.circle(frame, tuple(pt), 8, kenar, 1)


def banner(frame, fps, n_kisi, n_roi, cam_id, kayit=True):
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 56), CW_KOYU, -1)
    cv2.addWeighted(overlay, 0.88, frame, 0.12, 0, frame)
    cv2.line(frame, (0, 56), (w, 56), CW_ACIK_MAVI, 1)

    cv2.putText(frame, "CW",      (12, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.85, CW_ACIK_MAVI, 3)
    cv2.putText(frame, " ENERJI", (46, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.65, CW_BEYAZ, 2)
    cv2.putText(frame, "FACE ID", (14, 54), cv2.FONT_HERSHEY_SIMPLEX, 0.28, CW_ACIK_MAVI, 1)

    cv2.line(frame, (200, 8), (200, 50), (50, 50, 60), 1)
    cv2.putText(frame, f"CAM {cam_id}", (210, 36),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, CW_ACIK_MAVI, 1)
    cv2.line(frame, (280, 8), (280, 50), (50, 50, 60), 1)

    def blok(x, label, deger, renk):
        cv2.putText(frame, label,      (x, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (110,110,120), 1)
        cv2.putText(frame, str(deger), (x, 46), cv2.FONT_HERSHEY_SIMPLEX, 0.70, renk, 2)
        cv2.line(frame, (x+70, 8), (x+70, 50), (50,50,60), 1)

    blok(290, "FPS",        f"{fps:.1f}", CW_BEYAZ)
    blok(370, "REGISTERED", str(n_kisi),  CW_BEYAZ)
    blok(450, "IN ROI",     str(n_roi),   CW_YESIL if n_roi > 0 else (70,70,80))

    # Kayıt göstergesi — sağ üstte nabız atan kırmızı nokta
    if kayit:
        t      = time.time()
        parlak = abs(np.sin(t * 2.0)) > 0.5
        renk_r = (0, 0, 220) if parlak else (0, 0, 120)
        cv2.circle(frame, (w - 24, 20), 7, renk_r, -1)
        cv2.putText(frame, "REC", (w - 52, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 200), 1)


def log_goster(frame, log_list):
    if not log_list:
        return
    h, w = frame.shape[:2]

    satir_h  = 28
    padding  = 10
    baslik_h = 26
    goster   = log_list[-5:]
    kutu_h   = baslik_h + padding + len(goster) * satir_h + padding
    kutu_w   = 340
    x0       = w - kutu_w - 14
    y0       = h - kutu_h - 14

    overlay = frame.copy()
    cv2.rectangle(overlay, (x0, y0), (x0+kutu_w, y0+kutu_h), CW_KOYU, -1)
    cv2.addWeighted(overlay, 0.60, frame, 0.40, 0, frame)

    cv2.rectangle(frame, (x0, y0), (x0+kutu_w, y0+kutu_h), CW_MAVI, 1)
    cv2.rectangle(frame, (x0, y0), (x0+kutu_w, y0+baslik_h), CW_MAVI, -1)
    cv2.putText(frame, "  ACTIVITY LOG", (x0+6, y0+18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, CW_ACIK_MAVI, 1)

    yb = y0 + baslik_h + padding - 2
    for txt, xx in [("SAAT",8),("CAM",72),("KISI",108),("SKOR",270)]:
        cv2.putText(frame, txt, (x0+xx, yb), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (90,90,110), 1)
    cv2.line(frame, (x0+4, yb+4), (x0+kutu_w-4, yb+4), (40,40,55), 1)

    for i, (saat_str, cam_str, isim, skor) in enumerate(goster):
        y = y0 + baslik_h + padding + 16 + i * satir_h

        if i % 2 == 0:
            ov2 = frame.copy()
            cv2.rectangle(ov2, (x0+1, y-14), (x0+kutu_w-1, y+10), (25,25,35), -1)
            cv2.addWeighted(ov2, 0.4, frame, 0.6, 0, frame)

        cv2.putText(frame, saat_str, (x0+8,  y), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (100,110,130), 1)
        cv2.putText(frame, cam_str,  (x0+72, y), cv2.FONT_HERSHEY_SIMPLEX, 0.36, CW_ACIK_MAVI,  1)

        isim_renk = CW_YESIL if isim != "Unknown" else CW_KIRMIZI
        cv2.putText(frame, isim, (x0+108, y), cv2.FONT_HERSHEY_SIMPLEX, 0.40, isim_renk, 1)

        bar_x = x0 + 260
        bar_w = 60
        bar_h = 6
        bar_y = y - 5
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x+bar_w, bar_y+bar_h), (40,40,55), -1)
        dolu     = int(bar_w * min(skor, 1.0))
        bar_renk = CW_YESIL if isim != "Unknown" else CW_KIRMIZI
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x+dolu, bar_y+bar_h), bar_renk, -1)
        cv2.putText(frame, f"{skor:.2f}", (bar_x+bar_w+4, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.32, (110,110,130), 1)


# ════════════════════════════════════════════════════════════
# KAMERA THREAD
# ════════════════════════════════════════════════════════════
class KameraThread(threading.Thread):
    def __init__(self, url, cam_id):
        super().__init__(daemon=True)
        self.url     = url
        self.cam_id  = cam_id
        self.frame   = None
        self.running = True
        self.lock    = threading.Lock()

    def run(self):
        cap = cv2.VideoCapture(self.url)
        while self.running:
            ret, frame = cap.read()
            if not ret:
                print(f"[CAM {self.cam_id}] Kamera okunamadi!")
                time.sleep(0.1)
                continue
            with self.lock:
                self.frame = frame
        cap.release()

    def get_frame(self):
        with self.lock:
            return self.frame.copy() if self.frame is not None else None

    def stop(self):
        self.running = False


# ════════════════════════════════════════════════════════════
# VIDEO WRITER OLUŞTUR
# ════════════════════════════════════════════════════════════
def writer_olustur(cam_id, w, h, fps=15):
    zaman     = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    dosya_adi = KAYIT_DIR / f"cam{cam_id}_{zaman}.avi"
    fourcc    = cv2.VideoWriter_fourcc(*"XVID")
    writer    = cv2.VideoWriter(str(dosya_adi), fourcc, fps, (w, h))
    print(f"[CAM {cam_id}] Kayit basliyor: {dosya_adi}")
    return writer


# ════════════════════════════════════════════════════════════
# KAMERA İŞLEME
# ════════════════════════════════════════════════════════════
def isle_kamera(cam_thread, cam_id, roi_orig, fps_state, log_list, son_log, writer):
    frame = cam_thread.get_frame()
    if frame is None:
        return None, fps_state

    orig_h, orig_w = frame.shape[:2]
    frame          = cv2.resize(frame, (orig_w // 2, orig_h // 2))
    yeni_h, yeni_w = frame.shape[:2]

    roi_poly = roi_scaled(roi_orig, orig_w, orig_h, yeni_w, yeni_h)

    mask            = np.zeros((yeni_h, yeni_w), dtype=np.uint8)
    cv2.fillPoly(mask, [roi_poly], 255)
    frame_masked    = frame.copy()
    frame_masked[mask == 0] = 0

    aktif_yuz = 0
    fps       = fps_state["fps"]
    prev_time = fps_state["prev_time"]

    with model_lock:
        tum_yuzler = app.get(frame_masked)

    if len(tum_yuzler) > 0:
        print(f"\n[CAM {cam_id}] {len(tum_yuzler)} yuz bulundu (ROI icinde):")

    for face in tum_yuzler:
        x1, y1, x2, y2 = face.bbox.astype(int)

        crop     = face_align.norm_crop(frame, landmark=face.kps)
        gray     = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        blur_val = cv2.Laplacian(gray, cv2.CV_64F).var()

        if face.det_score < DET_SCORE_ESIK:
            print(f"  ATLANDI | det_score: {face.det_score:.3f} | blur: {blur_val:.1f}")
            continue

        aktif_yuz += 1
        emb = face.normed_embedding
        best_name, best_score = "Unknown", 0.0

        for name, reg_emb in registry.items():
            score = float(np.dot(emb, reg_emb))
            if score > best_score:
                best_score = score
                best_name  = name

        if best_score < SIMILARITY_ESIK:
            best_name = "Unknown"

        print(f"  det_score: {face.det_score:.3f} | blur: {blur_val:.1f} | {best_name} ({best_score:.3f})")

        simdi   = time.time()
        log_key = f"{cam_id}_{best_name}"
        gecen   = (simdi - son_log[log_key]) if log_key in son_log else (LOG_ARALIK + 1)

        if gecen >= LOG_ARALIK:
            simdi_dt       = datetime.datetime.now()
            saat_str       = simdi_dt.strftime("%H:%M:%S")
            tarih_saat_str = simdi_dt.strftime("%Y-%m-%d %H:%M:%S")

            log_list.append((saat_str, f"C{cam_id}", best_name, best_score))
            son_log[log_key] = simdi
            if len(log_list) > 5:
                log_list.pop(0)

            log_file.write(f"[{tarih_saat_str}] CAM {cam_id} | {best_name} | skor: {best_score:.3f}\n")
            log_file.flush()

        color = CW_YESIL if best_name != "Unknown" else CW_KIRMIZI
        koseli_cerceve(frame, x1, y1, x2, y2, color)

        label = f"{best_name}  {best_score:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 1)
        ov = frame.copy()
        cv2.rectangle(ov, (x1, y2+2), (x1+tw+10, y2+th+14), CW_KOYU, -1)
        cv2.addWeighted(ov, 0.65, frame, 0.35, 0, frame)
        cv2.putText(frame, label, (x1+5, y2+th+6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, color, 1)

        for kp in face.kps.astype(int):
            cv2.circle(frame, tuple(kp), 3, CW_ACIK_MAVI, -1)

    roi_ciz(frame, roi_poly, aktif=(aktif_yuz > 0))

    now  = time.time()
    fps  = 0.9 * fps + 0.1 / max(now - prev_time, 1e-6)
    fps_state["fps"]       = fps
    fps_state["prev_time"] = now

    banner(frame, fps, len(registry), aktif_yuz, cam_id, kayit=True)
    log_goster(frame, log_list)

    # AVI'ye yaz
    if writer is not None:
        writer.write(frame)

    return frame, fps_state


# ════════════════════════════════════════════════════════════
# ANA DÖNGÜ
# ════════════════════════════════════════════════════════════
url1 = "rtsp://admin:.com12167983@10.150.65.175:554/Streaming/Channels/101"
url2 = "rtsp://admin:.com12167983@10.150.65.176:554/Streaming/Channels/101"

cam1 = KameraThread(url1, cam_id=1)
cam2 = KameraThread(url2, cam_id=2)
cam1.start()
cam2.start()

# İlk frame'i bekle — boyut için
print("Ilk frame bekleniyor...")
while True:
    f1 = cam1.get_frame()
    f2 = cam2.get_frame()
    if f1 is not None and f2 is not None:
        break
    time.sleep(0.1)

h1, w1 = f1.shape[:2]
h2, w2 = f2.shape[:2]

writer1 = writer_olustur(1, w1 // 2, h1 // 2)
writer2 = writer_olustur(2, w2 // 2, h2 // 2)

fps1 = {"fps": 0.0, "prev_time": time.time()}
fps2 = {"fps": 0.0, "prev_time": time.time()}

shared_log  = []
shared_slog = {}

try:
    while True:
        frame1, fps1 = isle_kamera(cam1, cam_id=1, roi_orig=ROI_CAM1,
                                   fps_state=fps1, log_list=shared_log,
                                   son_log=shared_slog, writer=writer1)
        frame2, fps2 = isle_kamera(cam2, cam_id=2, roi_orig=ROI_CAM2,
                                   fps_state=fps2, log_list=shared_log,
                                   son_log=shared_slog, writer=writer2)

        if frame1 is not None:
            cv2.imshow("CAM 1 - CW ENERJI", frame1)
        if frame2 is not None:
            cv2.imshow("CAM 2 - CW ENERJI", frame2)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
finally:
    cam1.stop()
    cam2.stop()
    writer1.release()
    writer2.release()
    cv2.destroyAllWindows()
    log_file.write(f"\nOturum bitti: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    log_file.close()
    print("Kayitlar kaydedildi:", KAYIT_DIR)