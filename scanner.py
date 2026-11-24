import cv2
import os
import numpy as np
from skimage.metrics import structural_similarity as ssim
import customtkinter as ctk
from tkinter import filedialog, Toplevel, Canvas, Label, Button
from PIL import Image, ImageTk
from matplotlib import pyplot as plt
from PIL import ImageDraw
from tkinter import BOTH, X, Y, TOP, BOTTOM,RIGHT, LEFT, HORIZONTAL, VERTICAL, Frame, Scrollbar

ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")

ref_folder_paths = [
    r"C:\Users\bunny\OneDrive\Desktop\project\scanner\dataset_scan\train\1000",
    r"C:\Users\bunny\OneDrive\Desktop\project\scanner\dataset_scan\train\5000",
    r"C:\Users\bunny\OneDrive\Desktop\project\scanner\dataset_scan\train\10000"
]
currency_types = ["1000 Kyats", "5000 Kyats", "10000 Kyats"]

def preprocess_image(image, to_gray=True, apply_threshold=False):
    image = cv2.resize(image, (400, 200)) 
    if to_gray:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) 
    if apply_threshold:
        image = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    return image

def orb_feature_matching(img1, img2):
    orb = cv2.ORB_create(1500) 
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)
    if des1 is None or des2 is None:
        return 0.0, None
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    if len(matches) == 0:
        return 0.0, None
    matches = sorted(matches, key=lambda x: x.distance)
    score = len(matches) / max(len(kp1), len(kp2)) * 100
    img3 = cv2.drawMatches(img1, kp1, img2, kp2, matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    return float(score), img3

def sift_feature_matching(img1, img2):
    sift = cv2.SIFT_create(5000) 
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)
    if des1 is None or des2 is None:
        return 0.0, None
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)
    good_matches = [m for m, n in matches if m.distance < 0.75 * n.distance]
    if len(good_matches) == 0:
        return 0.0, None
    score = len(good_matches) / max(len(kp1), len(kp2)) * 100
    img3 = cv2.drawMatchesKnn(img1, kp1, img2, kp2, [good_matches], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    return float(score), img3

def compute_ssim(img1, img2):
    ssim_score, ssim_map = ssim(img1, img2, full=True)
    ssim_score = float(ssim_score * 100)
    ssim_map = (ssim_map * 255).astype("uint8")
    return ssim_score, ssim_map

def compute_histogram_similarity(img1, img2):
    hist1 = cv2.calcHist([img1], [0], None, [256], [0, 256])
    hist2 = cv2.calcHist([img2], [0], None, [256], [0, 256])
    cv2.normalize(hist1, hist1)
    cv2.normalize(hist2, hist2)
    similarity = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
    return float(similarity * 100)

def show_comparison_window(test_img, ref_img, orb_img, sift_img, ssim_img):
    if test_img is not None and ref_img is not None:
        test_img = cv2.cvtColor(test_img, cv2.COLOR_BGR2RGB)
        ref_img = cv2.cvtColor(ref_img, cv2.COLOR_BGR2RGB)
        fig, axs = plt.subplots(2, 3, figsize=(15, 10))
        axs[0, 0].imshow(test_img)
        axs[0, 0].set_title("Test Image")
        axs[0, 0].axis("off")
        axs[0, 1].imshow(ref_img)
        axs[0, 1].set_title("Reference Image")
        axs[0, 1].axis("off")
        axs[1, 0].imshow(orb_img)
        axs[1, 0].set_title("ORB Matching")
        axs[1, 0].axis("off")
        axs[1, 1].imshow(sift_img)
        axs[1, 1].set_title("SIFT Matching")
        axs[1, 1].axis("off")
        axs[1, 2].imshow(ssim_img, cmap='gray')
        axs[1, 2].set_title("SSIM")
        axs[1, 2].axis("off")
        plt.show()

def process_image(image_path):
    test_image = cv2.imread(image_path)
    if test_image is None:
        result_label.configure(text="Error: Could not load image.")
        return

    test_image = preprocess_image(test_image)

    best_similarity = 0
    best_currency_type = "Unknown"
    best_ref_image = None
    best_orb_img, best_sift_img, best_ssim_img = None, None, None

    for folder_path, currency_type in zip(ref_folder_paths, currency_types):
        for file_name in os.listdir(folder_path):
            if not file_name.endswith(".jpg"):
                continue
            ref_image_path = os.path.join(folder_path, file_name)
            ref_image = cv2.imread(ref_image_path)
            if ref_image is None:
                continue

            ref_image_preprocessed = preprocess_image(ref_image, to_gray=False)
            orb_score, orb_img = orb_feature_matching(test_image, ref_image_preprocessed)
            sift_score, sift_img = sift_feature_matching(test_image, ref_image_preprocessed)

            ref_image_gray = preprocess_image(ref_image, to_gray=True, apply_threshold=True)
            ssim_score, ssim_img = compute_ssim(test_image, ref_image_gray)
            hist_score = compute_histogram_similarity(test_image, ref_image_preprocessed)

            total_similarity = (0.3 * orb_score) + (0.3 * sift_score) + (0.2 * ssim_score) + (0.2 * hist_score)
            print(f'Reference Image: {file_name}\nCurrency Type: {currency_type}\nSimilarity: {total_similarity:.2f}%\n orb_score: {orb_score:.2f}\n ssim_score: {ssim_score:.2f}\n sift_score: {sift_score:.2f}\n hist_score:{hist_score:.2f}\n')
        
            if total_similarity > best_similarity:
                best_similarity = total_similarity
                best_currency_type = currency_type
                best_ref_image = ref_image
                best_orb_img, best_sift_img, best_ssim_img = orb_img, sift_img, ssim_img

    threshold = 82 
    classification_result = 'REAL' if best_similarity > threshold else 'FAKE'
    if classification_result == 'FAKE' and best_similarity < 50:
        best_currency_type = 'Unknown'
    elif classification_result == 'FAKE' and best_similarity >= 50:
        best_currency_type = best_currency_type

    result_label.configure(text=f'Currency: {best_currency_type}\nClassification: {classification_result}\nSimilarity: {best_similarity:.2f}%')

    if best_ref_image is not None:
        show_comparison_window(cv2.imread(image_path), best_ref_image, best_orb_img, best_sift_img, best_ssim_img)

    segmentation_button = ctk.CTkButton(root, text="Proceed to Segmentation", command=lambda: segment_and_compare(image_path, best_ref_image))
    segmentation_button.pack(pady=10)

def segment_and_compare(test_image_path, ref_image):
    if ref_image is None:
        result_label.configure(text="Segmentation Error: No valid reference image.")
        return

    test_image = cv2.imread(test_image_path)
    test_image = cv2.resize(test_image, (400, 200))
    ref_image = cv2.resize(ref_image, (400, 200))

    test_gray = cv2.cvtColor(test_image, cv2.COLOR_BGR2GRAY)
    ref_gray = cv2.cvtColor(ref_image, cv2.COLOR_BGR2GRAY)

    h, w = test_gray.shape
    grid_size_x, grid_size_y = w // 3, h // 3

    fake_detected = False
    grid_results = []

    for i in range(3):
        for j in range(3):
            x_start, y_start = i * grid_size_x, j * grid_size_y
            x_end, y_end = x_start + grid_size_x, y_start + grid_size_y

            test_grid = test_gray[y_start:y_end, x_start:x_end]
            ref_grid = ref_gray[y_start:y_end, x_start:x_end]

            similarity, _ = compute_ssim(test_grid, ref_grid)

            print(f"Grid ({i+1}, {j+1}) - Test Image:\n{test_grid}")
            print(f"Grid ({i+1}, {j+1}) - Reference Image:\n{ref_grid}")
            print(f"Grid ({i+1}, {j+1}) - Similarity: {similarity:.2f}%\n")

            if similarity < 90:
                fake_detected = True

            grid_results.append((test_grid, ref_grid, similarity))

    update_main_ui_with_grid_results(grid_results, fake_detected)
    display_segment_comparison(grid_results, fake_detected)

def display_segment_comparison(grid_results, fake_detected):
    segment_window = Toplevel(root)
    segment_window.title("Segmentation-Based Analysis")

    fig, axs = plt.subplots(3, 6, figsize=(12, 6))
    fig.suptitle("Test Image vs Reference Image (3x3 Grid)")

    for idx, (test_grid, ref_grid, similarity) in enumerate(grid_results):
        row, col = divmod(idx, 3)
        axs[row, col * 2].imshow(test_grid, cmap='gray')
        axs[row, col * 2].set_title(f"Test Grid {idx + 1}")
        axs[row, col * 2].axis("off")
        axs[row, col * 2 + 1].imshow(ref_grid, cmap='gray')
        axs[row, col * 2 + 1].set_title(f"Ref Grid {idx + 1}\nSim: {similarity:.2f}%")
        axs[row, col * 2 + 1].axis("off")

    plt.tight_layout()
    plt.show()

def update_main_ui_with_grid_results(grid_results, classification_result):
    results_frame = ctk.CTkFrame(root, bg_color="gray")
    results_frame.pack(pady=10)

    for idx, (test_grid, ref_grid, similarity) in enumerate(grid_results):
        status = "REAL" if similarity >= 90 else "FAKE"
        color = "green" if similarity >= 90 else "red"
        label_text = f"Grid {idx + 1}: Sim: {similarity:.2f}% - {status}"
        grid_label = ctk.CTkLabel(results_frame, text=label_text, font=("Arial", 20), text_color=color, bg_color="black")
        grid_label.pack()

def open_image():
    file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg;*.png;*.jpeg")])
    if not file_path:
        return
    process_image(file_path)

def capture_image():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        result_label.configure(text="Error: Cannot access webcam.")
        return

    def take_photo():
        ret, frame = cap.read()
        if ret:
            cap.release()
            cv2.destroyAllWindows()
            captured_image_path = "captured_image.png" 
            cv2.imwrite(captured_image_path, frame)
            open_edit_window(captured_image_path) 

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imshow("Live Camera - Press 'C' to Capture", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('c'):
            take_photo()
            break
        elif key == 27: 
            cap.release()
            cv2.destroyAllWindows()
            return

def open_edit_window(image_path):
    edit_window = Toplevel(root)
    edit_window.title("Edit Image")

    img = Image.open(image_path)
    img_tk = ImageTk.PhotoImage(img)

    button_frame = Frame(edit_window)
    button_frame.pack(side=TOP, fill=X)

    def rotate_image():
        nonlocal img
        img = img.rotate(90, expand=True)
        img_tk_new = ImageTk.PhotoImage(img)
        canvas.create_image(0, 0, anchor="nw", image=img_tk_new)
        canvas.image = img_tk_new
        canvas.config(scrollregion=(0, 0, img.width, img.height))

    def enable_cropping():
        canvas.bind("<ButtonPress-1>", start_crop)
        canvas.bind("<B1-Motion>", drag)
        canvas.bind("<ButtonRelease-1>", end_crop)

    def start_crop(event):
        nonlocal cropping, start_x, start_y, rect_id
        cropping = True
        start_x = event.x
        start_y = event.y
        if rect_id is not None:
            canvas.delete(rect_id)
        rect_id = canvas.create_rectangle(start_x, start_y, start_x, start_y, outline='red')

    def end_crop(event):
        nonlocal cropping, end_x, end_y
        cropping = False
        end_x = event.x
        end_y = event.y

    def crop_image():
        nonlocal img
        crop_box = (start_x, start_y, end_x, end_y)
        img = img.crop(crop_box)
        img_tk_new = ImageTk.PhotoImage(img)
        canvas.create_image(0, 0, anchor="nw", image=img_tk_new)
        canvas.image = img_tk_new
        canvas.config(scrollregion=(0, 0, img.width, img.height))
        canvas.unbind("<ButtonPress-1>")
        canvas.unbind("<B1-Motion>")
        canvas.unbind("<ButtonRelease-1>")

    def drag(event):
        nonlocal rect_id
        if cropping:
            end_x, end_y = event.x, event.y
            canvas.coords(rect_id, start_x, start_y, end_x, end_y)

    def save_and_process():
        img.save(image_path)
        edit_window.destroy()
        process_image(image_path)

    crop_button = ctk.CTkButton(button_frame, text="Enable Cropping", command=enable_cropping)
    crop_button.pack(side=LEFT, padx=5, pady=5)

    apply_crop_button = ctk.CTkButton(button_frame, text="Apply Crop", command=crop_image)
    apply_crop_button.pack(side=LEFT, padx=5, pady=5)

    rotate_button = ctk.CTkButton(button_frame, text="Rotate 90°", command=rotate_image)
    rotate_button.pack(side=LEFT, padx=5, pady=5)

    save_button = ctk.CTkButton(button_frame, text="Save & Process", command=save_and_process)
    save_button.pack(side=LEFT, padx=5, pady=5)

    canvas = Canvas(edit_window, width=img.width, height=img.height, scrollregion=(0, 0, img.width, img.height))
    canvas.create_image(0, 0, anchor="nw", image=img_tk)
    canvas.image = img_tk
    canvas.pack(fill=BOTH, expand=True)

    hbar = Scrollbar(edit_window, orient=HORIZONTAL, command=canvas.xview)
    hbar.pack(side=BOTTOM, fill=X)
    vbar = Scrollbar(edit_window, orient=VERTICAL, command=canvas.yview)
    vbar.pack(side=RIGHT, fill=Y)
    canvas.config(xscrollcommand=hbar.set, yscrollcommand=vbar.set)

    cropping = False
    start_x = start_y = end_x = end_y = 0
    rect_id = None

root = ctk.CTk()
root.title("Fake Currency Detection")
root.geometry("500x400")

upload_button = ctk.CTkButton(root, text="Upload Image", command=open_image)
upload_button.pack(pady=10)

capture_button = ctk.CTkButton(root, text="Capture Image", command=capture_image)
capture_button.pack(pady=10)

result_label = ctk.CTkLabel(root, text="Select or Capture an Image", font=("Arial", 16))
result_label.pack(pady=20)

root.mainloop()