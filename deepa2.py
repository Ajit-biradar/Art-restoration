import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import cv2
import numpy as np
import os
from basicsr.utils import imwrite
from gfpgan import GFPGANer

class CustomMessageBox:
    def __init__(self, parent, title, message):
        self.top = tk.Toplevel(parent)
        self.top.title(title)
        self.top.geometry("400x200")
        self.top.config(bg='#f0f0f0')
        tk.Label(self.top, text=message, font=("Arial", 12), bg='#f0f0f0', fg='#333').pack(pady=20)
        tk.Button(self.top, text="OK", command=self.top.destroy, font=("Arial", 10), bg="#4CAF50", fg="white", relief="flat").pack()

class ImageRestorationApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Image Restoration with GFPGAN")
        self.input_path = None
        self.output_path = "results"
        self.upscale = 2
        self.model_version = '1.3'
        self.setup_restorer()

        tk.Button(root, text="Select Image", command=self.select_image).pack()
        tk.Button(root, text="Restore Image", command=self.restore_image).pack()
        self.input_label = tk.Label(root, text="Input Image")
        self.input_label.pack()
        self.input_image_panel = tk.Label(root)
        self.input_image_panel.pack()
        self.output_label = tk.Label(root, text="Restored Image")
        self.output_label.pack()
        self.output_image_panel = tk.Label(root)
        self.output_image_panel.pack()

    def setup_restorer(self):
        model_name = 'GFPGANv1.3'
        model_path = os.path.join('gfpgan/weights', model_name + '.pth')
        self.restorer = GFPGANer(
            model_path=model_path, upscale=self.upscale, arch='clean', channel_multiplier=2)

    def select_image(self):
        self.input_path = filedialog.askopenfilename(
            title="Select an Image", filetypes=[("Image Files", "*.jpg;*.jpeg;*.png"), ("All Files", "*.*")]
        )
        if not self.input_path:
            CustomMessageBox(self.root, "No Image Selected", "Please select an image to restore.")
            return
        input_img = Image.open(self.input_path)
        input_img.thumbnail((400, 400))
        self.input_image = ImageTk.PhotoImage(input_img)
        self.input_image_panel.configure(image=self.input_image)
        self.input_image_panel.image = self.input_image

    def restore_image(self):
        if not self.input_path:
            CustomMessageBox(self.root, "No Image Selected", "Please select an image first.")
            return

        # Step 1: Load and convert the image to grayscale
        input_img = cv2.imread(self.input_path)
        gray_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)

        # Step 2: Detect damaged areas (using thresholding here as a simple example)
        _, mask = cv2.threshold(gray_img, 240, 255, cv2.THRESH_BINARY)  # Assuming bright spots are damaged areas

        # Step 3: Inpaint the damaged areas
        inpainted_img = cv2.inpaint(input_img, mask, 3, cv2.INPAINT_TELEA)

        # Step 4: Use GFPGAN for facial restoration
        _, _, restored_img = self.restorer.enhance(
            inpainted_img, has_aligned=False, only_center_face=False, paste_back=True
        )

        # Step 5: Save and display the restored image
        os.makedirs(self.output_path, exist_ok=True)
        output_file = os.path.join(self.output_path, "restored_img.png")
        imwrite(restored_img, output_file)
        restored_img_pil = Image.fromarray(cv2.cvtColor(restored_img, cv2.COLOR_BGR2RGB))
        restored_img_pil.thumbnail((400, 400))
        self.output_image = ImageTk.PhotoImage(restored_img_pil)
        self.output_image_panel.configure(image=self.output_image)
        self.output_image_panel.image = self.output_image

        CustomMessageBox(self.root, "Restoration Complete", f"Restored image saved at {output_file}")

root = tk.Tk()
app = ImageRestorationApp(root)
root.mainloop()
