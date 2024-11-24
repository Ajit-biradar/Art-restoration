
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import os
import cv2
from basicsr.utils import imwrite
from gfpgan import GFPGANer

class CustomMessageBox:
    def __init__(self, parent, title, message):
        self.top = tk.Toplevel(parent)
        self.top.title(title)
        self.top.geometry("700x4000")  # Customize the size

        # Set background color and other styling
        self.top.config(bg='#f0f0f0')

        # Label with custom font and colors
        self.label = tk.Label(self.top, text=message, font=("Arial", 12), bg='#f0f0f0', fg='#333')
        self.label.pack(pady=20)

        # Button with custom styling
        self.ok_button = tk.Button(self.top, text="OK", command=self.top.destroy, 
                                   font=("Arial", 10), bg="#4CAF50", fg="white", relief="flat")
        self.ok_button.pack()

        # Center the window on the screen
        self.center_window()

    def center_window(self):
        window_width = 700
        window_height = 4000
        screen_width = self.top.winfo_screenwidth()
        screen_height = self.top.winfo_screenheight()

        # Calculate the position of the window
        position_top = int(screen_height / 2 - window_height / 2)
        position_left = int(screen_width / 2 - window_width / 2)

        # Set the position of the window
        self.top.geometry(f'{window_width}x{window_height}+{position_left}+{position_top}')

class ImageRestorationApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Image Restoration with GFPGAN")

        # Initialize variables
        self.input_path = None
        self.output_path = "results"
        self.upscale = 2
        self.model_version = '1.3'

        # Set up GFPGAN restorer
        self.setup_restorer()

        # Create GUI elements
        self.select_button = tk.Button(root, text="Select Image", command=self.select_image)
        self.select_button.pack()

        self.restore_button = tk.Button(root, text="Restore Image", command=self.restore_image)
        self.restore_button.pack()

        self.input_label = tk.Label(root, text="Input Image")
        self.input_label.pack()

        self.input_image_panel = tk.Label(root)
        self.input_image_panel.pack()

        self.output_label = tk.Label(root, text="Restored Image")
        self.output_label.pack()

        self.output_image_panel = tk.Label(root)
        self.output_image_panel.pack()

    def setup_restorer(self):
        # Model URL and setup
        model_name = 'GFPGANv1.3'
        model_path = os.path.join('experiments/pretrained_models', model_name + '.pth')
        if not os.path.isfile(model_path):
            model_path = os.path.join('gfpgan/weights', model_name + '.pth')
        
        self.restorer = GFPGANer(
            model_path=model_path,
            upscale=self.upscale,
            arch='clean',
            channel_multiplier=2
        )

    def select_image(self):
        print("Select Image button pressed.")

        # Use file dialog to select an image file
        self.input_path = filedialog.askopenfilename(
            title="Select an Image",
            filetypes=[("Image Files", "*.jpg;*.jpeg;*.png"), ("All Files", "*.*")]
        )

        # If no image is selected, show a message box
        if not self.input_path:
            CustomMessageBox(self.root, "No Image Selected", "Please select an image to restore.")
            return

        print(f"Selected file path: {self.input_path}")

        # Display input image if a valid file path was selected
        try:
            input_img = Image.open(self.input_path)
            input_img.thumbnail((400, 400))  # Resize for display purposes
            self.input_image = ImageTk.PhotoImage(input_img)
            self.input_image_panel.configure(image=self.input_image)
            self.input_image_panel.image = self.input_image  # Keep a reference
            CustomMessageBox(self.root, "Image Selected", "The selected image will be restored.")
            print("Image selected and displayed successfully.")
        except Exception as e:
            CustomMessageBox(self.root, "Error", f"Error opening image: {e}")
            print(f"Error opening image: {e}")

    def restore_image(self):
        if not self.input_path:
            CustomMessageBox(self.root, "No Image Selected", "Please select an image first.")
            return

        # Read and restore image
        input_img = cv2.imread(self.input_path, cv2.IMREAD_COLOR)
        _, _, restored_img = self.restorer.enhance(
            input_img, has_aligned=False, only_center_face=False, paste_back=True, weight=0.5
        )

        # Save and display the restored image
        os.makedirs(self.output_path, exist_ok=True)
        output_file = os.path.join(self.output_path, "restored_img.png")
        imwrite(restored_img, output_file)

        # Display the restored image
        restored_img_pil = Image.fromarray(cv2.cvtColor(restored_img, cv2.COLOR_BGR2RGB))
        restored_img_pil.thumbnail((400, 400))
        self.output_image = ImageTk.PhotoImage(restored_img_pil)
        self.output_image_panel.configure(image=self.output_image)
        self.output_image_panel.image = self.output_image  # Keep a reference

        CustomMessageBox(self.root, "Restoration Complete", f"Restored image saved at {output_file}")
        print(f"Restored image saved at {output_file}")

# Run the Tkinter GUI
root = tk.Tk()
app = ImageRestorationApp(root)
root.mainloop()
