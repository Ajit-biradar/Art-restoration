from tkinter import Tk, filedialog

def select_image():
    # Open the file dialog and print the selected file path
    input_path = filedialog.askopenfilename(
        title="Select an Image",
        filetypes=[("Image Files", "*.jpg;*.jpeg;*.png"), ("All Files", "*.*")]
    )
    print(f"Selected file path: {input_path}")

# Set up a simple Tkinter window
root = Tk()
root.title("Test Image Selection")

# Run the select_image function directly to see if the dialog opens
select_image()

root.mainloop()
