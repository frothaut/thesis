import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageDraw, ImageTk
import os



class ImageDrawer:
    def __init__(self, root, image_path):
        self.root = root
        self.image_path = image_path
        self.image = Image.open(image_path)
        self.drawing_image = Image.new("RGBA", self.image.size, (255, 255, 255, 0))  # Transparent background for drawing
        self.draw = ImageDraw.Draw(self.drawing_image)
        self.last_x, self.last_y = None, None

        # Set brush size here (can be adjusted as needed)
        self.brush_size = 10  # Define the brush size (in pixels)

        self.canvas = tk.Canvas(self.root, width=self.image.width, height=self.image.height)
        self.canvas.pack()

        # Convert the image to a format that Tkinter can display
        self.tk_image = ImageTk.PhotoImage(self.image)
        self.canvas.create_image(0, 0, image=self.tk_image, anchor="nw")

        # Bind the drawing event to the canvas (drawing while dragging the mouse)
        self.canvas.bind("<B1-Motion>", self.paint)
        self.canvas.bind("<ButtonRelease-1>", self.stop_drawing)  # Stop drawing after mouse release

        # Add the Save button to save the drawing
        self.save_button = tk.Button(self.root, text="Save Drawing", command=self.save_drawing)
        self.save_button.pack()

        # Initialize drawing color and width
        self.color = "red"  # Default color is red

        # Add color selection buttons
        self.red_button = tk.Button(self.root, text="Red", command=self.set_red)
        self.red_button.pack(side=tk.LEFT)
        
        self.green_button = tk.Button(self.root, text="Green", command=self.set_green)
        self.green_button.pack(side=tk.LEFT)

        self.blue_button = tk.Button(self.root, text="Blue", command=self.set_blue)
        self.blue_button.pack(side=tk.LEFT)

    def set_red(self):
        """Set drawing color to red."""
        self.color = "red"

    def set_green(self):
        """Set drawing color to green."""
        self.color = "green"

    def set_blue(self):
        """Set drawing color to blue."""
        self.color = "blue"

    def paint(self, event):
        """Handle the drawing event while the mouse is dragged."""
        if self.last_x and self.last_y:
            # Draw a round brush (ellipse) at the current mouse position
            self.draw.ellipse([event.x - self.brush_size / 2, event.y - self.brush_size / 2, 
                               event.x + self.brush_size / 2, event.y + self.brush_size / 2],
                              fill=self.color, outline=self.color)
            # Update the canvas with the new drawn part (ellipse)
            self.canvas.create_oval(event.x - self.brush_size / 2, event.y - self.brush_size / 2,
                                    event.x + self.brush_size / 2, event.y + self.brush_size / 2, fill=self.color, outline=self.color)
        self.last_x, self.last_y = event.x, event.y

    def stop_drawing(self, event):
        """Reset the last_x and last_y when the mouse is released."""
        self.last_x, self.last_y = None, None

    def save_drawing(self):
        """Save the drawing to a file in the same folder as the original image."""
        # Get the directory of the original image
        dir_path = os.path.dirname(self.image_path)
        # Save the drawing image with the same name as the original image but with "_drawing" appended
        save_path = os.path.join(dir_path, os.path.splitext(os.path.basename(self.image_path))[0] + "_drawing.png")

        # Replace transparent pixels with black before saving
        drawing_with_black_bg = self.drawing_image.convert("RGBA")
        pixels = drawing_with_black_bg.load()

        for i in range(drawing_with_black_bg.width):
            for j in range(drawing_with_black_bg.height):
                if pixels[i, j][3] == 0:  # Check if pixel is fully transparent (alpha channel == 0)
                    pixels[i, j] = (0, 0, 0, 255)  # Change transparent pixels to black

        # Save the modified drawing image
        drawing_with_black_bg.save(save_path)
        print(f"Drawing saved to {save_path}")

        # Close the window after saving
        self.root.quit()

if __name__ == "__main__":
    root = tk.Tk()
    image_path = "dji_fly_20250414_131756_25_1744629601154_photo 2.png"
    #image_path = filedialog.askopenfilename(title="Open Image", filetypes=[("Image files", "*.png;*.jpg;*.jpeg;*.bmp")])
    
    if image_path:
        app = ImageDrawer(root, image_path)
        root.mainloop()