# Import necessary modules
import tkinter as tk
from tkinter import messagebox

# Define the action for each button
def release_gpu(gpu_id):
    messagebox.showinfo("GPU Release", f"GPU {gpu_id} has been released.")
    print(f"GPU {gpu_id} has been released.")

# Create the main application window
root = tk.Tk()
root.title("GPU Release Application")

# Set the window size
window_width = 300
window_height = 200

# Get the screen dimension
screen_width = root.winfo_screenwidth()
screen_height = root.winfo_screenheight()

# Find the center point
center_x = int(screen_width / 2 - window_width / 2)
center_y = int(screen_height / 2 - window_height / 2)

# Set the position of the window to the center of the screen
root.geometry(f'{window_width}x{window_height}+{center_x}+{center_y}')

# Create and place the buttons in the center of the window
button_texts = ["Release GPU 1", "Release GPU 2", "Release GPU 3", "Release GPU 4"]
buttons = []

for i, text in enumerate(button_texts):
    button = tk.Button(root, text=text, command=lambda i=i: release_gpu(i+1))
    buttons.append(button)

# Arrange buttons in a grid
for i, button in enumerate(buttons):
    button.grid(row=i, column=0, padx=10, pady=10)

# Run the application
root.mainloop()
