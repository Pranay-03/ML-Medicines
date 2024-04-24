import tkinter as tk
from tkinter import filedialog, Label, PhotoImage
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from skimage.io import imread
from skimage.transform import resize
import numpy as np
import joblib

# Load the trained model from file
clf = joblib.load('logo_detection_model_2.pkl')

def process_uploaded_image():
    file_path = filedialog.askopenfilename()
    if file_path:
        # Convert uploaded image to numpy array and preprocess
        img = imread(file_path)
        img_resized = resize(img, (100, 100))
        img_flattened = img_resized.flatten().reshape(1, -1)

        # Make predictions using the model
        prediction = clf.predict(img_flattened)

        # Display the uploaded image and prediction result
        ax.clear()
        ax.imshow(img)
        ax.set_title("Uploaded Image", fontsize=14)
        canvas.draw()

        # Update the prediction result
        if prediction == 0:
            result_label.config(text="Prediction: Fake", fg="red", font=("Arial", 16, "bold"))
        else:
            result_label.config(text="Prediction: Real", fg="green", font=("Arial", 16, "bold"))

# Create a Tkinter window
window = tk.Tk()
window.title("Logo Detection App")
window.geometry("800x800")
window.configure(bg="#f0f0f0")

# Create a frame for image display
frame_image = tk.Frame(window, bg="white")
frame_image.pack(pady=20)

# Create Matplotlib figure and canvas for image display
fig, ax = plt.subplots(figsize=(6, 6))
canvas = FigureCanvasTkAgg(fig, master=frame_image)
canvas.get_tk_widget().pack()

# Create a button to upload image
upload_button = tk.Button(window, text="Upload Image", command=process_uploaded_image, font=("Arial", 14), bg="#4CAF50", fg="white", padx=20, pady=10, bd=0)
upload_button.pack(pady=20)

# Create a label to display the prediction result
result_label = Label(window, text="", font=("Arial", 16), bg="#f0f0f0")
result_label.pack()

# Add some decorative elements
title_label = Label(window, text="Logo Detection App", font=("Arial", 24, "bold"), bg="#f0f0f0")
title_label.pack(pady=20)

separator = tk.Frame(window, height=2, bd=1, relief=tk.SUNKEN, bg="#4CAF50")
separator.pack(fill=tk.X, padx=20, pady=10)

# Run the Tkinter event loop
window.mainloop()