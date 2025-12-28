import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
from tkinter import ttk

class DigitDrawingApp:
    def __init__(self, root):
        self.root = root
        self.root.title("MNIST Digit Recognizer")

        # Load the trained model
        try:
            self.model = keras.models.load_model('mnist_model.h5')
            print("Model loaded successfully!")
        except Exception as e:
            print(f"Error loading model: {e}")
            print("Please train the model first by running model_training.py")
            return

        # Create 28x28 drawing canvas (scaled up for visibility)
        self.canvas_size = 280  # 28 * 10 for better visibility
        self.pixel_size = 10
        self.drawing = np.zeros((28, 28))

        # Setup UI
        self.setup_ui()

    def setup_ui(self):
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # Left frame for drawing
        left_frame = ttk.Frame(main_frame)
        left_frame.grid(row=0, column=0, padx=10, pady=10, sticky=(tk.N, tk.S))

        # Canvas for drawing (larger)
        self.canvas = tk.Canvas(left_frame, width=self.canvas_size, height=self.canvas_size,
                               bg='black', cursor='cross')
        self.canvas.grid(row=0, column=0, padx=5, pady=5)

        # Bind mouse events
        self.canvas.bind('<B1-Motion>', self.paint)
        self.canvas.bind('<Button-1>', self.paint)

        # Buttons below canvas
        button_frame = ttk.Frame(left_frame)
        button_frame.grid(row=1, column=0, pady=10)

        ttk.Button(button_frame, text="Predict", command=self.predict).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Clear", command=self.clear_canvas).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Exit", command=self.root.quit).pack(side=tk.LEFT, padx=5)

        # Right frame for results
        right_frame = ttk.Frame(main_frame)
        right_frame.grid(row=0, column=1, padx=10, pady=10, sticky=(tk.N, tk.S))

        # Instruction label
        instruction_label = ttk.Label(right_frame, text="Draw a digit (0-9)",
                                     font=('Arial', 14))
        instruction_label.grid(row=0, column=0, pady=10)

        # Prediction label
        self.result_label = ttk.Label(right_frame, text="Predicted Digit: -",
                                     font=('Arial', 20, 'bold'))
        self.result_label.grid(row=1, column=0, pady=10)

        # Confidence label
        self.confidence_label = ttk.Label(right_frame, text="Confidence: -", font=('Arial', 14))
        self.confidence_label.grid(row=2, column=0, pady=5)

        # Probability visualization frame
        self.prob_frame = ttk.Frame(right_frame)
        self.prob_frame.grid(row=3, column=0, pady=10)

    def paint(self, event):
        # Convert canvas coordinates to pixel coordinates
        x = event.x // self.pixel_size
        y = event.y // self.pixel_size

        # Ensure coordinates are within bounds
        if 0 <= x < 28 and 0 <= y < 28:
            # Draw on the pixel array with some smoothing
            brush_size = 1
            for dx in range(-brush_size, brush_size + 1):
                for dy in range(-brush_size, brush_size + 1):
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < 28 and 0 <= ny < 28:
                        self.drawing[ny, nx] = min(1.0, self.drawing[ny, nx] + 0.5)

            # Update canvas display
            self.update_canvas()

    def update_canvas(self):
        # Clear and redraw
        self.canvas.delete('all')
        for i in range(28):
            for j in range(28):
                if self.drawing[i, j] > 0:
                    intensity = int(self.drawing[i, j] * 255)
                    color = f'#{intensity:02x}{intensity:02x}{intensity:02x}'
                    self.canvas.create_rectangle(
                        j * self.pixel_size, i * self.pixel_size,
                        (j + 1) * self.pixel_size, (i + 1) * self.pixel_size,
                        fill=color, outline=''
                    )

    def predict(self):
        # Prepare the image for prediction
        img = self.drawing.reshape(1, 28, 28, 1).astype('float32')

        # Make prediction
        predictions = self.model.predict(img, verbose=0)
        predicted_digit = np.argmax(predictions[0])
        confidence = predictions[0][predicted_digit] * 100

        # Update result labels
        self.result_label.config(text=f"Predicted Digit: {predicted_digit}")
        self.confidence_label.config(text=f"Confidence: {confidence:.2f}%")

        # Show probability distribution
        self.show_probabilities(predictions[0])

    def show_probabilities(self, probabilities):
        # Clear previous probability display
        for widget in self.prob_frame.winfo_children():
            widget.destroy()

        # Create bar chart for probabilities
        fig, ax = plt.subplots(figsize=(8, 3))
        digits = range(10)
        bars = ax.bar(digits, probabilities * 100, color='steelblue')

        # Highlight the predicted digit
        max_idx = np.argmax(probabilities)
        bars[max_idx].set_color('orange')

        ax.set_xlabel('Digit')
        ax.set_ylabel('Probability (%)')
        ax.set_title('Prediction Probabilities')
        ax.set_xticks(digits)
        ax.set_ylim(0, 105)
        ax.grid(axis='y', alpha=0.3)

        # Embed in tkinter
        canvas = FigureCanvasTkAgg(fig, master=self.prob_frame)
        canvas.draw()
        canvas.get_tk_widget().pack()

        plt.close(fig)

    def clear_canvas(self):
        # Reset drawing array
        self.drawing = np.zeros((28, 28))
        self.canvas.delete('all')
        self.result_label.config(text="Predicted Digit: -")
        self.confidence_label.config(text="Confidence: -")

        # Clear probability display
        for widget in self.prob_frame.winfo_children():
            widget.destroy()

def main():
    root = tk.Tk()
    app = DigitDrawingApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()
