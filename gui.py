import os
import sys
import json
import numpy as np
import tkinter as tk
from tkinter import messagebox, Button, Label, Canvas, Frame, Entry, StringVar, ttk
from PIL import Image, ImageDraw
import onnxruntime as rt
from sklearn.linear_model import LogisticRegression
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
from skimage.transform import resize

onnx_file_path = "hwemoji2.onnx"
dataset_file = "emoji_dataset.json"
highscore_file = "emoji_highscore.json"  # New file to store high score


class EmojiDrawingApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Emoji Drawing Recognizer")
        self.root.resizable(True, True)

        # Initialize variables
        self.canvas_width = 400
        self.canvas_height = 400
        self.points = []
        self.model = None
        self.drawing = False
        self.last_x = 0
        self.last_y = 0
        self.dataset = self.load_dataset()
        self.is_dataset_mode = False
        self.score = 0  # Track user's score
        self.high_score = self.load_high_score()  # Load high score
        self.target_emoji = None  # Current target emoji
        self.target_name = None  # Current target name

        # Create UI elements
        self.setup_ui()

        # Load or train the model
        self.load_model()

        # Initialize target and score display if model is loaded
        if hasattr(self, "class_names") and self.class_names:
            # self.target_frame.pack(pady=10, fill=tk.X)
            self.score_label.pack(pady=5)
            self.high_score_label.pack(pady=5)  # Display high score
            self.select_new_target()
        else:
            messagebox.showwarning(
                "Warning", "No model or class names found. Game mode disabled."
            )

    def load_dataset(self):
        if os.path.exists(dataset_file):
            try:
                with open(dataset_file, "r") as f:
                    return json.load(f)
            except:
                return []
        return []

    def save_dataset(self):
        with open(dataset_file, "w") as f:
            json.dump(self.dataset, f, indent=2)

    def load_high_score(self):
        """Load high score from file if it exists"""
        if os.path.exists(highscore_file):
            try:
                with open(highscore_file, "r") as f:
                    data = json.load(f)
                    return data.get("high_score", 0)
            except:
                return 0
        return 0

    def save_high_score(self):
        """Save high score to file"""
        with open(highscore_file, "w") as f:
            json.dump({"high_score": self.high_score}, f)

    def update_score(self, points):
        """Update score and check if it's a new high score"""
        self.score += points
        self.score_label.config(text=f"Score: {self.score}")

        if self.score > self.high_score:
            self.high_score = self.score
            self.high_score_label.config(text=f"High Score: {self.high_score}")
            self.save_high_score()
            # Flash the high score label to indicate a new high score
            self.flash_high_score_label()

    def flash_high_score_label(
        self, times=6, current=0, original_bg="SystemButtonFace"
    ):
        """Flash the high score label to celebrate a new high score"""
        if current >= times:
            self.high_score_label.config(bg=original_bg)
            return

        new_bg = (
            "#FFD700" if current % 2 == 0 else original_bg
        )  # Gold color for flashing
        self.high_score_label.config(bg=new_bg)
        self.root.after(
            200, lambda: self.flash_high_score_label(times, current + 1, original_bg)
        )

    def save_sample(self):
        if not self.points:
            messagebox.showinfo("Info", "Please draw something first!")
            return

        emoji = self.emoji_var.get().strip()
        emoji_name = self.emoji_name_var.get().strip()

        if not emoji or not emoji_name:
            messagebox.showinfo("Info", "Please enter both emoji and emoji name!")
            return

        # Create a new sample
        sample = {"emoji": emoji, "emojiName": emoji_name, "points": self.points}

        # Add to dataset
        self.dataset.append(sample)

        # Save dataset to file
        self.save_dataset()

        # Update stats
        self.dataset_stats_label.config(
            text=f"Dataset size: {len(self.dataset)} samples"
        )

        # Clear canvas for next sample
        self.clear_canvas()
        # messagebox.showinfo(
        # "Success", f"Sample added to dataset: {emoji} ({emoji_name})"
        # )

    def start_drawing(self, event):
        self.drawing = True
        self.last_x = event.x
        self.last_y = event.y
        self.points.append({"x": event.x, "y": event.y})
        self.canvas.create_oval(
            event.x - 2, event.y - 2, event.x + 2, event.y + 2, fill="black"
        )

    def draw(self, event):
        if self.drawing:
            self.canvas.create_line(
                self.last_x,
                self.last_y,
                event.x,
                event.y,
                width=4,
                fill="black",
                capstyle=tk.ROUND,
                smooth=tk.TRUE,
            )
            self.last_x = event.x
            self.last_y = event.y
            self.points.append({"x": event.x, "y": event.y})

    def stop_drawing(self, event):
        self.drawing = False

    def clear_canvas(self):
        self.canvas.delete("all")
        self.points = []
        # Clear the predictions tree
        # for item in self.predictions_tree.get_children():
        # self.predictions_tree.delete(item)
        # Clear the emoji display
        # self.emoji_display.config(text="")
        # self.emoji_info.config(text="Draw and click 'Submit'")

    def reset_game(self):
        """Reset the current game score but keep high score"""
        self.score = 0
        self.score_label.config(text=f"Score: {self.score}")
        self.clear_canvas()
        self.select_new_target()
        messagebox.showinfo("Game Reset", "Game has been reset. Good luck!")

    def toggle_dataset_mode(self):
        self.is_dataset_mode = not self.is_dataset_mode

        if self.is_dataset_mode:
            self.dataset_button.config(text="Dataset Mode: ON", bg="#FF9800")
            self.dataset_controls_frame.pack(pady=10, fill=tk.X)
        else:
            self.dataset_button.config(text="Dataset Mode: OFF", bg="#2196F3")
            self.dataset_controls_frame.pack_forget()

    def setup_ui(self):
        # Main container
        main_container = Frame(self.root)
        main_container.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)

        # Left column for drawing
        left_column = Frame(main_container)
        left_column.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))

        # Right column for predictions and controls
        right_column = Frame(main_container)
        right_column.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # ===== LEFT COLUMN CONTENTS =====

        # Title and instructions
        Label(left_column, text="AI-moji challenge", font=("Arial", 18, "bold")).pack(
            pady=5
        )

        Label(left_column, text="Draw an emoji of", font=("Arial", 12)).pack(padx=5)

        self.target_name_label = Label(left_column, text="", font=("Arial", 16, "bold"))
        self.target_name_label.pack(padx=5)

        # Score Display
        self.score_label = Label(
            left_column, text="Score: 0", font=("Arial", 12, "bold")
        )
        self.score_label.pack(pady=1)

        # High Score Display
        self.high_score_label = Label(
            left_column,
            text=f"High Score: {self.high_score}",
            font=("Arial", 12, "bold"),
        )
        self.high_score_label.pack(pady=1)

        # Canvas for drawing
        self.canvas = Canvas(
            left_column,
            width=self.canvas_width,
            height=self.canvas_height,
            bg="white",
            bd=2,
            relief="ridge",
        )
        self.canvas.pack(pady=10)

        # Bind mouse events to canvas
        self.canvas.bind("<Button-1>", self.start_drawing)
        self.canvas.bind("<B1-Motion>", self.draw)
        self.canvas.bind("<ButtonRelease-1>", self.stop_drawing)

        # Button frame
        button_frame = Frame(left_column)
        button_frame.pack(pady=10)

        # Control buttons
        Button(
            button_frame,
            text="Submit",
            command=self.predict_emoji,
            width=10,
            bg="#4CAF50",
            fg="white",
            font=("Arial", 12),
        ).pack(side=tk.LEFT, padx=5)

        Button(
            button_frame,
            text="Clear",
            command=self.clear_canvas,
            width=10,
            bg="#F44336",
            fg="white",
            font=("Arial", 12),
        ).pack(side=tk.LEFT, padx=5)

        # Add Reset Game button
        Button(
            button_frame,
            text="Reset Game",
            command=self.reset_game,
            width=10,
            bg="#FF5722",
            fg="white",
            font=("Arial", 12),
        ).pack(side=tk.LEFT, padx=5)

        # Dataset mode toggle button
        self.dataset_button = Button(
            button_frame,
            text="Dataset Mode: OFF",
            command=self.toggle_dataset_mode,
            width=15,
            bg="#2196F3",
            fg="white",
            font=("Arial", 12),
        )
        self.dataset_button.pack(side=tk.LEFT, padx=5)

        # ===== RIGHT COLUMN CONTENTS =====

        # Top prediction display frame
        self.top_prediction_frame = Frame(right_column, relief="ridge", bd=2)
        self.top_prediction_frame.pack(pady=10, fill=tk.X)

        Label(
            self.top_prediction_frame,
            text="Top Prediction:",
            font=("Arial", 12, "bold"),
        ).pack(pady=(5, 0))

        # Emoji display label
        self.emoji_display = Label(
            self.top_prediction_frame,
            text="",
            font=("Arial", 72),
            width=2,
            height=1,
        )
        self.emoji_display.pack(pady=(0, 5))

        # Emoji name and confidence label
        self.emoji_info = Label(
            self.top_prediction_frame,
            text="Draw and click 'Submit'",
            font=("Arial", 12),
        )
        self.emoji_info.pack(pady=(0, 5))

        # Prediction results frame
        self.result_frame = Frame(right_column)
        self.result_frame.pack(pady=5, fill=tk.BOTH, expand=True)

        Label(
            self.result_frame, text="All Predictions:", font=("Arial", 12, "bold")
        ).pack(anchor="w")

        # Create a frame for the prediction results table
        self.predictions_table_frame = Frame(self.result_frame)
        self.predictions_table_frame.pack(fill=tk.BOTH, expand=True, pady=5)

        # Create a treeview for displaying predictions with probabilities
        self.predictions_tree = ttk.Treeview(
            self.predictions_table_frame,
            columns=("Emoji", "Name", "Probability"),
            show="headings",
            height=8,
        )
        self.predictions_tree.heading("Emoji", text="Emoji")
        self.predictions_tree.heading("Name", text="Name")
        self.predictions_tree.heading("Probability", text="Probability")

        # Configure column widths
        self.predictions_tree.column("Emoji", width=80, anchor="center")
        self.predictions_tree.column("Name", width=150)
        self.predictions_tree.column("Probability", width=150, anchor="center")

        # Add scrollbar
        scrollbar = ttk.Scrollbar(
            self.predictions_table_frame,
            orient="vertical",
            command=self.predictions_tree.yview,
        )
        self.predictions_tree.configure(yscrollcommand=scrollbar.set)

        # Pack the treeview and scrollbar
        self.predictions_tree.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        # Dataset controls (initially hidden)
        self.dataset_controls_frame = Frame(right_column)

        # Emoji label input
        emoji_frame = Frame(self.dataset_controls_frame)
        emoji_frame.pack(pady=5, fill=tk.X)
        Label(emoji_frame, text="Emoji:").pack(side=tk.LEFT, padx=5)
        self.emoji_var = StringVar()
        self.emoji_entry = Entry(
            emoji_frame, textvariable=self.emoji_var, width=5, font=("Arial", 14)
        )
        self.emoji_entry.pack(side=tk.LEFT, padx=5)

        # Emoji name input
        emoji_name_frame = Frame(self.dataset_controls_frame)
        emoji_name_frame.pack(pady=5, fill=tk.X)
        Label(emoji_name_frame, text="Emoji Name:").pack(side=tk.LEFT, padx=5)
        self.emoji_name_var = StringVar()
        Entry(emoji_name_frame, textvariable=self.emoji_name_var, width=20).pack(
            side=tk.LEFT, padx=5
        )

        # Save button
        Button(
            self.dataset_controls_frame,
            text="Save Sample",
            command=self.save_sample,
            width=15,
            bg="#4CAF50",
            fg="white",
            font=("Arial", 12),
        ).pack(pady=5)

        # Dataset stats
        self.dataset_stats_label = Label(
            self.dataset_controls_frame,
            text=f"Dataset size: {len(self.dataset)} samples",
        )
        self.dataset_stats_label.pack(pady=5)

    def select_new_target(self):
        """Select a new target emoji from the class names and update UI."""
        if not hasattr(self, "class_names") or not self.class_names:
            return

        # Keep track of the previous target emoji
        previous_emoji = self.target_emoji

        # If we have more than one class, ensure we don't repeat
        if len(self.class_names) > 1 and previous_emoji is not None:
            # Create a list of emojis excluding the previous one
            available_emojis = [
                emoji for emoji in self.class_names if emoji != previous_emoji
            ]
            self.target_emoji = np.random.choice(available_emojis)
        else:
            # If we only have one class or previous is None, just pick randomly
            self.target_emoji = np.random.choice(self.class_names)

        # Find the name for the selected emoji
        self.target_name = "Unknown"
        for sample in self.dataset:
            if sample["emoji"] == self.target_emoji:
                self.target_name = sample["emojiName"]
                break

        # Update the UI
        self.target_name_label.config(text=self.target_name)

    # ... (keep existing methods unchanged until load_model)
    def transform_emoji_points(self, points):
        input_size = 400
        black_and_white_points = np.zeros((input_size, input_size))
        for point in points:
            x = int(point["x"])
            y = int(point["y"])
            point_inside_the_matrix = (
                x < input_size and y < input_size and x >= 0 and y >= 0
            )
            if point_inside_the_matrix:
                black_and_white_points[y][x] = 1
        return black_and_white_points

    def normalized_value(self, x):
        if x >= 0.01:
            return 1
        else:
            return 0

    def crop_data_sample(self, sample):
        r = sample.any(1)
        if r.any():
            m, n = sample.shape
            c = sample.any(0)
            out = sample[
                r.argmax() : m - r[::-1].argmax(), c.argmax() : n - c[::-1].argmax()
            ]
        else:
            out = np.empty((0, 0), dtype=int)
        return out

    def preprocess_drawing(self, points):
        # Use the same preprocessing steps as in training
        map_to_zero_or_one = np.vectorize(self.normalized_value, otypes=[float])

        # Transform points to image
        sample = self.transform_emoji_points(points)

        # Clean and normalize
        clean_sample = map_to_zero_or_one(sample)

        # Crop
        cropped_sample = self.crop_data_sample(clean_sample)

        # Resize to match the trained model's input size
        if cropped_sample.size == 0:
            # If there's no drawing, return a blank 100x100 image
            return np.zeros((100, 100)).flatten()

        cropped_and_resized_sample = resize(
            cropped_sample, (100, 100), anti_aliasing=False
        )

        # Final cleaning
        clean_final_sample = map_to_zero_or_one(cropped_and_resized_sample)

        # Flatten to match model input
        return clean_final_sample.flatten()

    def load_model(self):
        if not os.path.exists(onnx_file_path):
            print("No model found, but continuing in dataset mode")
        else:
            try:
                self.sess = rt.InferenceSession(onnx_file_path)
                self.input_name = self.sess.get_inputs()[0].name
                self.label_name = self.sess.get_outputs()[0].name
                model_metadata = self.sess.get_modelmeta()
                self.class_names = json.loads(
                    model_metadata.custom_metadata_map.get("classlabels", "[]")
                )
                print(f"Loaded class names: {self.class_names}")

                self.prob_output = None
                for output in self.sess.get_outputs():
                    if (
                        "probability" in output.name.lower()
                        or "prob" in output.name.lower()
                    ):
                        self.prob_output = output.name
                        break
                print("Model loaded successfully")
            except Exception as e:
                messagebox.showwarning("Warning", f"Failed to load model: {str(e)}")
                print(f"Error loading model: {e}")

    def predict_emoji(self):
        if not self.points:
            messagebox.showinfo("Info", "Please draw something first!")
            return
        if self.is_dataset_mode:
            messagebox.showinfo(
                "Dataset Mode", "Use 'Save Sample' to add this drawing to your dataset."
            )
            return

        try:
            processed_drawing = self.preprocess_drawing(self.points)
            # Clear the predictions tree
            for item in self.predictions_tree.get_children():
                self.predictions_tree.delete(item)

            if hasattr(self, "sess"):
                input_data = processed_drawing.reshape(1, -1).astype(np.float32)

                # Debug print to verify model session attributes
                print(f"Input name: {self.input_name}")
                print(f"Label name: {self.label_name}")
                print(f"Probability output name: {self.prob_output}")

                if hasattr(self, "prob_output") and self.prob_output:
                    # Run the model with both label and probability outputs
                    outputs = self.sess.run(
                        [self.label_name, self.prob_output],
                        {self.input_name: input_data},
                    )

                    # Debug print to verify outputs
                    print(f"Outputs shape: {len(outputs)}")
                    print(f"Label output: {outputs[0]}")
                    print(f"Probability output shape: {outputs[1].shape}")

                    predicted_class = outputs[0][0]
                    probabilities = outputs[1][0]

                    # Make sure we have class_names available
                    if not hasattr(self, "class_names") or not self.class_names:
                        print("No class names available")
                        self.emoji_display.config(text="⚠️")
                        self.emoji_info.config(text="No class names available")
                        return

                    class_names = self.class_names
                    print(f"Available class names: {class_names}")

                    # Sort probabilities in descending order
                    sorted_indices = np.argsort(probabilities)[::-1]

                    # Get top prediction
                    top_idx = sorted_indices[0]
                    top_emoji = class_names[top_idx]
                    top_prob = probabilities[top_idx]

                    # Find emoji name
                    emoji_name = f"Class {top_idx}"
                    for sample in self.dataset:
                        if sample.get("emoji") == top_emoji:
                            emoji_name = sample.get("emojiName", emoji_name)
                            break

                    # Update UI with top prediction
                    print(
                        f"Top emoji: {top_emoji}, Name: {emoji_name}, Prob: {top_prob*100:.2f}%"
                    )
                    self.emoji_display.config(text=top_emoji)
                    self.emoji_info.config(text=f"{emoji_name} ({top_prob*100:.2f}%)")

                    # Update predictions tree with all results
                    for i, idx in enumerate(sorted_indices):
                        if i >= 10:  # Limit to top 10
                            break
                        class_label = class_names[idx]
                        emoji_char = class_label
                        emoji_name = f"Class {idx}"

                        # Find emoji name from dataset
                        for sample in self.dataset:
                            if sample.get("emoji") == emoji_char:
                                emoji_name = sample.get("emojiName", emoji_name)
                                break

                        prob_percent = f"{probabilities[idx]*100:.2f}%"
                        print(
                            f"Adding to tree: {emoji_char}, {emoji_name}, {prob_percent}"
                        )

                        # Insert into tree view
                        self.predictions_tree.insert(
                            "", "end", values=(emoji_char, emoji_name, prob_percent)
                        )

                    # Force UI update
                    self.root.update_idletasks()

                    # Check if prediction matches target
                    if hasattr(self, "target_emoji") and self.target_emoji is not None:
                        if top_emoji == self.target_emoji:
                            # Calculate points based on confidence
                            points_earned = round(probabilities[top_idx] * 100)
                            self.update_score(points_earned)
                            self.show_temp_notification(
                                f"+{points_earned} points!", "green"
                            )
                        else:
                            # Penalty for incorrect answers
                            self.update_score(
                                -100
                            )  # Reduced penalty to -10 instead of -100
                            self.show_temp_notification("-100 points!", "red")

                        self.clear_canvas()
                        self.select_new_target()
                else:
                    # Fallback if no probability output is available
                    result = self.sess.run(
                        [self.label_name], {self.input_name: input_data}
                    )
                    predicted_emoji = result[0][0]

                    emoji_name = "Unknown"
                    for sample in self.dataset:
                        if sample.get("emoji") == predicted_emoji:
                            emoji_name = sample.get("emojiName", emoji_name)
                            break

                    self.emoji_display.config(text=predicted_emoji)
                    self.emoji_info.config(text=f"{emoji_name}")
                    self.predictions_tree.insert(
                        "", "end", values=(predicted_emoji, emoji_name, "100%")
                    )
                    self.predictions_tree.insert(
                        "", "end", values=("Note:", "No probability scores", "")
                    )

                    # Check prediction against target
                    if predicted_emoji == self.target_emoji:
                        self.update_score(1)
                        self.show_temp_notification("+1 point!", "green")
                        self.clear_canvas()
                        self.select_new_target()
                    else:
                        self.update_score(-1)
                        self.show_temp_notification("-1 point!", "red")

            else:
                self.emoji_display.config(text="⚠️")
                self.emoji_info.config(text="No model loaded")
                self.predictions_tree.insert(
                    "", "end", values=("⚠️", "No model loaded", "")
                )
                messagebox.showinfo(
                    "Info", "No model loaded. Please train a model first."
                )

        except Exception as e:
            self.emoji_display.config(text="❌")
            self.emoji_info.config(text="Error in prediction")
            messagebox.showerror("Error", f"Prediction failed: {str(e)}")
            self.predictions_tree.insert(
                "", "end", values=("❌", f"Error: {str(e)}", "")
            )
            print(f"Prediction error: {e}")
            import traceback

            traceback.print_exc()

    def show_temp_notification(self, message, color="black", duration=1500):
        """Show a temporary notification that fades after duration"""
        # Create a label for the notification
        notification = Label(
            self.canvas.master,
            text=message,
            font=("Arial", 16, "bold"),
            fg=color,
            bg="white",
            bd=1,
            relief="solid",
            padx=10,
            pady=5,
        )
        # Position it over the canvas
        notification.place(relx=0.5, rely=0.3, anchor="center")

        # Schedule it to be destroyed after duration
        self.root.after(duration, notification.destroy)


if __name__ == "__main__":
    root = tk.Tk()
    app = EmojiDrawingApp(root)
    root.mainloop()
