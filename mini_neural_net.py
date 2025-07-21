import tkinter as tk
import math
import random

# Sigmoid and its derivative
def sigmoid(x):
    return 1 / (1 + math.exp(-x))

def sigmoid_derivative(x):
    s = sigmoid(x)
    return s * (1 - s)

class BackpropGame:
    def __init__(self, master):
        self.master = master
        master.title("Backpropagation Game - Level 2")

        self.in1 = 1.0
        self.in2 = 1.0
        self.target = 1.0
        self.learning_rate = 0.1

        # Initialize weights randomly
        self.w1 = random.uniform(-1, 1)
        self.w2 = random.uniform(-1, 1)
        self.w3 = random.uniform(-1, 1)

        self.canvas = tk.Canvas(master, width=500, height=400, bg="white")
        self.canvas.pack()

        self.output_label = tk.Label(master, text="Output: ", font=("Arial", 14))
        self.output_label.pack()
        self.loss_label = tk.Label(master, text="Loss: ", font=("Arial", 14))
        self.loss_label.pack()
        self.hint_label = tk.Label(master, text="Click 'Train Once' to adjust weights!", fg="blue", font=("Arial", 12))
        self.hint_label.pack()

        self.button_frame = tk.Frame(master)
        self.button_frame.pack()

        tk.Button(self.button_frame, text="Train Once", command=self.train_once).grid(row=0, column=0, padx=5)
        tk.Button(self.button_frame, text="Reset", command=self.reset).grid(row=0, column=1, padx=5)

        self.forward_pass()  # Ensure output and loss are initialized
        self.draw_network()
        self.update_output()

    def forward_pass(self):
        self.h_input = self.w1 * self.in1 + self.w2 * self.in2
        self.h_output = sigmoid(self.h_input)
        self.o_input = self.w3 * self.h_output
        self.output = sigmoid(self.o_input)
        self.loss = (self.target - self.output) ** 2

    def backward_pass(self):
        # Calculate gradients
        dL_dout = -2 * (self.target - self.output)
        dout_dzin = sigmoid_derivative(self.o_input)
        dzin_dw3 = self.h_output

        dL_dw3 = dL_dout * dout_dzin * dzin_dw3

        dzin_dh = self.w3
        dh_dhin = sigmoid_derivative(self.h_input)

        dL_dw1 = dL_dout * dout_dzin * dzin_dh * dh_dhin * self.in1
        dL_dw2 = dL_dout * dout_dzin * dzin_dh * dh_dhin * self.in2

        # Update weights
        self.w1 -= self.learning_rate * dL_dw1
        self.w2 -= self.learning_rate * dL_dw2
        self.w3 -= self.learning_rate * dL_dw3

    def train_once(self):
        self.forward_pass()
        self.backward_pass()
        self.update_output()

    def reset(self):
        self.w1 = random.uniform(-1, 1)
        self.w2 = random.uniform(-1, 1)
        self.w3 = random.uniform(-1, 1)
        self.update_output()

    def update_output(self):
        self.forward_pass()
        self.output_label.config(text=f"Output: {self.output:.3f}")
        self.loss_label.config(text=f"Loss: {self.loss:.4f}")

        if self.loss < 0.01:
            self.hint_label.config(text="🎉 You trained the brain! Loss is very low!", fg="green")
        else:
            self.hint_label.config(text="Click 'Train Once' to adjust weights!", fg="blue")

        self.draw_network()

    def draw_network(self):
        self.canvas.delete("all")

        # Nodes
        self.canvas.create_oval(80, 100, 120, 140, fill="gold")  # IN1
        self.canvas.create_oval(80, 200, 120, 240, fill="gold")  # IN2
        self.canvas.create_oval(220, 150, 260, 190, fill="orange")  # H1
        self.canvas.create_oval(370, 150, 410, 190, fill="red")  # OUT

        # Labels
        self.canvas.create_text(100, 145, text="IN1")
        self.canvas.create_text(100, 245, text="IN2")
        self.canvas.create_text(240, 195, text="H1")
        self.canvas.create_text(390, 195, text="OUT")

        # Lines
        self.canvas.create_line(120, 120, 220, 170, arrow=tk.LAST)
        self.canvas.create_text(170, 130, text=f"w1={self.w1:.2f}")

        self.canvas.create_line(120, 220, 220, 170, arrow=tk.LAST)
        self.canvas.create_text(170, 210, text=f"w2={self.w2:.2f}")

        self.canvas.create_line(260, 170, 370, 170, arrow=tk.LAST)
        self.canvas.create_text(315, 150, text=f"w3={self.w3:.2f}")

        self.canvas.create_text(390, 130, text=f"{self.output:.2f}", fill="black")
        self.canvas.create_text(390, 110, text=f"Loss={self.loss:.2f}", fill="gray")

if __name__ == "__main__":
    root = tk.Tk()
    app = BackpropGame(root)
    root.mainloop()








