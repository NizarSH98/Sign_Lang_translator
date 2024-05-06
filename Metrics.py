import tkinter as tk
from tkinter import ttk
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

def load_and_evaluate_model():
    # Load the test dataset
    test_data = pd.read_csv("dataset\sign_mnist_test\sign_mnist_test.csv")  # Correct path for testing data
    test_labels = test_data['label'].values
    test_images = test_data.iloc[:, 1:].values
    test_images = test_images / 255.0
    test_images = test_images.reshape(-1, 28, 28, 1)

    # Load the trained model
    model = load_model("sign_language_model1.keras")

    # Get predictions
    predictions = np.argmax(model.predict(test_images), axis=-1)

    # Compute confusion matrix
    conf_matrix = confusion_matrix(test_labels, predictions)

    # Compute precision, recall, and F1 score
    report = classification_report(test_labels, predictions, output_dict=True)

    # Display results in a GUI
    conf_matrix_window = tk.Tk()
    
    conf_matrix_window.option_add("*Font", "Helvetica 30")  # Set default font for all widgets

   
    conf_matrix_window.title("Confusion Matrix")
    conf_matrix_window.protocol("WM_DELETE_WINDOW", conf_matrix_window.quit)  # Quit application when this window is closed

    # Create a figure for confusion matrix
    fig_conf = plt.figure(figsize=(7, 5), dpi=240)
    ax_conf = fig_conf.add_subplot(111)
    cax_conf = ax_conf.matshow(conf_matrix, cmap='Blues', origin='lower')
    ax_conf.set_xlabel('Predicted Label')
    ax_conf.set_ylabel('True Label')
    ax_conf.set_xticks(np.arange(len(conf_matrix)))
    ax_conf.set_yticks(np.arange(len(conf_matrix)))

    ax_conf.set_xticklabels(np.arange(len(conf_matrix)))
    ax_conf.set_yticklabels(np.arange(len(conf_matrix))[::-1])  # Reverse the order of y-axis labels

    ax_conf.tick_params(axis='x', direction='out')  # Set direction of x-axis ticks
    ax_conf.tick_params(axis='y', direction='out')  # Set direction of y-axis ticks
    plt.title('Confusion Matrix')
    plt.colorbar(cax_conf)

    # Display confusion matrix figure in Tkinter window
    canvas_conf = FigureCanvasTkAgg(fig_conf, master=conf_matrix_window)
    canvas_conf.draw()
    canvas_conf.get_tk_widget().pack()

    # Classification Report Window
    report_window = tk.Toplevel(conf_matrix_window)
    report_window.title("Classification Report")
    report_window.option_add("*Font", "Helvetica 30")


    # Create a treeview for classification report
    report_tree = ttk.Treeview(report_window)
    report_tree["columns"] = ("precision", "recall", "f1-score")
    report_tree.column("#0", width=640, minwidth=360)
    report_tree.column("precision", anchor=tk.CENTER, width=100)
    report_tree.column("recall", anchor=tk.CENTER, width=100)
    report_tree.column("f1-score", anchor=tk.CENTER, width=100)
    report_tree.heading("#0", text="Class", anchor=tk.W)
    report_tree.heading("precision", text="Precision", anchor=tk.CENTER)
    report_tree.heading("recall", text="Recall", anchor=tk.CENTER)
    report_tree.heading("f1-score", text="F1-Score", anchor=tk.CENTER)

    for class_name, metrics in report.items():
        if class_name.isdigit():  # Class names are digits
            class_name = f"Class {class_name}"
            precision = metrics.get('precision', 0)
            recall = metrics.get('recall', 0)
            f1_score = metrics.get('f1-score', 0)
            # Insert item with specified font
            report_tree.insert("", "end", text=class_name, values=(f"{precision:.2f}", f"{recall:.2f}", f"{f1_score:.2f}"))

    report_tree.pack()

    conf_matrix_window.mainloop()

if __name__ == "__main__":
    load_and_evaluate_model()
