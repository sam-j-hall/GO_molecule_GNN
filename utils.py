import matplotlib.pyplot as plt
import numpy as np

def plot_spectra(predicted_spectrum, true_spectrum, save_var):
    """
    Plots the predicted spectrum and true spectrum on a graph.
    
    :param predicted_spectrum: Array containing the predicted spectrum values.
    :param true_spectrum: Array containing the true spectrum values.
    """
    x = np.linspace(280,300,200)
    # Plot the true spectrum in blue
    plt.plot(x, true_spectrum, color='blue', label='True Spectrum')

    # Plot the predicted spectrum in red
    plt.plot(x, predicted_spectrum, color='red', label='Predicted Spectrum')

    # Set labels and title
    plt.xlabel('Energy')
    plt.ylabel('Intensity')
    plt.title('Comparison of True and Predicted Spectra')

    # Add legend
    plt.legend()
    
    if save_var==1:
        plt.savefig('pred_spec.png')
        
    # Show the plot
    plt.show()
    

def plot_learning_curve(num_epochs, train_loss, val_loss):
    
    """
    Plots the learning curve showing the validation and training losses decrease over number of epochs.
    
    :param num_epochs: Int containing the number of training epochs.
    :param predicted_spectrum: List containing the training loss values.
    :param true_spectrum: List containing the validation loss values.
    """
    
    epochs = range(0, num_epochs)

    plt.plot(epochs, train_loss, label='Train Loss')
    plt.plot(epochs, val_loss, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()