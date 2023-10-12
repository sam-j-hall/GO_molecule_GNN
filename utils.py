import matplotlib.pyplot as plt
import numpy as np
from bokeh.plotting import figure
from bokeh.palettes import HighContrast3

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

def bokeh_spectra(ml_spectra, true_spectra):
    p = figure(
    x_axis_label = 'Photon Energy (eV)', y_axis_label = 'arb. units',
    x_range = (280,300),
    width = 400, height = 400,
    outline_line_color = 'black', outline_line_width = 2
    )

    p.toolbar.logo = None
    p.toolbar_location = None
    p.min_border = 25

    # x-axis settings
    p.xaxis.ticker.desired_num_ticks = 3
    p.xaxis.axis_label_text_font_size = '24px'
    p.xaxis.major_label_text_font_size = '24px'
    p.xaxis.major_tick_in = 0
    p.xaxis.major_tick_out = 10
    p.xaxis.minor_tick_out = 6
    p.xaxis.major_tick_line_width = 2
    p.xaxis.minor_tick_line_width = 2
    p.xaxis.major_tick_line_color = 'black'
    p.xaxis.minor_tick_line_color = 'black'
    # y-axis settings
    p.yaxis.axis_label_text_font_size = '24px'
    p.yaxis.major_tick_line_color = None
    p.yaxis.minor_tick_line_color = None
    p.yaxis.major_label_text_color = None
    # grid settings
    p.grid.grid_line_color = 'grey'
    p.grid.grid_line_alpha = 0.3
    p.grid.grid_line_width = 1.5
    p.grid.grid_line_dash = "dashed"

    # plot data
    x = np.linspace(280,300,200)
    p.line(x, true_spectra, line_width=3, line_color=HighContrast3[0], legend_label='True')
    p.line(x, ml_spectra, line_width=3, line_color=HighContrast3[1], legend_label='ML Model')

    # legend settings
    p.legend.location = 'bottom_right'
    p.legend.label_text_font_size = '20px'

    return p

def calculate_rse(prediction, true_result):
    
    del_E = 20 / len(prediction)

    numerator = np.sum(del_E * np.power((true_result - prediction),2))

    denominator = np.sum(del_E * true_result)

    return np.sqrt(numerator) / denominator