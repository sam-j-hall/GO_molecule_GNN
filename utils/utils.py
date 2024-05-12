import matplotlib.pyplot as plt
import numpy as np
from bokeh.plotting import figure
from bokeh.models import SingleIntervalTicker, LinearAxis, NumeralTickFormatter, Span
from bokeh.palettes import HighContrast3
from rdkit import Chem

def plot_spectra(true_spectrum, predicted_spectrum, save_var):
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
    width = 350, height = 350,
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

def bokeh_hist(hist, edges, average, spacing):
    p = figure(
        x_axis_label = 'RSE value', y_axis_label = 'Frequency',
        x_range = (edges[0], edges[-1]), y_range = (0, max(hist)+spacing),
        width = 500, height = 450,
        outline_line_color = 'black', outline_line_width = 2
    )

    p.toolbar.logo = None
    p.toolbar_location = None
    p.min_border = 25

    # --- x-axis settings
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
    p.xaxis[0].ticker.desired_num_ticks = 4
    # --- y-axis settings
    p.yaxis.axis_label_text_font_size = '24px'
    p.yaxis.major_label_text_font_size = '24px'
    p.yaxis.major_tick_in = 0
    p.yaxis.major_tick_out = 10
    p.yaxis.major_tick_line_width = 2
    p.yaxis.major_tick_line_color = 'black'
    p.yaxis.minor_tick_line_color = None
    p.yaxis.major_label_text_color = 'black'
    # --- grid settings
    p.grid.grid_line_color = 'grey'
    p.grid.grid_line_alpha = 0.3
    p.grid.grid_line_width = 1.5
    p.grid.grid_line_dash = "dashed"

    # --- Format x-axis
    ticker = SingleIntervalTicker(interval=20)
    xaxis = LinearAxis(ticker=ticker)
    p.add_layout(xaxis, 'below')

    # --- Plot data
    # --- Add histogram
    p.quad(top=hist, bottom=0, left=edges[:-1], right=edges[1:], fill_color='skyblue', line_color='black')
    # --- Add average line
    vline = Span(location=average, dimension='height', line_color='black', line_width=3, line_dash='dashed')
    p.renderers.extend([vline])

    return(p)

def calculate_rse(prediction, true_result):
    
    del_E = 20 / len(prediction)

    numerator = np.sum(del_E * np.power((true_result - prediction),2))

    denominator = np.sum(del_E * true_result)

    return np.sqrt(numerator) / denominator

def count_funct_group(dataset):

    """
    Goes through a dataset and counts the functional group that the chosen
    atom belongs to
    """

    oh_count = 0
    cooh_count = 0
    epo_count = 0
    ald_count = 0
    ket_count = 0

    for i in range(len(dataset)):
        # --- Turn molecule into rdkit mol
        mol = Chem.MolFromSmiles(dataset[i].smiles)

        # --- Define the functional group fragments to search for
        oh_frag = Chem.MolFromSmarts('[#6][OX2H]')
        cooh_frag = Chem.MolFromSmarts('[CX3](=[OX1])O')
        epo_frag = Chem.MolFromSmarts('[#6]-[O]-[#6]')
        ald_frag = Chem.MolFromSmarts('[CX3H1](=O)')
        ket_frag = Chem.MolFromSmarts('[CX3C](=O)')

        # --- Search mol for functional group matches
        oh_match = mol.GetSubstructMatches(oh_frag)
        cooh_match = mol.GetSubstructMatches(cooh_frag)
        epo_match = mol.GetSubstructMatches(epo_frag)
        ald_match = mol.GetSubstructMatches(ald_frag)
        ket_match = mol.GetSubstructMatches(ket_frag)

        # --- Turn match n-dimensional tuples to 1D list
        oh_list = list(sum(oh_match, ()))
        cooh_list = list(sum(cooh_match, ()))
        epo_list = list(sum(epo_match, ()))
        ald_list = list(sum(ald_match, ()))
        ket_list = list(sum(ket_match, ()))

        # --- Loop through molecule by atom and check if atom 
        # --- matches the atom_num and then count if it belongs to 
        # --- a certain functional group
        for atom in mol.GetAtoms():
            if atom.GetIdx() == dataset[i].atom_num:
                if dataset[i].atom_num in epo_list:
                    epo_count += 1
                elif dataset[i].atom_num in ald_list:
                    ald_count += 1
                elif dataset[i].atom_num in cooh_list:
                    cooh_count += 1
                elif dataset[i].atom_num in oh_list:
                    oh_count += 1
                elif dataset[i].atom_num in ket_list:
                    ket_count += 1

    return oh_count, cooh_count, epo_count, ald_count, ket_count