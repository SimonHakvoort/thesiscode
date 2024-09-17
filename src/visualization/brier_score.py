from typing import Tuple, Union
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from src.climatology.climatology import Climatology
from src.linreg_emos.emos import BaseForecastModel



def make_brier_skill_plot(basemodel: BaseForecastModel, 
                             models: dict[str, BaseForecastModel], 
                             data: tf.data.Dataset, 
                             values: np.ndarray, 
                             xlim: Tuple[float,float] = None, 
                             ylim: Tuple[float,float] = None, 
                             title: str = None, 
                             name_base_model: str = 'Reference Model') -> None:
    """
    Plots the Brier skill score (BSS) for the models, which is a dict for the numbers in values. 
    Evaluates the performance for a single batch in data.

    Arguments:
        basemodel (BaseForecastModel): reference model
        models (dict[str, BaseForecastModel]): models to compare to basemodel
        data (tf.data.Dataset): data to compute Brier scores.
        values (np.array): values to compute the BSS.
        xlim (tuple, optional): tuple specifying the range of the x-axis.
        ylim (tuple, optional): tuple specifying the range of the y-axis.
        title (str, optional: Title for the plot.
        name_base_model (str, optional): name for the reference model in the legend.

    Returns:
        None
    """
    brier_base_model = basemodel.Brier_Score(data, values)
    for model in models:
        brier_scores = models[model].Brier_Score(data, values)
        brier_skill_scores = 1 - brier_scores / brier_base_model
        plt.plot(values, brier_skill_scores, label = model)

    # print a striped black horizontal line at y=0
    plt.axhline(0, color='black', linestyle='--', label=name_base_model)

    plt.xlabel('wind speed threshold (m/s)')
    plt.ylabel('Brier skill score')
    if values[0] < 0.5:
        plt.xlim(0, values[-1])
    else:
        plt.xlim(values[0], values[-1])
        
    if ylim != None:
        plt.ylim(ylim[0], ylim[1])
    
    if title is not None:
        plt.title(title)

    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.legend()
    plt.show()


def make_bootstrap_brier(base_model: BaseForecastModel, 
                         models: dict[str, BaseForecastModel], 
                         data: tf.data.Dataset,
                         values: np.ndarray, 
                         ylim: Union[Tuple[float, float], None] = None, 
                         bootstrap_size: int = 1000, 
                         title: str = None, 
                         name_base_model: str = None) -> None:
    """
    Plots the bootstrapped Brier scores in a single plot. 
    For each model in dict we compute the Brier scores at values plus a small constant and then compute the BSS.
    We add the small constant to ensure that the error bars do not overlap, and vary the constant per model.

    Arguments:
        base_model (BaseForecastModel): model to use as reference model.
        models (dict[str, BaseForecastModel]): BaseForecastModel instances to compare to base_model.
        data (tf.data.Dataset): data for which we compute the bootstrapped BS.
        values (np.ndarray): array at which we want to compute the bootstrapped BS.
        ylim (Union[Tuple[float, float], None], optional): limit of the y-axis of the plot.
        bootstrap_size (int, optional): size of the bootstrap.
        title (str, optional): title for the plot.
        name_base_model (str, optional): the name of the base_model to put in the legend.

    Returns:
        None.
    """
    scores = {key: np.zeros((bootstrap_size, len(values))) for key in models.keys()}

    base_model_scores = base_model.seperate_Brier_Score(data, values)

    all_brier_scores_dict = {}

    for i, name in enumerate(models.keys()):
        new_values = values + (i + 1) * 0.1
        base_scores_new_values = base_model.seperate_Brier_Score(data, new_values)
        other_model_scores = models[name].seperate_Brier_Score(data, new_values)

        all_brier_scores_dict[name] = (base_scores_new_values, other_model_scores)

    data_size = base_model_scores.shape[1]

    scores = {key: np.zeros((bootstrap_size, len(values))) for key in models.keys()}

    for i in range(bootstrap_size):
        samples_selected = np.random.choice(data_size, data_size, replace=True)

        for name in all_brier_scores_dict.keys():
            scores_base, scores_other_model = all_brier_scores_dict[name]

            selected_scores_base = np.take(scores_base, samples_selected, axis=1)
            selected_scores_other = np.take(scores_other_model, samples_selected, axis=1)

            avg_scores_base = np.mean(selected_scores_base, axis=1)
            avg_scores_other = np.mean(selected_scores_other, axis=1)

            scores[name][i, :] = 1 - avg_scores_other / avg_scores_base


    for i, name in enumerate(models.keys()):
        score = scores[name]

        new_values = values + (i + 1) * 0.1

        mean = np.mean(score, axis=0)
        std = np.std(score, axis=0)

        plt.errorbar(new_values, y=mean, yerr=std, capsize=2, label=name)


    base_bss_mean = [0 for i in range(len(values))]

    # plt.errorbar(values, y=base_bss_mean, yerr=base_bss_std, capsize=2, label='Base Model')
    if isinstance(base_model, Climatology):
        plt.plot(values, base_bss_mean, label='Climatology', color='black', linestyle='--')
    else:
        if name_base_model is None:
            naming = 'Reference Model'
        else:
            naming = name_base_model
        plt.plot(values, base_bss_mean, label=naming, color='black', linestyle='--')
    
    plt.xlim(0, values[-1])
    if ylim is not None:
        plt.ylim(ylim[0], ylim[1])
    plt.xlabel('Threshold')
    plt.ylabel('Brier Skill Scores')
    plt.legend()
    if title is not None:
        plt.title(title)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.show()




        



        










