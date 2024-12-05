import optuna
import pickle
import argparse
import matplotlib.pyplot as plt 
import plotly.io as pio
from plotly.io import to_image
study_name = "optuna_studies/tft_study_5/tft_study_5"  # Unique identifier of the study.
storage_name = "sqlite:///{}.db".format(study_name)
study = optuna.load_study(study_name=study_name, storage=storage_name)
#fig, ax = plt.subplots(figsize=(10, 6))

fig= optuna.visualization.plot_optimization_history(study)

fig.update_layout(
    title=dict(
        text="Optimization History",
        x=0.5,  # Center the title
        xanchor='center',
        yanchor='top'
    ),
    legend=dict(
        x=0.70,  # Position the legend inside the figure
        y=0.99,
        #bgcolor='rgba(255, 255, 255, 0.5)'  # Semi-transparent white background for the legend
    ),
    # paper_bgcolor='white',  # Set the paper background to white
    # plot_bgcolor='white',    # Set the plot background to white
    xaxis=dict(
        title="Trial",
        showgrid=True,
        range=[-1, 50]
      ),  # Show gridlines for x-axis
    #     gridcolor='lightgrey',
    #     linecolor="black"  # Set gridline color
    # ),
    yaxis=dict(
        title="Validation loss",
        showgrid=True,
        range=[15, 45]),  # Show gridlines for y-axis
    #     gridcolor='lightgrey',
    #     linecolor="black"  # Set gridline color
     #)
)
#pio.write_image(fig, "optuna_studies/tft_study_5/optimization_history.pdf", format="pdf")

##Does not give deterministic results for the same study probably due to low number of trials
##Did not inlcude in thesis results
fig2= optuna.visualization.plot_param_importances(study)
test = optuna.importance.get_param_importances(study)
print(test)
# fig2.update_layout(
#     title=dict(
#         text="Hpyerparameter Importances",
#         x=0.5,  # Center the title
#         xanchor='center',
#         yanchor='top'
#     ),
    
#     # paper_bgcolor='white',  # Set the paper background to white
#     # plot_bgcolor='white',    # Set the plot background to white
#     xaxis=dict(
        
#         range=[0, 0.40]
#       ),  # Show gridlines for x-axis
#     #     gridcolor='lightgrey',
#     #     linecolor="black"  # Set gridline color
#     # ),
   
# )
pio.write_image(fig2, "optuna_studies/tft_study_5/param_importance.png", format="png")