from lifelines import KaplanMeierFitter
import matplotlib.pyplot as plt

def plot_survival_curve(df, duration_col, event_col, label="Survival Curve"):
    """
    Description:
        Plots Kaplan–Meier survival curve for time-to-event data.

    Args:
        df (pandas.DataFrame): Input dataset.
        duration_col (str): Column representing time to event.
        event_col (str): Column representing event occurrence (1=event, 0=censored).
        label (str): Label for the curve.

    Returns:
        matplotlib.figure.Figure: Survival curve figure.
    """
    kmf = KaplanMeierFitter()
    kmf.fit(df[duration_col], event_observed=df[event_col], label=label)
    fig = kmf.plot_survival_function()
    return fig.get_figure()
