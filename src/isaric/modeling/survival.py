from lifelines import CoxPHFitter

def fit_cox_model(df, duration_col, event_col):
    """
    Description:
        Fits a Cox Proportional Hazards model for survival analysis.

    Args:
        df (pandas.DataFrame): Input dataset.
        duration_col (str): Column name representing time to event.
        event_col (str): Column name representing event occurrence (1=event, 0=censored).

    Returns:
        lifelines.CoxPHFitter: Fitted Cox model.
    """
    cph = CoxPHFitter()
    cph.fit(df, duration_col=duration_col, event_col=event_col)
    return cph
