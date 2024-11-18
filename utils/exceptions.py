import pandas as pd

def generate_activity_dataframe(activities):
    data = [{
        'timestamp': activity.timestamp,
        'application': activity.application,
        'action': activity.action,
        'notes': activity.notes
    } for activity in activities]
    return pd.DataFrame(data)

def export_to_csv(dataframe, filename='activity_report.csv'):
    dataframe.to_csv(filename, index=False)
    return filename
