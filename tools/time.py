from datetime import datetime

def time_tool():
    now = datetime.now()
    current_month_year = now.strftime("%B %Y")
    return current_month_year