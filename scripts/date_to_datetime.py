import json
from datetime import datetime
import sys

DATE_JSON = sys.argv[1] #date.json file
DATETIME_JSON = sys.argv[2] #datetime.json file

# Load original date.json
with open(DATE_JSON, "r") as f:
    date_data = json.load(f)

# Define Wuhan reference date
reference_date = datetime(2019, 12, 31)

# Convert to number of days since Wuhan reference
processed_dates = {}
for variant_id, date_str in date_data.items():
    try:
        variant_date = datetime.strptime(date_str, "%Y-%m-%d")
        processed_dates[variant_id] = (variant_date - reference_date).days
    except:
        processed_dates[variant_id] = -1  # For hypothetical or invalid dates

# Save the processed file
with open(DATETIME_JSON, "w") as f:
    json.dump(processed_dates, f, indent=2)

print("Saved processed date.")
