import firebase_admin 
from firebase_admin import db, credentials
import json

cred = credentials.Certificate("bin/db_credentials.json")
firebase_admin.initialize_app(cred, {"databaseURL": "https://senserator-default-rtdb.firebaseio.com/"})


# Read data from the JSON file
with open('Scores/test.json', 'r') as file:
    data = json.load(file)

# Get a reference to the Firebase database
ref = db.reference('regions')  # 'regions' is the node where data will be stored

# Push the data to Firebase Realtime Database
new_data_ref = ref.push(data)

print(f'Data successfully stored under ID: {new_data_ref.key}')


