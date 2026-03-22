import os
import random
import csv


folder_path = "data/data_sources"
number_files =5
number_rows = 1000000


# Function to generate random data for each row
def generate_random_data():
    pregnancies = random.randint(0, 15)
    glucose = random.randint(70, 200)
    bloodpressure = random.randint(40, 120)
    skinthickness = random.randint(0, 60)
    insulin = random.randint(0, 300)
    bmi = round(random.uniform(18.0, 45.0), 1)
    dpf = round(random.uniform(0.05, 2.5), 3)
    age = random.randint(18, 80)
    outcome = random.randint(0, 1)  # Outcome is either 0 or 1
    
    return [pregnancies, glucose, bloodpressure, skinthickness, insulin, bmi, dpf, age, outcome]

# Function to create 10 files with 1000 rows each
def create_files():
    for file_num in range(1, number_files):
        filename = f"file_{file_num}.csv"
        with open(filename, mode='w', newline='') as file:
            writer = csv.writer(file)
            
            # Write the header
            writer.writerow(['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome'])
            
            # Write 1000 rows of random data
            for _ in range(number_rows):
                writer.writerow(generate_random_data())

# Function to create a folder and write the files in it
def create_files_with_path(folder_path):
    # Create the folder if it doesn't exist
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    
    # Write 10 files with 1000 rows each
    for file_num in range(1, number_files):
        filename = os.path.join(folder_path, f"file_{file_num}.csv")
        with open(filename, mode='w', newline='') as file:
            writer = csv.writer(file)
            # Write the header
            writer.writerow(['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome'])
            # Write 1000 rows of random data
            for _ in range(number_rows):
                writer.writerow(generate_random_data())

if __name__ == "__main__":
    # Call the function to create the files
    create_files_with_path(folder_path)
