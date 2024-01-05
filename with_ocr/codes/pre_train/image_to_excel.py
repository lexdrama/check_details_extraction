
import os 
import csv


def create_excel_file(folder_path):

    with open("phase1.csv", 'w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)

        csv_writer.writerow(['filename'])

        for filename in os.listdir(folder_path):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                csv_writer.writerow([filename])

folder_path = 'data/phase1/'

create_excel_file(folder_path)