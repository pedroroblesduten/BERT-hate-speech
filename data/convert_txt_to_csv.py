import csv

# Define the path to your text file and the output CSV file
input_txt_path = 'classificacao.txt'  # Replace with your text file path
output_csv_path = 'classificacao.csv'

# Open the text file and the output CSV file
with open(input_txt_path, 'r', encoding='utf-8') as txt_file, \
     open(output_csv_path, 'w', newline='', encoding='utf-8') as csv_file:
    
    # Create a CSV writer object
    csv_writer = csv.writer(csv_file)

    # Write the headers to the CSV file
    headers = ['sexism', 'body', 'racism', 'homophobia', 'neutral', 'message']
    csv_writer.writerow(headers)

    # Process each line in the text file, skipping the first line
    for i, line in enumerate(txt_file):
        if i == 0:  # Skip the first line which is the header
            continue

        # Split the line at commas and strip whitespace
        parts = [part.strip() for part in line.split(',')]

        # Separate labels from message
        labels = parts[:5]  # First 5 elements are labels
        message = parts[5] if len(parts) > 5 else ''

        # Write to CSV
        csv_writer.writerow(labels + [message])

print("Conversion completed. The CSV file has been saved to:", output_csv_path)

