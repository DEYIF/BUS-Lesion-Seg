"""
Script usage:
python3 calculate_stats.py <input_csv> <output_csv>
"""

import csv
import os
import argparse
import statistics

def calculate_metrics(values, metric_name):
    if not values:
        print(f"No data found for {metric_name}.")
        return {}
    
    return {
        'Count': len(values),
        'Mean': statistics.mean(values),
        'Median': statistics.median(values),
        'Variance': statistics.variance(values) if len(values) > 1 else 0.0,
        'Std_Dev': statistics.stdev(values) if len(values) > 1 else 0.0,
        'Min': min(values),
        'Max': max(values)
    }

def calculate_stats_from_csv(input_path, output_path):
    print(f"Reading from {input_path}")

    # Data structure: {'Overall': {'Dice': [], 'IoU': []}, 'Dataset1': {'Dice': [], 'IoU': []}, ...}
    grouped_data = {'Overall': {'Dice': [], 'IoU': []}}
    
    try:
        with open(input_path, mode='r', newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            
            headers = reader.fieldnames
            if not headers:
                print(f"Error: Empty CSV or no headers in {input_path}")
                return

            if 'Dice' not in headers:
                print(f"Error: 'Dice' column not found in {input_path}")
                return
            
            has_iou = 'IoU' in headers
            has_dataset = 'Dataset' in headers

            if not has_iou:
                print(f"Warning: 'IoU' column not found in {input_path}, skipping IoU stats.")

            for row in reader:
                try:
                    dice_val = float(row['Dice'])
                    dataset = row['Dataset'] if has_dataset and row['Dataset'] else "Unknown"
                    
                    # Initialize dataset group if new
                    if dataset not in grouped_data:
                        grouped_data[dataset] = {'Dice': [], 'IoU': []}

                    # Append Dice
                    grouped_data['Overall']['Dice'].append(dice_val)
                    grouped_data[dataset]['Dice'].append(dice_val)
                    
                    if has_iou:
                        iou_val = float(row['IoU'])
                        grouped_data['Overall']['IoU'].append(iou_val)
                        grouped_data[dataset]['IoU'].append(iou_val)
                            
                except ValueError:
                    print(f"Skipping invalid value at row: {row}")
                    continue

        if not grouped_data['Overall']['Dice']:
            print("No valid data found.")
            return

        print("-" * 30)
        print("Calculating Statistics...")

        # Write result to output CSV
        with open(output_path, mode='w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            
            # Header
            header = ['Dataset', 'Metric', 'Total_Samples', 'Count_gt_0.7', 'Ratio_gt_0.7', 'Mean', 'Median', 'Variance', 'Std_Dev', 'Min', 'Max']
            writer.writerow(header)
            
            # Sort datasets: Overall first, then alphabetical
            sorted_datasets = ['Overall'] + sorted([d for d in grouped_data.keys() if d != 'Overall'])

            for dataset in sorted_datasets:
                data = grouped_data[dataset]
                total_count = len(data['Dice'])
                
                # Dice Stats
                dice_values = data['Dice']
                dice_gt_07_count = sum(1 for v in dice_values if v > 0.7)
                dice_ratio = dice_gt_07_count / total_count if total_count > 0 else 0
                dice_stats = calculate_metrics(dice_values, "Dice")
                
                dice_row = [dataset, 'Dice', total_count, dice_gt_07_count, f"{dice_ratio:.4f}"]
                dice_row.extend([f"{dice_stats.get(k, 0):.4f}" for k in ['Mean', 'Median', 'Variance', 'Std_Dev', 'Min', 'Max']])
                writer.writerow(dice_row)

                # IoU Stats
                if has_iou:
                    iou_values = data['IoU']
                    iou_gt_07_count = sum(1 for v in iou_values if v > 0.7)
                    iou_ratio = iou_gt_07_count / total_count if total_count > 0 else 0
                    iou_stats = calculate_metrics(iou_values, "IoU")

                    iou_row = [dataset, 'IoU', total_count, iou_gt_07_count, f"{iou_ratio:.4f}"]
                    iou_row.extend([f"{iou_stats.get(k, 0):.4f}" for k in ['Mean', 'Median', 'Variance', 'Std_Dev', 'Min', 'Max']])
                    writer.writerow(iou_row)
        
        print(f"Statistics written to {output_path}")
        print("-" * 30)


    except FileNotFoundError:
        print(f"File not found: {input_path}")
    except Exception as e:
        print(f"An error occurred: {e}")

def main():
    parser = argparse.ArgumentParser(description="Calculate Dice and IoU score statistics from a CSV file.")
    parser.add_argument("input_csv", help="Path to the input CSV file (e.g., metrics.csv)")
    parser.add_argument("output_csv", help="Path to save the output statistics CSV (e.g., statistic.csv)")
    
    args = parser.parse_args()
    
    calculate_stats_from_csv(args.input_csv, args.output_csv)

if __name__ == "__main__":
    main()
