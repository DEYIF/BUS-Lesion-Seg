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

    dice_values = []
    iou_values = []
    
    dice_gt_07_count = 0
    iou_gt_07_count = 0

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
            if not has_iou:
                print(f"Warning: 'IoU' column not found in {input_path}, skipping IoU stats.")

            for row in reader:
                try:
                    dice_val = float(row['Dice'])
                    dice_values.append(dice_val)
                    if dice_val > 0.7:
                        dice_gt_07_count += 1
                    
                    if has_iou:
                        iou_val = float(row['IoU'])
                        iou_values.append(iou_val)
                        if iou_val > 0.7:
                            iou_gt_07_count += 1
                            
                except ValueError:
                    print(f"Skipping invalid value at row: {row}")
                    continue

        total_count = len(dice_values)
        if total_count == 0:
            print("No valid data found.")
            return

        dice_ratio = dice_gt_07_count / total_count
        dice_stats = calculate_metrics(dice_values, "Dice")
        
        iou_ratio = 0.0
        iou_stats = {}
        if has_iou:
            iou_ratio = iou_gt_07_count / total_count
            iou_stats = calculate_metrics(iou_values, "IoU")

        print("-" * 30)
        print(f"Total Samples: {total_count}")
        print("-" * 30)
        print("Done Statistics:")
        print(f"  > 0.7 Count: {dice_gt_07_count}")
        print(f"  > 0.7 Ratio: {dice_ratio:.4f}")
        for k, v in dice_stats.items():
            if k != 'Count':
                print(f"  {k}: {v:.4f}")
        
        if has_iou:
            print("-" * 30)
            print("IoU Statistics:")
            print(f"  > 0.7 Count: {iou_gt_07_count}")
            print(f"  > 0.7 Ratio: {iou_ratio:.4f}")
            for k, v in iou_stats.items():
                if k != 'Count':
                    print(f"  {k}: {v:.4f}")
        print("-" * 30)

        # Write result to output CSV
        with open(output_path, mode='w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            
            # Header
            header = ['Metric', 'Total_Samples', 'Count_gt_0.7', 'Ratio_gt_0.7', 'Mean', 'Median', 'Variance', 'Std_Dev', 'Min', 'Max']
            writer.writerow(header)
            
            # Dice Row
            dice_row = ['Dice', total_count, dice_gt_07_count, f"{dice_ratio:.4f}"]
            dice_row.extend([f"{dice_stats[k]:.4f}" for k in ['Mean', 'Median', 'Variance', 'Std_Dev', 'Min', 'Max']])
            writer.writerow(dice_row)
            
            # IoU Row
            if has_iou:
                iou_row = ['IoU', total_count, iou_gt_07_count, f"{iou_ratio:.4f}"]
                iou_row.extend([f"{iou_stats[k]:.4f}" for k in ['Mean', 'Median', 'Variance', 'Std_Dev', 'Min', 'Max']])
                writer.writerow(iou_row)
        
        print(f"Statistics written to {output_path}")

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
