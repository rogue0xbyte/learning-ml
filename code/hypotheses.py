import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
from openpyxl import Workbook

def generate_hypotheses(dataframe, num_hypotheses=100):
    hypotheses = []
    
    specificity_levels = []
    specificity_levels.extend(random.sample(range(0, 34), num_hypotheses // 3))
    specificity_levels.extend(random.sample(range(34, 67), num_hypotheses // 3))
    specificity_levels.extend(random.sample(range(67, 101), num_hypotheses // 3))
    random.shuffle(specificity_levels)
    
    for specificity in specificity_levels:
        random_row = dataframe.sample(n=1)
        hypothesis = []
        for col, value in random_row.iloc[0].items():
            if random.randint(0, 100) <= specificity:
                hypothesis.append(f"{col}={value}")
            else:
                hypothesis.append(f"{col}=?")
        
        hypotheses.append(hypothesis)

    return hypotheses[-1::-1]

def MapInstances(hypotheses, instances):
    mapped_instances = []
    
    for idx, hypothesis in enumerate(hypotheses, start=1):
        matched_instances = []
        
        # print(f"Hypothesis {idx}: {hypothesis}")  # Print the hypothesis being checked
        
        for _, instance in instances.iterrows():
            instance_match = True
            for constraint in hypothesis:
                col, value = constraint.split('=')
                if value == '?':
                    pass
                else:
                    if str(instance[col]) != str(value):
                        instance_match = False
                        break

            if instance_match:
                matched_instances.append(instance.tolist())
        
        # print(f"Matched instances: {matched_instances}")  # Print the matched instances for this hypothesis

        
        # Append the dictionary with ID, hypothesis, and matching instances to the list
        mapped_instances.append({"ID": idx, "Hypothesis": hypothesis, "Instances": matched_instances})
    
    return mapped_instances


def store_output_to_excel(hypotheses, mapped_instances, dataframe):
    wb = Workbook()
    sheet_hypotheses = wb.active
    sheet_hypotheses.title = "Hypotheses"

    original_columns = list(dataframe.columns)

    # Store Hypotheses
    hypotheses_columns = ["Hypothesis Number"] + original_columns
    sheet_hypotheses.append(hypotheses_columns)

    for idx, hypothesis in enumerate(hypotheses, start=1):
        simplified_hypothesis = [constraint.split('=')[1] if '=' in constraint else '?' for constraint in hypothesis]
        sheet_hypotheses.append([idx] + simplified_hypothesis)

    sheet_instances = wb.create_sheet(title="Instances")
    sheet_instances.append(original_columns + ["Hypothesis Number"])

    # Store Instances
    instances_data = []  # Store instances data to be written in a batch
    
    for index, instance in dataframe.iterrows():
        matched_hypothesis = 0
        instance_data = instance.tolist()
        
        for idx, hypothesis in enumerate(hypotheses, start=1):
            hypothesis_match = all(str(instance[col]) == str(val.split('=')[1]) if '=' in val else True for col, val in enumerate(hypothesis))
            
            if hypothesis_match:
                matched_hypothesis = idx
                break

        if matched_hypothesis != 0:
            instance_data.append(matched_hypothesis)
            instances_data.append(instance_data)

    # Write instances data in a batch to the sheet
    for instance_row in instances_data:
        sheet_instances.append(instance_row)

    wb.save('output.xlsx')
    print("Output saved to output.xlsx")

def visualize_hypotheses(hypotheses, mapped_instances):
    # Create dot plots for hypotheses and instances
    num_hypotheses = len(hypotheses)
    
    plt.figure(figsize=(8, 6))

    # Plotting hypotheses (Graph 2) with inverted axes
    plt.subplot(2, 1, 2)
    specialization_ratings = [sum(1 for constraint in hypothesis if '?' not in constraint) for hypothesis in hypotheses]
    plt.scatter(range(1, num_hypotheses + 1), specialization_ratings, marker='o', color='black')
    plt.xlabel('Hypothesis Number')
    plt.ylabel('Specialization Rating')
    plt.title('Hypotheses - Specialization Rating')

    # Label specific hypotheses
    label_indices = [1, num_hypotheses // 2, num_hypotheses]
    for i in label_indices:
        plt.text(i, specialization_ratings[i - 1], f'H{i}', ha='right', va='center', color='black')

    # Plotting instances (Graph 1)
    plt.subplot(2, 1, 1)
    plt.xlabel('Hypothesis Number')
    plt.ylabel('Instance Number')
    plt.title('Instances - Hypotheses')

    # Filter instances for the specified three hypotheses
    specified_hypotheses = [1, num_hypotheses // 2, num_hypotheses]
    instances_to_plot = []

    total_instcs = 100  # sum([len(i["Instances"]) for i in mapped_instances])

    # Keep track of plotted instances for each hypothesis
    plotted_instances = set()

    for instance_info in mapped_instances:
        hypothesis_num = instance_info["ID"]
        instances = instance_info["Instances"]
        random_y = random.uniform(0.5, total_instcs + 0.5)
        
        for instance in instances:
            plt.scatter(hypothesis_num, random_y, marker='o', color='grey')  # Plot all instances in grey

        if hypothesis_num in specified_hypotheses:
            plt.scatter(hypothesis_num, random_y, marker='o', color='black')  # Plot instances for specified hypotheses
            plotted_instances.add(hypothesis_num)
            instances_to_plot.extend(instances)

    # Keep track of annotated instances
    annotated_instances = set()
    plotted_hypotheses = set()

    for hypothesis_idx, hypothesis_num in enumerate(specified_hypotheses, start=1):
        for instance_counter, instance_data in enumerate(instances_to_plot, start=1):
            
            if len(annotated_instances) <= 3 and hypothesis_num in specified_hypotheses and hypothesis_num not in plotted_hypotheses:
                # Randomize y-coordinate for instance
                random_y = random.uniform(0.5, total_instcs + 0.5)
                
                plt.scatter(hypothesis_num, random_y, marker='o', color='black')  # Plot instance
            
                instance_id = f'I{random.randint(1000, 9999)}'  # Generate random instance ID
                plt.annotate(instance_id, (hypothesis_num, random_y), textcoords="offset points", xytext=(5, 5), ha='center', fontsize=8, color='black')  # Label instance

                # Print instance details
                print(f"Instance ID: {instance_id}, Instance Data: {instance_data}, Hypothesis ID: H{hypothesis_num}")
                annotated_instances.add(hypothesis_num)
                plotted_hypotheses.add(hypothesis_num)
                break  # Break after annotating one instance for each hypothesis

    # Show the plot
    plt.tight_layout()
    plt.show()