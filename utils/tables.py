import warnings
import numpy as np
from tabulate import tabulate

import pandas as pd




def print_metric_table(train_acc, val_acc, train_miou, val_miou, mapping, epoch):
    headers = ['Class','Train Acc', 'Val Acc', 'Train MIoU', 'Val MIoU']

    table = []
    
    for idx in range(len(train_acc)):
        acc_train = f"{100*train_acc[idx]:4.3f}"
        acc_val   = f"{100*val_acc[idx]:4.3f}"

        miou_train = f"{train_miou[idx]:2.3f}"
        miou_val   = f"{val_miou[idx]:2.3f}"

        table.append([
            str(mapping[idx]),
            acc_train,
            acc_val,
            miou_train,
            miou_val
        ])
    
    print(tabulate(table, headers=headers, tablefmt="pretty"))
    
def print_test_metric_table(test_acc, test_miou, 
                            test_recall, test_precision,
                            test_f1score, test_support,
                            class_names):
      
    headers = ['Class','Acc', 'mIoU', "Recall", "Precision", "F1-Score", "Support"]

    table = []
    
    for idx in range(len(test_acc)):
        acc       = f"{100*test_acc[idx]:4.3f}"
        miou      = f"{test_miou[idx]:2.3f}"
        recall    = f"{test_recall[idx]:2.3f}"
        precision = f"{test_precision[idx]:2.3f}"
        f1score   = f"{test_f1score[idx]:2.3f}"
        support   = f"{int(test_support[idx]):10d}"
        
        table.append([
            str(class_names[idx]),
            acc,
            miou,
            recall,
            precision,
            f1score,
            support
        ])
        
    print(tabulate(table, headers=headers, tablefmt="pretty"))
    
def print_test_macro_table(report):
    
    metrics_names = ["recall", "precision", "f1-score", "support"]
        
    macro_avg      = [np.round(report.get("macro avg").get(k, 0.0),3)    for k in metrics_names]
    weighted_avg   = [np.round(report.get("weighted avg").get(k, 0.0),3) for k in metrics_names]
        
    headers = ['Metric', "Recall", "Precision", "F1-Score", "Support"]

    table = [
        ["Macro Avg.", *macro_avg],
        ["Weighted Avg.", *weighted_avg],
    ]
        
    print(tabulate(table, headers=headers, tablefmt="pretty"))
    
    
    
def print_class_distribution_table(
    class_names: dict,
    train_dist: dict, train_weights: dict, train_count: dict,
    val_dist: dict, val_weights: dict, val_count: dict,
    test_dist: dict, test_weights: dict, test_count: dict,
    include_test_for_final_distribution:bool=True
):
    all_classes = sorted(class_names.keys())
    
    total_train = sum([v for _, v in train_count.items()])
    total_val = sum([v for _, v in val_count.items()])
    total_test = sum([v for _, v in test_count.items()])
    
    if include_test_for_final_distribution:
        total_pixeis = total_train + total_val + total_test
        sum_c = sum((train_count.get(c, 0) + val_count.get(c, 0) + test_count.get(c, 0)) for c in all_classes)
    else:
        total_pixeis = total_train + total_val
        sum_c = sum((train_count.get(c, 0) + val_count.get(c, 0)) for c in all_classes)
    
    table = []

    for c in all_classes:
        name = class_names.get(c, f"Class {c}")

        train_d = train_dist.get(c, 0.0)
        train_w = train_weights.get(c, 0.0)
        train_c = train_count.get(c, 0.0)
        
        val_d = val_dist.get(c, 0.0)
        val_w = val_weights.get(c, 0.0)
        val_c = val_count.get(c, 0.0)
        
        test_d = test_dist.get(c, 0.0)
        test_w = test_weights.get(c, 0.0)
        test_c = test_count.get(c, 0.0)

        if include_test_for_final_distribution:
            total_d = 100*(train_c+val_c+test_c)/total_pixeis 
            total_w = (train_w * train_c + val_w * val_c + test_w * test_c) / (train_c + val_c + test_c + 1e-8) 
        else:
            total_d = 100*(train_c+val_c)/total_pixeis 
            total_w = (train_w * train_c + val_w * val_c) / (train_c + val_c + 1e-8) 
            
        table.append([
            name,
            f"{train_d:6.4f}%", f"{train_w:6.4f}",
            f"{val_d:6.4f}%",   f"{val_w:6.4f}",
            f"{test_d:6.4f}%",  f"{test_w:6.4f}",
            f"{total_d:6.4f}%", f"{total_w:6.4f}",
        ])

    headers = [
        "Class",
        "Train Dist.", "Train Weight",
        "Val Dist.", "Val Weight",
        "Test Dist.", "Test Weight",
        "Total Dist.", "Total Weight",
    ]

    print(tabulate(table, headers=headers, tablefmt="pretty"))
    
def print_report_table(report_dict):
    print(pd.DataFrame.from_dict(report_dict, orient='index').round(4))