"""
Evaluation Module for Medical Coding System

This module provides comprehensive evaluation metrics and analysis
for the medical coding prediction system.
"""

import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, multilabel_confusion_matrix
)
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Tuple, Any
import json
import logging
from pathlib import Path

class MedicalCodingEvaluator:
    """Comprehensive evaluation for medical coding predictions"""
    
    def __init__(self):
        """Initialize the evaluator"""
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Set up plotting style
        plt.style.use('default')
        sns.set_palette("husl")
    
    def calculate_basic_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
        """Calculate basic classification metrics"""
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision_micro': precision_score(y_true, y_pred, average='micro', zero_division=0),
            'precision_macro': precision_score(y_true, y_pred, average='macro', zero_division=0),
            'precision_weighted': precision_score(y_true, y_pred, average='weighted', zero_division=0),
            'recall_micro': recall_score(y_true, y_pred, average='micro', zero_division=0),
            'recall_macro': recall_score(y_true, y_pred, average='macro', zero_division=0),
            'recall_weighted': recall_score(y_true, y_pred, average='weighted', zero_division=0),
            'f1_micro': f1_score(y_true, y_pred, average='micro', zero_division=0),
            'f1_macro': f1_score(y_true, y_pred, average='macro', zero_division=0),
            'f1_weighted': f1_score(y_true, y_pred, average='weighted', zero_division=0)
        }
        
        return metrics
    
    def calculate_top_k_metrics(self, y_true_codes: List[List[str]], 
                               y_pred_codes: List[List[str]], k_values: List[int] = [1, 3, 5]) -> Dict:
        """Calculate top-k accuracy metrics"""
        metrics = {}
        
        for k in k_values:
            correct = 0
            total = 0
            partial_correct = 0
            
            for true_codes, pred_codes in zip(y_true_codes, y_pred_codes):
                top_k_pred = pred_codes[:k]
                
                # Exact match accuracy (all true codes in top-k)
                if all(code in top_k_pred for code in true_codes):
                    correct += 1
                
                # Partial match accuracy (at least one true code in top-k)
                if any(code in top_k_pred for code in true_codes):
                    partial_correct += 1
                
                total += 1
            
            metrics[f'top_{k}_exact_accuracy'] = correct / total if total > 0 else 0
            metrics[f'top_{k}_partial_accuracy'] = partial_correct / total if total > 0 else 0
        
        return metrics
    
    def calculate_code_specific_metrics(self, y_true_codes: List[List[str]], 
                                      y_pred_codes: List[List[str]], 
                                      code_mappings: Dict = None) -> Dict:
        """Calculate metrics for individual codes"""
        # Get all unique codes
        all_true_codes = set()
        all_pred_codes = set()
        
        for codes in y_true_codes:
            all_true_codes.update(codes)
        for codes in y_pred_codes:
            all_pred_codes.update(codes)
        
        all_codes = all_true_codes.union(all_pred_codes)
        
        code_metrics = {}
        
        for code in all_codes:
            # Calculate binary metrics for this code
            y_true_binary = [1 if code in codes else 0 for codes in y_true_codes]
            y_pred_binary = [1 if code in codes else 0 for codes in y_pred_codes]
            
            true_positives = sum(1 for t, p in zip(y_true_binary, y_pred_binary) if t == 1 and p == 1)
            false_positives = sum(1 for t, p in zip(y_true_binary, y_pred_binary) if t == 0 and p == 1)
            false_negatives = sum(1 for t, p in zip(y_true_binary, y_pred_binary) if t == 1 and p == 0)
            
            precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
            recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            code_metrics[code] = {
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'support': sum(y_true_binary),
                'predicted_count': sum(y_pred_binary)
            }
            
            # Add description if available
            if code_mappings:
                if code in code_mappings.get('icd10', {}):
                    code_metrics[code]['description'] = code_mappings['icd10'][code]
                elif code in code_mappings.get('cpt', {}):
                    code_metrics[code]['description'] = code_mappings['cpt'][code]
        
        return code_metrics
    
    def evaluate_model_performance(self, predictions_file: str, ground_truth_file: str,
                                 code_mappings: Dict = None) -> Dict:
        """Comprehensive evaluation of model performance"""
        # Load predictions and ground truth
        if predictions_file.endswith('.csv'):
            pred_df = pd.read_csv(predictions_file)
        else:
            pred_df = pd.read_json(predictions_file)
        
        if ground_truth_file.endswith('.csv'):
            true_df = pd.read_csv(ground_truth_file)
        else:
            true_df = pd.read_json(ground_truth_file)
        
        # Parse code lists
        y_true_icd = true_df['icd_codes'].apply(lambda x: x.split('|') if isinstance(x, str) else []).tolist()
        y_pred_icd = pred_df['icd_codes'].apply(lambda x: x.split('|') if isinstance(x, str) else []).tolist()
        y_true_cpt = true_df['cpt_codes'].apply(lambda x: x.split('|') if isinstance(x, str) else []).tolist()
        y_pred_cpt = pred_df['cpt_codes'].apply(lambda x: x.split('|') if isinstance(x, str) else []).tolist()
        
        # Calculate metrics
        evaluation_results = {
            'icd_metrics': {},
            'cpt_metrics': {},
            'overall_metrics': {}
        }
        
        # Top-k metrics
        evaluation_results['icd_metrics']['top_k'] = self.calculate_top_k_metrics(y_true_icd, y_pred_icd)
        evaluation_results['cpt_metrics']['top_k'] = self.calculate_top_k_metrics(y_true_cpt, y_pred_cpt)
        
        # Code-specific metrics
        evaluation_results['icd_metrics']['code_specific'] = self.calculate_code_specific_metrics(
            y_true_icd, y_pred_icd, code_mappings
        )
        evaluation_results['cpt_metrics']['code_specific'] = self.calculate_code_specific_metrics(
            y_true_cpt, y_pred_cpt, code_mappings
        )
        
        # Overall statistics
        evaluation_results['overall_metrics'] = {
            'total_notes': len(y_true_icd),
            'avg_true_icd_per_note': np.mean([len(codes) for codes in y_true_icd]),
            'avg_pred_icd_per_note': np.mean([len(codes) for codes in y_pred_icd]),
            'avg_true_cpt_per_note': np.mean([len(codes) for codes in y_true_cpt]),
            'avg_pred_cpt_per_note': np.mean([len(codes) for codes in y_pred_cpt]),
            'unique_true_icd_codes': len(set().union(*y_true_icd)),
            'unique_pred_icd_codes': len(set().union(*y_pred_icd)),
            'unique_true_cpt_codes': len(set().union(*y_true_cpt)),
            'unique_pred_cpt_codes': len(set().union(*y_pred_cpt))
        }
        
        return evaluation_results
    
    def create_confusion_matrix_plot(self, y_true: List[List[str]], y_pred: List[List[str]], 
                                   code_type: str, top_codes: int = 10, save_path: str = None):
        """Create confusion matrix visualization for top codes"""
        # Get top codes by frequency
        all_codes = []
        for codes in y_true:
            all_codes.extend(codes)
        
        code_counts = pd.Series(all_codes).value_counts()
        top_code_list = code_counts.head(top_codes).index.tolist()
        
        # Create binary matrices for top codes
        y_true_binary = []
        y_pred_binary = []
        
        for true_codes, pred_codes in zip(y_true, y_pred):
            true_binary = [1 if code in true_codes else 0 for code in top_code_list]
            pred_binary = [1 if code in pred_codes else 0 for code in top_code_list]
            y_true_binary.append(true_binary)
            y_pred_binary.append(pred_binary)
        
        y_true_binary = np.array(y_true_binary)
        y_pred_binary = np.array(y_pred_binary)
        
        # Calculate confusion matrices for each code
        fig, axes = plt.subplots(2, 5, figsize=(20, 8))
        axes = axes.flatten()
        
        for i, code in enumerate(top_code_list):
            if i >= 10:  # Only plot top 10
                break
                
            cm = confusion_matrix(y_true_binary[:, i], y_pred_binary[:, i])
            
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                       xticklabels=['Not Predicted', 'Predicted'],
                       yticklabels=['Not True', 'True'],
                       ax=axes[i])
            axes[i].set_title(f'{code_type} Code: {code}')
        
        # Hide unused subplots
        for i in range(len(top_code_list), 10):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def create_performance_report(self, evaluation_results: Dict, output_file: str = None) -> str:
        """Create a comprehensive performance report"""
        report = []
        report.append("MEDICAL CODING SYSTEM - EVALUATION REPORT")
        report.append("=" * 60)
        report.append("")
        
        # Overall statistics
        overall = evaluation_results['overall_metrics']
        report.append("OVERALL STATISTICS:")
        report.append(f"Total clinical notes processed: {overall['total_notes']}")
        report.append(f"Average ICD codes per note (true): {overall['avg_true_icd_per_note']:.2f}")
        report.append(f"Average ICD codes per note (predicted): {overall['avg_pred_icd_per_note']:.2f}")
        report.append(f"Average CPT codes per note (true): {overall['avg_true_cpt_per_note']:.2f}")
        report.append(f"Average CPT codes per note (predicted): {overall['avg_pred_cpt_per_note']:.2f}")
        report.append(f"Unique ICD codes (true/predicted): {overall['unique_true_icd_codes']}/{overall['unique_pred_icd_codes']}")
        report.append(f"Unique CPT codes (true/predicted): {overall['unique_true_cpt_codes']}/{overall['unique_pred_cpt_codes']}")
        report.append("")
        
        # ICD Performance
        icd_top_k = evaluation_results['icd_metrics']['top_k']
        report.append("ICD-10 PERFORMANCE:")
        report.append(f"Top-1 Exact Accuracy: {icd_top_k['top_1_exact_accuracy']:.3f}")
        report.append(f"Top-3 Exact Accuracy: {icd_top_k['top_3_exact_accuracy']:.3f}")
        report.append(f"Top-5 Exact Accuracy: {icd_top_k['top_5_exact_accuracy']:.3f}")
        report.append(f"Top-1 Partial Accuracy: {icd_top_k['top_1_partial_accuracy']:.3f}")
        report.append(f"Top-3 Partial Accuracy: {icd_top_k['top_3_partial_accuracy']:.3f}")
        report.append(f"Top-5 Partial Accuracy: {icd_top_k['top_5_partial_accuracy']:.3f}")
        report.append("")
        
        # CPT Performance
        cpt_top_k = evaluation_results['cpt_metrics']['top_k']
        report.append("CPT PERFORMANCE:")
        report.append(f"Top-1 Exact Accuracy: {cpt_top_k['top_1_exact_accuracy']:.3f}")
        report.append(f"Top-3 Exact Accuracy: {cpt_top_k['top_3_exact_accuracy']:.3f}")
        report.append(f"Top-5 Exact Accuracy: {cpt_top_k['top_5_exact_accuracy']:.3f}")
        report.append(f"Top-1 Partial Accuracy: {cpt_top_k['top_1_partial_accuracy']:.3f}")
        report.append(f"Top-3 Partial Accuracy: {cpt_top_k['top_3_partial_accuracy']:.3f}")
        report.append(f"Top-5 Partial Accuracy: {cpt_top_k['top_5_partial_accuracy']:.3f}")
        report.append("")
        
        # Top performing codes
        icd_codes = evaluation_results['icd_metrics']['code_specific']
        cpt_codes = evaluation_results['cpt_metrics']['code_specific']
        
        # Sort ICD codes by F1 score
        icd_sorted = sorted(icd_codes.items(), key=lambda x: x[1]['f1'], reverse=True)
        report.append("TOP PERFORMING ICD CODES:")
        for code, metrics in icd_sorted[:10]:
            report.append(f"{code}: F1={metrics['f1']:.3f}, Precision={metrics['precision']:.3f}, Recall={metrics['recall']:.3f}")
        report.append("")
        
        # Sort CPT codes by F1 score
        cpt_sorted = sorted(cpt_codes.items(), key=lambda x: x[1]['f1'], reverse=True)
        report.append("TOP PERFORMING CPT CODES:")
        for code, metrics in cpt_sorted[:10]:
            report.append(f"{code}: F1={metrics['f1']:.3f}, Precision={metrics['precision']:.3f}, Recall={metrics['recall']:.3f}")
        report.append("")
        
        report_text = "\n".join(report)
        
        if output_file:
            with open(output_file, 'w') as f:
                f.write(report_text)
            self.logger.info(f"Report saved to {output_file}")
        
        return report_text
    
    def create_performance_plots(self, evaluation_results: Dict, save_dir: str = None):
        """Create visualization plots for performance analysis"""
        if save_dir:
            save_dir = Path(save_dir)
            save_dir.mkdir(exist_ok=True)
        
        # Plot 1: Top-K Accuracy Comparison
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # ICD Top-K Accuracy
        icd_metrics = evaluation_results['icd_metrics']['top_k']
        k_values = [1, 3, 5]
        icd_exact = [icd_metrics[f'top_{k}_exact_accuracy'] for k in k_values]
        icd_partial = [icd_metrics[f'top_{k}_partial_accuracy'] for k in k_values]
        
        x = np.arange(len(k_values))
        width = 0.35
        
        ax1.bar(x - width/2, icd_exact, width, label='Exact Match', alpha=0.8)
        ax1.bar(x + width/2, icd_partial, width, label='Partial Match', alpha=0.8)
        ax1.set_xlabel('Top-K')
        ax1.set_ylabel('Accuracy')
        ax1.set_title('ICD-10 Top-K Accuracy')
        ax1.set_xticks(x)
        ax1.set_xticklabels([f'Top-{k}' for k in k_values])
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # CPT Top-K Accuracy
        cpt_metrics = evaluation_results['cpt_metrics']['top_k']
        cpt_exact = [cpt_metrics[f'top_{k}_exact_accuracy'] for k in k_values]
        cpt_partial = [cpt_metrics[f'top_{k}_partial_accuracy'] for k in k_values]
        
        ax2.bar(x - width/2, cpt_exact, width, label='Exact Match', alpha=0.8)
        ax2.bar(x + width/2, cpt_partial, width, label='Partial Match', alpha=0.8)
        ax2.set_xlabel('Top-K')
        ax2.set_ylabel('Accuracy')
        ax2.set_title('CPT Top-K Accuracy')
        ax2.set_xticks(x)
        ax2.set_xticklabels([f'Top-{k}' for k in k_values])
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        if save_dir:
            plt.savefig(save_dir / 'top_k_accuracy.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Plot 2: Code Performance Distribution
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # ICD F1 Score Distribution
        icd_codes = evaluation_results['icd_metrics']['code_specific']
        icd_f1_scores = [metrics['f1'] for metrics in icd_codes.values() if metrics['support'] > 0]
        
        ax1.hist(icd_f1_scores, bins=20, alpha=0.7, edgecolor='black')
        ax1.set_xlabel('F1 Score')
        ax1.set_ylabel('Number of Codes')
        ax1.set_title('ICD-10 F1 Score Distribution')
        ax1.axvline(np.mean(icd_f1_scores), color='red', linestyle='--', 
                   label=f'Mean: {np.mean(icd_f1_scores):.3f}')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # CPT F1 Score Distribution
        cpt_codes = evaluation_results['cpt_metrics']['code_specific']
        cpt_f1_scores = [metrics['f1'] for metrics in cpt_codes.values() if metrics['support'] > 0]
        
        ax2.hist(cpt_f1_scores, bins=20, alpha=0.7, edgecolor='black')
        ax2.set_xlabel('F1 Score')
        ax2.set_ylabel('Number of Codes')
        ax2.set_title('CPT F1 Score Distribution')
        ax2.axvline(np.mean(cpt_f1_scores), color='red', linestyle='--', 
                   label=f'Mean: {np.mean(cpt_f1_scores):.3f}')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        if save_dir:
            plt.savefig(save_dir / 'f1_distribution.png', dpi=300, bbox_inches='tight')
        plt.show()

def main():
    """Main function to demonstrate evaluation"""
    # This would normally run with real prediction results
    print("Medical Coding Evaluation System")
    print("=" * 40)
    print("Usage:")
    print("evaluator = MedicalCodingEvaluator()")
    print("results = evaluator.evaluate_model_performance('predictions.csv', 'ground_truth.csv')")
    print("report = evaluator.create_performance_report(results)")
    print("evaluator.create_performance_plots(results)")

if __name__ == "__main__":
    main()
