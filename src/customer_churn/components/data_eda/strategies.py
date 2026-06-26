from abc import ABC, abstractmethod
import json
from pathlib import Path

class ReportStrategy(ABC):
    @abstractmethod
    def generate(self, results: dict, filepath: Path) -> None:
        pass


class TextReportStrategy(ReportStrategy):
    def generate(self, results: dict, filepath: Path) -> None:
        output = ["=" * 60, "               AUTOMATED EDA REPORT               ", "=" * 60]
        
        if "num_rows" in results:
            output.extend([
                "\n--- DATASET OVERVIEW ---",
                f"Rows: {results['num_rows']} | Columns: {results['num_cols']}",
                f"Memory Footprint: {results['memory_usage_mb']:.2f} MB"
            ])
            
        if "missing_data" in results:
            output.append("\n--- MISSING DATA ---")
            if not results["missing_data"]:
                output.append("  Perfect! No missing values detected.")
            else:
                for col, metrics in results["missing_data"].items():
                    output.append(f"  * {col}: {metrics['count']} missing ({metrics['percentage']}%)")

        if "outliers" in results:
            output.append("\n--- OUTLIERS DETECTED ---")
            if not results["outliers"]:
                output.append("  No severe outliers caught with current threshold settings.")
            else:
                for col, metrics in results["outliers"].items():
                     output.append(f"  * {col}: {metrics['outlier_count']} outliers ({metrics['percentage']}%)")

        if "mutual_information_scores" in results:
            output.append("\n--- FEATURE IMPORTANCE (MUTUAL INFORMATION) ---")
            if isinstance(results["mutual_information_scores"], dict):
                for col, score in results["mutual_information_scores"].items():
                    output.append(f"  * {col}: {score}")
            else:
                output.append(f"  {results['mutual_information_scores']}")

        if "vif_scores" in results:
            output.append("\n--- MULTICOLLINEARITY (VIF SCORES) ---")
            if isinstance(results["vif_scores"], dict):
                for col, score in results["vif_scores"].items():
                    status = "⚠️ HIGH" if score > 5.0 else "OK"
                    output.append(f"  * {col}: {score} [{status}]")
            else:
                output.append(f"  {results['vif_scores']}")

        with open(filepath, 'w', encoding='utf-8') as f:
            f.write("\n".join(output))

class JsonReportStrategy(ReportStrategy):
    def generate(self, results: dict, filepath: Path) -> None:
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=4)