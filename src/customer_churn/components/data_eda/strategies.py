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
                output.append("Perfect! No missing values detected.")
            else:
                for col, metrics in results["missing_data"].items():
                    output.append(f"  * {col}: {metrics['count']} missing ({metrics['percentage']}%)")

        if "outliers" in results:
            output.append("\n--- OUTLIERS DETECTED ---")
            for col, metrics in results["outliers"].items():
                 output.append(f"  * {col}: {metrics['outlier_count']} outliers ({metrics['percentage']}%)")

        with open(filepath, 'w') as f:
            f.write("\n".join(output))

class JsonReportStrategy(ReportStrategy):
    def generate(self, results: dict, filepath: Path) -> None:
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=4)