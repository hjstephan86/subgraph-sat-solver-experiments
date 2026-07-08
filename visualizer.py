"""
Visualizer für Subgraph-SAT-Solver Experiment-Ergebnisse

Erzeugt Diagramme mit matplotlib und exportiert als SVG/PDF für LaTeX
"""

import json
import statistics
from pathlib import Path
from typing import List, Dict, Optional
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.ticker import FuncFormatter
import numpy as np
from experiment_runner import ExperimentResult


class ExperimentVisualizer:
    """Visualisiert Experiment-Ergebnisse"""
    
    # LaTeX-kompatible Schriftarten
    FONT_CONFIG = {
        'font.family': 'serif',
        'font.serif': ['Computer Modern'],
        'text.usetex': False,  # Deaktiviert bei LaTeX-Kompatibilität
        'figure.figsize': (10, 6),
        'figure.dpi': 300,
        'savefig.dpi': 300,
        'lines.linewidth': 2,
        'lines.markersize': 8,
        'axes.labelsize': 12,
        'axes.titlesize': 14,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'grid.alpha': 0.3
    }
    
    # Farben für verschiedene Circuit-Typen
    COLORS = {
        'sequential': '#1f77b4',  # Blau
        'parallel': '#ff7f0e',    # Orange
        'grid': '#2ca02c',        # Grün
        'random': '#d62728'       # Rot
    }
    
    def __init__(self, results: List[ExperimentResult], output_dir: str = "."):
        """
        Initialisiert Visualizer
        
        Args:
            results: Liste von ExperimentResult Objekten
            output_dir: Ausgabeverzeichnis für Plots
        """
        plt.rcParams.update(self.FONT_CONFIG)
        self.results = results
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    @staticmethod
    def load_results(json_path: str) -> List[ExperimentResult]:
        """Lädt Ergebnisse aus JSON-Datei"""
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        results = [ExperimentResult(**item) for item in data]
        return results
    
    def _group_results(self) -> Dict[str, List[ExperimentResult]]:
        """Gruppiert Ergebnisse nach Circuit-Typ"""
        grouped = {}
        for result in self.results:
            if result.circuit_type not in grouped:
                grouped[result.circuit_type] = []
            grouped[result.circuit_type].append(result)
        return grouped
    
    def plot_runtime_vs_size(self, figsize: tuple = (12, 6)):
        """Plottet Runtime vs. Circuit-Größe"""
        fig, ax = plt.subplots(figsize=figsize)
        
        grouped = self._group_results()
        
        for circuit_type, results in grouped.items():
            # Sortiere nach Größe
            results_sorted = sorted(results, key=lambda r: r.num_gates)
            
            sizes = [r.num_gates for r in results_sorted]
            runtimes = [r.runtime_ms for r in results_sorted]
            
            ax.plot(sizes, runtimes, 
                   marker='o', 
                   label=circuit_type.capitalize(),
                   color=self.COLORS.get(circuit_type, '#000000'),
                   alpha=0.7)
        
        ax.set_xlabel('Circuit Size (Number of Gates)', fontsize=12)
        ax.set_ylabel('Runtime (ms)', fontsize=12)
        ax.set_title('Subgraph-SAT-Solver: Runtime vs. Circuit Size', fontsize=14, fontweight='bold')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        ax.set_yscale('log')
        
        plt.tight_layout()
        return fig
    
    def plot_runtime_distribution(self, figsize: tuple = (12, 6)):
        """Plottet Runtime-Verteilung als Box-Plot"""
        fig, ax = plt.subplots(figsize=figsize)
        
        grouped = self._group_results()
        data = []
        labels = []
        colors = []
        
        for circuit_type in sorted(grouped.keys()):
            results = grouped[circuit_type]
            runtimes = [r.runtime_ms for r in results if r.runtime_ms > 0]
            if runtimes:
                data.append(runtimes)
                labels.append(circuit_type.capitalize())
                colors.append(self.COLORS.get(circuit_type, '#000000'))
        
        if not data or len(data) == 0:
            print("Warning: No data available to plot runtime distribution.")
            return fig
            
        bp = ax.boxplot(data, tick_labels=labels, patch_artist=True)
        
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        ax.set_ylabel('Runtime (ms)', fontsize=12)
        ax.set_title('Subgraph-SAT-Solver: Runtime Distribution', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_yscale('log')
        
        plt.tight_layout()
        return fig
    
    def plot_gates_vs_runtime(self, figsize: tuple = (12, 6)):
        """Plottet Anzahl Gates vs. Runtime (Scatter)"""
        fig, ax = plt.subplots(figsize=figsize)
        
        grouped = self._group_results()
        
        for circuit_type, results in grouped.items():
            gates = [r.num_gates for r in results]
            runtimes = [r.runtime_ms for r in results]
            
            ax.scatter(gates, runtimes,
                      label=circuit_type.capitalize(),
                      color=self.COLORS.get(circuit_type, '#000000'),
                      alpha=0.6,
                      s=100)
        
        ax.set_xlabel('Number of Gates', fontsize=12)
        ax.set_ylabel('Runtime (ms)', fontsize=12)
        ax.set_title('Subgraph-SAT-Solver: Gates vs. Runtime', fontsize=14, fontweight='bold')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        ax.set_xscale('log')
        ax.set_yscale('log')
        
        plt.tight_layout()
        return fig
    
    def plot_clauses_histogram(self, figsize: tuple = (12, 6)):
        """Histogramm der CNF-Klausel-Verteilung"""
        fig, ax = plt.subplots(figsize=figsize)
        
        grouped = self._group_results()
        
        for circuit_type in sorted(grouped.keys()):
            results = grouped[circuit_type]
            clauses = [r.num_clauses for r in results]
            ax.hist(clauses, alpha=0.5, label=circuit_type.capitalize(),
                   color=self.COLORS.get(circuit_type, '#000000'))
        
        ax.set_xlabel('Number of Clauses', fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)
        ax.set_title('Subgraph-SAT-Solver: CNF Clause Distribution', fontsize=14, fontweight='bold')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        return fig
    
    def plot_performance_comparison(self, figsize: tuple = (12, 7)):
        """Vergleicht Performance verschiedener Circuit-Typen"""
        grouped = self._group_results()
        
        circuit_types = sorted(grouped.keys())
        
        fig, axes = plt.subplots(1, 3, figsize=figsize)
        
        # Durchschnittliche Runtime
        runtimes = [np.mean([r.runtime_ms for r in grouped[ct]]) for ct in circuit_types]
        axes[0].bar(range(len(circuit_types)), runtimes, color=[self.COLORS.get(ct, '#7f8c8d') for ct in circuit_types])
        axes[0].set_ylabel('Avg Runtime (ms)', fontsize=11)
        axes[0].set_title('Average Runtime', fontsize=12, fontweight='bold')
        axes[0].set_xticks(range(len(circuit_types)))
        axes[0].set_xticklabels([ct.capitalize() for ct in circuit_types], rotation=15)
        axes[0].grid(True, alpha=0.3, axis='y')
        
        # Durchschnittliche Gates
        gates = [np.mean([r.num_gates for r in grouped[ct]]) for ct in circuit_types]
        axes[1].bar(range(len(circuit_types)), gates, color=[self.COLORS.get(ct, '#7f8c8d') for ct in circuit_types])
        axes[1].set_ylabel('Avg Gates', fontsize=11)
        axes[1].set_title('Average Gates', fontsize=12, fontweight='bold')
        axes[1].set_xticks(range(len(circuit_types)))
        axes[1].set_xticklabels([ct.capitalize() for ct in circuit_types], rotation=15)
        axes[1].grid(True, alpha=0.3, axis='y')
        
        # Durchschnittliche Clauses
        clauses = [np.mean([r.num_clauses for r in grouped[ct]]) for ct in circuit_types]
        axes[2].bar(range(len(circuit_types)), clauses, color=[self.COLORS.get(ct, '#7f8c8d') for ct in circuit_types])
        axes[2].set_ylabel('Avg Clauses', fontsize=11)
        axes[2].set_title('Average Clauses', fontsize=12, fontweight='bold')
        axes[2].set_xticks(range(len(circuit_types)))
        axes[2].set_xticklabels([ct.capitalize() for ct in circuit_types], rotation=15)
        axes[2].grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        return fig
    
    def plot_scaling_analysis(self, figsize: tuple = (14, 10)):
        """Analysiert Skalierungsverhalten"""
        grouped = self._group_results()
        
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        axes = axes.flatten()
        
        for idx, (circuit_type, results) in enumerate(sorted(grouped.items())):
            if idx >= 4:
                break
            
            # Sortiere nach Größe
            results_sorted = sorted(results, key=lambda r: r.num_gates)
            
            sizes = [r.num_gates for r in results_sorted]
            runtimes = [r.runtime_ms for r in results_sorted]
            
            ax = axes[idx]
            ax.plot(sizes, runtimes, marker='o', color=self.COLORS.get(circuit_type, '#7f8c8d'),
                   linewidth=2, markersize=8)
            
            # Fit curve if data is sufficient
            if len(sizes) > 2:
                try:
                    z = np.polyfit(np.log(sizes), np.log(runtimes), 1)
                    p = np.poly1d(z)
                    x_smooth = np.logspace(np.log10(min(sizes)), np.log10(max(sizes)), 100)
                    ax.plot(x_smooth, np.exp(p(np.log(x_smooth))), '--', alpha=0.5, 
                           color=self.COLORS.get(circuit_type, '#7f8c8d'), label='Fit (exp)')
                except Exception:
                    pass
            
            ax.set_xlabel('Circuit Size', fontsize=11)
            ax.set_ylabel('Runtime (ms)', fontsize=11)
            ax.set_title(f'{circuit_type.capitalize()} Scaling', fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.legend()
        
        plt.tight_layout()
        return fig
    
    def export_all_plots(self, formats: List[str] = ['svg', 'pdf']):
        """Exportiert alle Plots"""
        plots = [
            ('runtime_vs_size', self.plot_runtime_vs_size()),
            ('runtime_distribution', self.plot_runtime_distribution()),
            ('gates_vs_runtime', self.plot_gates_vs_runtime()),
            ('clauses_histogram', self.plot_clauses_histogram()),
            ('performance_comparison', self.plot_performance_comparison()),
            ('scaling_analysis', self.plot_scaling_analysis())
        ]
        
        for name, fig in plots:
            for fmt in formats:
                filepath = self.output_dir / f"{name}.{fmt}"
                fig.savefig(filepath, format=fmt, bbox_inches='tight')
                print(f"✓ Saved: {filepath}")
            plt.close(fig)
    
    def create_summary_report(self, output_file: str = "report.txt"):
        """Erstellt Text-Report der Ergebnisse"""
        grouped = self._group_results()
        
        report = "=" * 70 + "\n"
        report += "SUBGRAPH-SAT-SOLVER EXPERIMENT REPORT\n"
        report += "=" * 70 + "\n\n"
        
        report += f"Total Experiments: {len(self.results)}\n\n"
        
        for circuit_type in sorted(grouped.keys()):
            results = grouped[circuit_type]
            runtimes = [r.runtime_ms for r in results if r.runtime_ms > 0]
            gates = [r.num_gates for r in results]
            clauses = [r.num_clauses for r in results]
            
            report += f"\n{circuit_type.upper()}\n"
            report += "-" * 40 + "\n"
            report += f"  Experiments: {len(results)}\n"
            report += f"  Avg Runtime: {statistics.mean(runtimes):.2f} ms\n"
            report += f"  Min Runtime: {min(runtimes):.2f} ms\n"
            report += f"  Max Runtime: {max(runtimes):.2f} ms\n"
            report += f"  Median Runtime: {statistics.median(runtimes):.2f} ms\n"
            if len(runtimes) > 1:
                report += f"  Std Dev: {statistics.stdev(runtimes):.2f} ms\n"
            report += f"  Avg Gates: {statistics.mean(gates):.1f}\n"
            report += f"  Avg Clauses: {statistics.mean(clauses):.1f}\n"
        
        report += "\n" + "=" * 70 + "\n"
        
        output_path = self.output_dir / output_file
        with open(output_path, 'w') as f:
            f.write(report)
        
        print(f"\n✓ Report saved to {output_path}")
        return report


def main():
    """Hauptfunktion zum Testen"""
    import sys
    
    if len(sys.argv) > 1:
        json_path = sys.argv[1]
    else:
        json_path = "experiment_results/experiment_results.json"
    
    if not Path(json_path).exists():
        print(f"Error: {json_path} not found")
        return
    
    print(f"Loading results from {json_path}...")
    results = ExperimentVisualizer.load_results(json_path)
    print(f"Loaded {len(results)} results")
    
    output_dir = Path(json_path).parent / "plots"
    visualizer = ExperimentVisualizer(results, output_dir=str(output_dir))
    
    print("\nGenerating plots...")
    visualizer.export_all_plots(formats=['svg', 'pdf'])
    
    print("\nGenerating report...")
    visualizer.create_summary_report()
    
    print("\n✓ All visualizations complete!")


if __name__ == "__main__":
    main()