import csv
import glob
import os
from typing import Dict, List, Tuple


def load_errors_from_csv(path: str) -> List[float]:
    errors = []
    with open(path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get('confidence') == 'failed':
                continue
            val = row.get('distance_error_m')
            if val is None:
                continue
            try:
                errors.append(float(val))
            except ValueError:
                continue
    return errors


def compute_accuracy_distribution(errors: List[float], thresholds: List[int]) -> List[Tuple[int, float]]:
    total = len(errors)
    if total == 0:
        return [(k, 0.0) for k in thresholds]
    results = []
    for k in thresholds:
        count = sum(1 for e in errors if e <= k)
        results.append((k, count / total))
    return results


def write_csv_report(path: str, rows: List[Dict[str, str]]):
    with open(path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_text_report(path: str, rows: List[Dict[str, str]]):
    with open(path, 'w', encoding='utf-8') as f:
        f.write('Accuracy @k meters report\n')
        f.write('=' * 40 + '\n')
        for row in rows:
            f.write(f"{row['file']}: total={row['total']}\n")
            for k in ['10', '50', '100', '500', '1000']:
                f.write(f"  <= {k}m: {row[f'@{k}m_pct']}% ({row[f'@{k}m_count']})\n")
            f.write('\n')


def main():
    result_patterns = [
        os.path.join('results', 'evaluation*.csv'),
        os.path.join('results', 'baseline', 'evaluation*.csv'),
    ]
    files = []
    for pattern in result_patterns:
        files.extend(glob.glob(pattern))
    files = sorted(set(files))

    if not files:
        print('No evaluation CSV files found in results/ or results/baseline/.')
        return

    report_dir = os.path.join('results', 'reports')
    os.makedirs(report_dir, exist_ok=True)

    thresholds = [10, 50, 100, 500, 1000]
    rows = []

    for path in files:
        errors = load_errors_from_csv(path)
        dist = compute_accuracy_distribution(errors, thresholds)
        total = len(errors)

        row = {
            'file': os.path.basename(path),
            'total': str(total),
        }

        for k, ratio in dist:
            count = sum(1 for e in errors if e <= k)
            row[f'@{k}m_pct'] = f"{ratio * 100:.2f}"
            row[f'@{k}m_count'] = str(count)

        rows.append(row)

    if rows:
        csv_path = os.path.join(report_dir, 'evaluation_accuracy_report.csv')
        txt_path = os.path.join(report_dir, 'evaluation_accuracy_report.txt')
        write_csv_report(csv_path, rows)
        write_text_report(txt_path, rows)

        print(f'Report written to: {csv_path}')
        print(f'Report written to: {txt_path}')


if __name__ == '__main__':
    main()
