"""Exercise training, checkpoint loading, and summarization on temporary data."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


class SmokeTests(unittest.TestCase):
    def test_train_evaluate_and_skip(self):
        with tempfile.TemporaryDirectory(prefix='tomi_test_') as folder:
            folder = Path(folder)
            rng = np.random.default_rng(2021)
            frame = pd.DataFrame(rng.normal(size=(480, 3)), columns=['a', 'b', 'target'])
            frame.insert(0, 'date', pd.date_range('2000-01-01', periods=480, freq='h'))
            source = folder / 'series.csv'
            frame.to_csv(source, index=False)
            output = folder / 'outputs'
            for model in ('tomi', 'tomi-v'):
                command = [sys.executable, str(ROOT / 'run.py'), '--model', model,
                           '--data', 'custom', '--data_path', str(source),
                           '--seq_len', '24', '--pred_len', '12', '--window_size', '8',
                           '--embed_size', '4', '--d_model', '16', '--n_heads', '2',
                           '--d_ff', '32', '--batch_size', '16', '--eval_batch_size', '8',
                           '--train_epochs', '1', '--device', 'cpu', '--output_dir', str(output)]
                subprocess.run(command, cwd=ROOT, check=True, capture_output=True, text=True)
                run = next(output.glob(f'{model}_custom_*/metrics.json'))
                metrics = json.loads(run.read_text())
                self.assertTrue(np.isfinite(metrics['mse']) and metrics['mse'] >= 0)
                self.assertTrue(np.isfinite(metrics['mae']) and metrics['mae'] >= 0)
                resumed = subprocess.run(command + ['--skip_completed'], cwd=ROOT, check=True,
                                         capture_output=True, text=True)
                self.assertIn('Skip completed:', resumed.stdout)
                test_command = command + ['--mode', 'test', '--checkpoint',
                                          str(run.parent / 'checkpoint.pth')]
                subprocess.run(test_command, cwd=ROOT, check=True, capture_output=True, text=True)
            report = folder / 'metrics.csv'
            subprocess.run([sys.executable, str(ROOT / 'summarize_results.py'), '--output_dir',
                            str(output), '--csv', str(report)], check=True, capture_output=True, text=True)
            self.assertEqual(len(pd.read_csv(report)), 4)


if __name__ == '__main__':
    unittest.main()
