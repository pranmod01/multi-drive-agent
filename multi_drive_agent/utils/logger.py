import logging
import os
from datetime import datetime
from typing import Optional, Dict, Any
import json


class ExperimentLogger:
    """
    Logger for multi-drive agent experiments.

    Provides structured logging for:
    - Console output with different verbosity levels
    - File-based experiment logs
    - JSON-formatted metrics for analysis
    - TensorBoard integration (optional)
    """

    def __init__(
        self,
        experiment_name: str,
        log_dir: str = 'logs',
        console_level: int = logging.INFO,
        file_level: int = logging.DEBUG,
        use_tensorboard: bool = False
    ):
        """
        Initialize experiment logger.

        Args:
            experiment_name: Name of the experiment
            log_dir: Directory to store log files
            console_level: Logging level for console output
            file_level: Logging level for file output
            use_tensorboard: Whether to enable TensorBoard logging
        """
        self.experiment_name = experiment_name
        self.log_dir = log_dir
        self.use_tensorboard = use_tensorboard

        # Create log directory
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.experiment_dir = os.path.join(log_dir, f'{experiment_name}_{timestamp}')
        os.makedirs(self.experiment_dir, exist_ok=True)

        # Set up Python logger
        self.logger = logging.getLogger(experiment_name)
        self.logger.setLevel(logging.DEBUG)
        self.logger.handlers.clear()

        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(console_level)
        console_formatter = logging.Formatter(
            '%(asctime)s [%(levelname)s] %(message)s',
            datefmt='%H:%M:%S'
        )
        console_handler.setFormatter(console_formatter)
        self.logger.addHandler(console_handler)

        # File handler
        log_file = os.path.join(self.experiment_dir, 'experiment.log')
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(file_level)
        file_formatter = logging.Formatter(
            '%(asctime)s [%(levelname)s] %(name)s: %(message)s'
        )
        file_handler.setFormatter(file_formatter)
        self.logger.addHandler(file_handler)

        # Metrics file
        self.metrics_file = os.path.join(self.experiment_dir, 'metrics.jsonl')

        # TensorBoard writer (lazy initialization)
        self.tb_writer = None
        if use_tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter
                tb_dir = os.path.join(self.experiment_dir, 'tensorboard')
                self.tb_writer = SummaryWriter(tb_dir)
                self.logger.info(f'TensorBoard logging enabled: {tb_dir}')
            except ImportError:
                self.logger.warning('TensorBoard requested but torch not available')
                self.use_tensorboard = False

        self.logger.info(f'Experiment logger initialized: {self.experiment_dir}')

    def info(self, message: str):
        """Log info message."""
        self.logger.info(message)

    def debug(self, message: str):
        """Log debug message."""
        self.logger.debug(message)

    def warning(self, message: str):
        """Log warning message."""
        self.logger.warning(message)

    def error(self, message: str):
        """Log error message."""
        self.logger.error(message)

    def log_metrics(self, metrics: Dict[str, Any], step: int):
        """
        Log metrics at a given step.

        Args:
            metrics: Dictionary of metric name to value
            step: Current step/episode number
        """
        # Write to metrics file
        metrics_entry = {
            'step': step,
            'timestamp': datetime.now().isoformat(),
            **metrics
        }

        with open(self.metrics_file, 'a') as f:
            f.write(json.dumps(metrics_entry) + '\n')

        # Write to TensorBoard
        if self.tb_writer is not None:
            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    self.tb_writer.add_scalar(key, value, step)

    def log_hyperparameters(self, config: Dict[str, Any]):
        """
        Log experiment hyperparameters.

        Args:
            config: Configuration dictionary
        """
        config_file = os.path.join(self.experiment_dir, 'config.json')
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=2)

        self.logger.info('Hyperparameters saved')

        # Log to TensorBoard
        if self.tb_writer is not None:
            # TensorBoard expects flat dictionary with string values
            flat_config = self._flatten_dict(config)
            self.tb_writer.add_hparams(flat_config, {})

    def _flatten_dict(self, d: Dict[str, Any], parent_key: str = '') -> Dict[str, str]:
        """Flatten nested dictionary for TensorBoard."""
        items = []
        for k, v in d.items():
            new_key = f'{parent_key}/{k}' if parent_key else k
            if isinstance(v, dict):
                items.extend(self._flatten_dict(v, new_key).items())
            else:
                items.append((new_key, str(v)))
        return dict(items)

    def close(self):
        """Close logger and cleanup."""
        if self.tb_writer is not None:
            self.tb_writer.close()

        self.logger.info('Experiment logger closed')

    def get_log_dir(self) -> str:
        """Get the experiment log directory."""
        return self.experiment_dir
