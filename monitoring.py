import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from evidently import ColumnMapping
from evidently.report import Report
from evidently.metric_suite import MetricSuite
from evidently.metrics import DataDriftMetric, DataQualityMetric
import boto3
import json
from prometheus_client import Counter, Histogram, Gauge, start_http_server
import time

# Métricas Prometheus
prediction_counter = Counter('predictions_total', 'Total number of predictions made')
prediction_latency = Histogram('prediction_duration_seconds', 'Time spent generating predictions')
model_accuracy = Gauge('model_accuracy', 'Model accuracy on recent data')

def log_prediction_metrics(duration, accuracy=None):
	prediction_counter.inc()
	prediction_latency.observe(duration)
	if accuracy is not None:
		model_accuracy.set(accuracy)

def run_evidently_report(ref_data: pd.DataFrame, curr_data: pd.DataFrame, column_mapping: ColumnMapping):
	report = Report(metrics=[DataDriftMetric(), DataQualityMetric()])
	report.run(reference_data=ref_data, current_data=curr_data, column_mapping=column_mapping)
	return report.as_dict()

if __name__ == "__main__":
	# Exemplo: iniciar servidor Prometheus
	start_http_server(8001)
	print("Prometheus metrics server running on port 8001...")
	# Simulação de logging de métricas
	for i in range(10):
		start = time.time()
		time.sleep(np.random.uniform(0.1, 0.5))
		duration = time.time() - start
		acc = np.random.uniform(0.7, 0.95)
		log_prediction_metrics(duration, acc)
		print(f"Logged prediction {i+1} | duration: {duration:.2f}s | acc: {acc:.2f}")