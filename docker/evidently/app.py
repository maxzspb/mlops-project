import os
import json
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
from threading import Thread, Lock
import schedule
import time

from flask import Flask, jsonify
from prometheus_client import CollectorRegistry, Gauge, generate_latest

try:
    from evidently.report import Report
    from evidently.metrics import DataDriftTable, ColumnDriftMetric
except ImportError:
    Report = None
    print("⚠️ Evidently not installed, using simulated metrics only")

# ===== SETUP =====
app = Flask(__name__)
REPORTS_DIR = Path("/var/evidently/reports")
REPORTS_DIR.mkdir(exist_ok=True)

# Prometheus metrics
registry = CollectorRegistry()
drift_score = Gauge(
    'ml_data_drift_score',
    'Evidently drift detection score [0-1]',
    ['feature', 'model'],
    registry=registry
)
drift_check_timestamp = Gauge(
    'ml_drift_check_timestamp',
    'Last drift check timestamp',
    ['model'],
    registry=registry
)
drift_check_duration = Gauge(
    'ml_drift_check_duration_seconds',
    'Drift check duration',
    ['model'],
    registry=registry
)

# State
metrics_lock = Lock()
current_metrics = {}
last_check_time = None

# ===== DATA GENERATION =====
def generate_synthetic_data(n_samples=100):
    """Generate synthetic iris-like data with drift"""
    np.random.seed(int(time.time()) % 10000)
    
    baseline = pd.DataFrame({
        'sepal_length': np.random.normal(5.8, 0.8, n_samples),
        'sepal_width': np.random.normal(3.0, 0.4, n_samples),
        'petal_length': np.random.normal(3.7, 1.7, n_samples),
        'petal_width': np.random.normal(1.2, 0.8, n_samples),
    })
    
    current = pd.DataFrame({
        'sepal_length': np.random.normal(6.2 + np.random.uniform(-0.5, 0.5), 0.9, n_samples),
        'sepal_width': np.random.normal(3.0 + np.random.uniform(-0.3, 0.3), 0.4, n_samples),
        'petal_length': np.random.normal(4.0 + np.random.uniform(-0.5, 0.5), 1.8, n_samples),
        'petal_width': np.random.normal(1.3 + np.random.uniform(-0.3, 0.3), 0.8, n_samples),
    })
    
    return baseline, current

def calculate_drift_scores(baseline, current):
    """Calculate drift using Wasserstein distance"""
    from scipy.stats import wasserstein_distance
    
    drift_metrics = {}
    for col in baseline.columns:
        distance = wasserstein_distance(baseline[col], current[col])
        drift_score_val = min(distance / 2.0, 1.0)
        drift_metrics[col] = {
            "drift_score": float(drift_score_val),
            "drifted": drift_score_val > 0.3,
            "distance": float(distance)
        }
    return drift_metrics

def generate_evidently_report(baseline, current):
    """Generate Evidently HTML report"""
    if Report is None:
        return None
    
    try:
        report = Report(metrics=[
            DataDriftTable(),
            ColumnDriftMetric(column_name='sepal_length'),
            ColumnDriftMetric(column_name='sepal_width'),
            ColumnDriftMetric(column_name='petal_length'),
            ColumnDriftMetric(column_name='petal_width'),
        ])
        
        report.run(reference_data=baseline, current_data=current)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = REPORTS_DIR / f"drift_report_{timestamp}.html"
        report.save_html(str(report_path))
        
        print(f"✅ Evidently report saved: {report_path}")
        return str(report_path)
    except Exception as e:
        print(f"⚠️ Could not generate Evidently report: {e}")
        return None

def push_metrics_to_prometheus(drift_metrics):
    """Push metrics to Prometheus"""
    try:
        for feature, metrics in drift_metrics.items():
            drift_score.labels(
                feature=feature,
                model='iris-classifier'
            ).set(metrics['drift_score'])
        
        drift_check_timestamp.labels(model='iris-classifier').set(
            datetime.now().timestamp()
        )
        
        print("✅ Metrics updated")
    except Exception as e:
        print(f"⚠️ Error updating metrics: {e}")

def save_metrics_locally(drift_metrics):
    """Save metrics as JSON"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    metrics_path = REPORTS_DIR / f"drift_metrics_{timestamp}.json"
    
    with open(metrics_path, 'w') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'metrics': drift_metrics,
        }, f, indent=2)
    
    print(f"✅ Metrics saved to {metrics_path}")

def run_drift_check():
    """Main drift check function"""
    global current_metrics, last_check_time
    
    print(f"\n[{datetime.now()}] Starting drift check...")
    start_time = time.time()
    
    try:
        baseline, current = generate_synthetic_data(n_samples=100)
        drift_metrics = calculate_drift_scores(baseline, current)
        
        with metrics_lock:
            current_metrics = drift_metrics
            last_check_time = datetime.now()
        
        print("\n📊 Drift Metrics:")
        for feature, metrics in drift_metrics.items():
            status = "🔴 DRIFTED" if metrics['drifted'] else "🟢 OK"
            print(f"  {feature}: {metrics['drift_score']:.3f} {status}")
        
        generate_evidently_report(baseline, current)
        save_metrics_locally(drift_metrics)
        push_metrics_to_prometheus(drift_metrics)
        
        duration = time.time() - start_time
        drift_check_duration.labels(model='iris-classifier').set(duration)
        
        print(f"\n✅ Drift check completed in {duration:.2f}s\n")
        
    except Exception as e:
        print(f"\n❌ Error during drift check: {e}\n")

def scheduler_thread():
    """Background thread for scheduled tasks"""
    run_drift_check()
    schedule.every(15).minutes.do(run_drift_check)
    
    print("🔄 Scheduler started (runs every 15 minutes)")
    
    while True:
        schedule.run_pending()
        time.sleep(60)

# ===== FLASK ROUTES =====
@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat()
    }), 200

@app.route('/metrics', methods=['GET'])
def metrics():
    """Prometheus metrics endpoint"""
    return generate_latest(registry), 200, {'Content-Type': 'text/plain; charset=utf-8'}

@app.route('/api/drift', methods=['GET'])
def api_drift():
    """API: Get latest drift metrics"""
    with metrics_lock:
        return jsonify({
            'timestamp': last_check_time.isoformat() if last_check_time else None,
            'metrics': current_metrics
        }), 200

@app.route('/api/reports', methods=['GET'])
def api_reports():
    """API: List all reports"""
    reports = sorted(
        [f.name for f in REPORTS_DIR.glob('drift_report_*.html')],
        reverse=True
    )
    return jsonify({
        'reports': reports,
        'total': len(reports)
    }), 200

@app.route('/reports/<filename>', methods=['GET'])
def serve_report(filename):
    """Serve HTML reports"""
    if not filename.endswith('.html'):
        return jsonify({'error': 'Invalid file'}), 400
    
    report_path = REPORTS_DIR / filename
    if not report_path.exists():
        return jsonify({'error': 'Report not found'}), 404
    
    with open(report_path, 'r') as f:
        return f.read(), 200, {'Content-Type': 'text/html'}

@app.route('/', methods=['GET'])
def dashboard():
    """Main dashboard with HTML"""
    with metrics_lock:
        metrics_data = current_metrics
        check_time = last_check_time.isoformat() if last_check_time else "Never"
    
    reports = sorted(
        [f.name for f in REPORTS_DIR.glob('drift_report_*.html')],
        reverse=True
    )[:10]
    
    html = '''
    <!DOCTYPE html>
    <html>
    <head>
        <title>Evidently Drift Detection Dashboard</title>
        <meta charset="utf-8">
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <style>
            * { margin: 0; padding: 0; box-sizing: border-box; }
            body {
                font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
                background: #f5f5f5;
                padding: 20px;
            }
            .container { max-width: 1200px; margin: 0 auto; }
            .header {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                padding: 30px;
                border-radius: 8px;
                margin-bottom: 30px;
            }
            .header h1 { font-size: 32px; margin-bottom: 10px; }
            .header p { opacity: 0.9; }
            
            .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin-bottom: 30px; }
            @media (max-width: 768px) { .grid { grid-template-columns: 1fr; } }
            
            .card {
                background: white;
                border-radius: 8px;
                padding: 20px;
                box-shadow: 0 2px 8px rgba(0,0,0,0.1);
            }
            .card h2 { margin-bottom: 20px; color: #333; font-size: 18px; }
            
            .metric-item {
                display: flex;
                justify-content: space-between;
                align-items: center;
                padding: 12px 0;
                border-bottom: 1px solid #eee;
            }
            .metric-item:last-child { border-bottom: none; }
            .metric-name { font-weight: 500; }
            .metric-score {
                font-size: 16px;
                font-weight: bold;
                display: flex;
                align-items: center;
                gap: 8px;
            }
            .metric-ok { color: #10b981; }
            .metric-drift { color: #ef4444; }
            
            .status-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; }
            .status-item {
                background: #f9fafb;
                padding: 12px;
                border-radius: 6px;
                border-left: 3px solid #667eea;
            }
            .status-label { font-size: 12px; color: #666; }
            .status-value { font-size: 18px; font-weight: bold; margin-top: 4px; }
            
            .reports-list {
                list-style: none;
                max-height: 300px;
                overflow-y: auto;
            }
            .reports-list li {
                padding: 10px;
                border-bottom: 1px solid #eee;
                cursor: pointer;
                transition: background 0.2s;
            }
            .reports-list li:hover { background: #f9fafb; }
            .reports-list a {
                color: #667eea;
                text-decoration: none;
                font-size: 14px;
            }
            
            .footer { text-align: center; color: #666; margin-top: 40px; font-size: 12px; }
            .refresh-time { color: #999; font-size: 12px; margin-top: 10px; }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>🔍 Evidently Drift Detection</h1>
                <p>Real-time ML data drift monitoring and reporting</p>
            </div>
            
            <div class="grid">
                <div class="card">
                    <h2>📊 Current Drift Metrics</h2>
    '''
    
    for feature, data in metrics_data.items():
        status_class = 'metric-drift' if data['drifted'] else 'metric-ok'
        status_emoji = '🔴' if data['drifted'] else '🟢'
        html += f'''
                    <div class="metric-item">
                        <span class="metric-name">{feature}</span>
                        <span class="metric-score {status_class}">{status_emoji} {data['drift_score']:.3f}</span>
                    </div>
        '''
    
    html += f'''
                    <div class="refresh-time">Last check: {check_time}</div>
                </div>
                
                <div class="card">
                    <h2>📈 Status Summary</h2>
                    <div class="status-grid">
                        <div class="status-item">
                            <div class="status-label">Features Checked</div>
                            <div class="status-value">{len(metrics_data)}</div>
                        </div>
                        <div class="status-item">
                            <div class="status-label">Drifted Features</div>
                            <div class="status-value">{sum(1 for m in metrics_data.values() if m['drifted'])}</div>
                        </div>
                        <div class="status-item">
                            <div class="status-label">Max Drift Score</div>
                            <div class="status-value">{max((m['drift_score'] for m in metrics_data.values()), default=0):.3f}</div>
                        </div>
                        <div class="status-item">
                            <div class="status-label">Avg Drift Score</div>
                            <div class="status-value">{sum(m['drift_score'] for m in metrics_data.values()) / len(metrics_data) if metrics_data else 0:.3f}</div>
                        </div>
                    </div>
                </div>
            </div>
            
            <div class="card">
                <h2>📄 Recent Reports</h2>
                <ul class="reports-list">
    '''
    
    for report in reports:
        html += f'<li><a href="/reports/{report}" target="_blank">📊 {report}</a></li>'
    
    html += '''
                </ul>
            </div>
            
            <div class="footer">
                <p>Evidently Drift Detection Service | Auto-refreshes every 15 minutes</p>
                <p>Metrics available at <code>/metrics</code> | API at <code>/api/drift</code></p>
            </div>
        </div>
        
        <script>
            setInterval(() => location.reload(), 30000);
        </script>
    </body>
    </html>
    '''
    
    return html, 200, {'Content-Type': 'text/html; charset=utf-8'}

# ===== MAIN =====
if __name__ == '__main__':
    scheduler = Thread(target=scheduler_thread, daemon=True)
    scheduler.start()
    
    print(f"\n🌐 Evidently Service starting on 0.0.0.0:5000")
    print(f"📊 Dashboard: http://localhost:5000")
    print(f"📈 Metrics: http://localhost:5000/metrics")
    print(f"🔍 API: http://localhost:5000/api/drift\n")
    
    app.run(host='0.0.0.0', port=5000, debug=False, use_reloader=False)