import os
import sys

from flask import Flask, request, jsonify, render_template, send_file

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(CURRENT_DIR)

def resolve_file(filename):
    p1 = os.path.join(CURRENT_DIR, filename)
    if os.path.exists(p1):
        return p1
    p2 = os.path.join(PARENT_DIR, filename)
    if os.path.exists(p2):
        return p2
    return p1

templates_dir = os.path.join(CURRENT_DIR, "templates")
if not os.path.exists(templates_dir):
    templates_dir = os.path.join(PARENT_DIR, "templates")

app = Flask(__name__, template_folder=templates_dir)

# Shared pure-NumPy inference (synced into api/ by scripts/sync_vercel_bundle.py)
try:
    from flarewatch.inference import Classifier
    from flarewatch.live import LiveFeed
except ImportError:
    sys.path.insert(0, PARENT_DIR)
    from flarewatch.inference import Classifier
    from flarewatch.live import LiveFeed

classifier = Classifier([CURRENT_DIR, PARENT_DIR])
live_feed = LiveFeed(classifier)

def process_classification(tweet):
    res = classifier.classify(tweet)
    res['confidence'] = res['confidence_percent']
    res['prob'] = round(res['probability'], 3)
    return res

try:
    import data_service
except ImportError:
    sys.path.insert(0, PARENT_DIR)
    import data_service

def _get_context_data():
    return {
        'initial_stats': data_service.get_stats(),
        'initial_tweets': data_service.query_tweets(page=1, page_size=25),
        'initial_model_info': data_service.get_model_performance(),
        'initial_dataset': data_service.query_dataset(page=1, page_size=25)
    }

# Explicit JSON API Routes
@app.route('/api/predict', methods=['POST'])
@app.route('/api/index/api/predict', methods=['POST'])
def api_predict():
    data = request.get_json(force=True, silent=True) or {}
    tweet = data.get('tweet', '')
    if not tweet.strip():
        return jsonify({'error': 'No tweet text provided'}), 400
    res = classifier.classify(tweet)
    return jsonify({
        'tweet': res['tweet'],
        'cleaned': res['cleaned'],
        'tokens': res['tokens'],
        'emergency_signals': res['emergency_signals'],
        'is_request': res['is_request'],
        'probability': res['probability'],
        'confidence_percent': res['confidence_percent'],
        'resource': res['resource'],
        'resources': res['resources'],
        'locations': res['locations'],
        'explanation': res['explanation'],
        'status': res['status']
    })

@app.route('/api/live', methods=['GET'])
@app.route('/api/index/api/live', methods=['GET'])
def api_live():
    return jsonify(live_feed.get())

@app.route('/api/stats', methods=['GET'])
@app.route('/api/index/api/stats', methods=['GET'])
def api_stats():
    return jsonify(data_service.get_stats())

@app.route('/api/tweets', methods=['GET'])
@app.route('/api/index/api/tweets', methods=['GET'])
def api_tweets():
    page = int(request.args.get('page', 1))
    page_size = min(int(request.args.get('page_size', 25)), 100)
    search = request.args.get('search', '')
    filter_type = request.args.get('filter_type', 'all')
    resource = request.args.get('resource', 'all')
    sort_by = request.args.get('sort_by', 'default')
    result = data_service.query_tweets(
        page=page,
        page_size=page_size,
        search=search,
        filter_type=filter_type,
        resource=resource,
        sort_by=sort_by
    )
    return jsonify(result)

@app.route('/api/dataset', methods=['GET'])
@app.route('/api/index/api/dataset', methods=['GET'])
def api_dataset():
    page = int(request.args.get('page', 1))
    page_size = min(int(request.args.get('page_size', 25)), 100)
    search = request.args.get('search', '')
    label = request.args.get('label', 'all')
    result = data_service.query_dataset(
        page=page,
        page_size=page_size,
        search=search,
        label=label
    )
    return jsonify(result)

@app.route('/api/model-info', methods=['GET'])
@app.route('/api/index/api/model-info', methods=['GET'])
def api_model_info():
    return jsonify(data_service.get_model_performance())

@app.route('/api/health', methods=['GET'])
@app.route('/api/index/api/health', methods=['GET'])
def api_health():
    return jsonify({'status': 'healthy', 'service': 'disaster-tweet-classifier', 'runtime': 'vercel-serverless'})

@app.route('/export/<file_type>', methods=['GET'])
@app.route('/api/index/export/<file_type>', methods=['GET'])
def export_file(file_type):
    if file_type in ('results.csv', 'classification_results.csv'):
        filepath = resolve_file('classification_results.csv')
        if os.path.exists(filepath):
            return send_file(filepath, as_attachment=True, download_name='classification_results.csv', mimetype='text/csv')
    elif file_type in ('dataset.csv', 'dataset_cleaned.csv'):
        filepath = resolve_file('dataset_cleaned.csv')
        if os.path.exists(filepath):
            return send_file(filepath, as_attachment=True, download_name='dataset_cleaned.csv', mimetype='text/csv')
    elif file_type == 'stats.json':
        stats = data_service.get_stats()
        return jsonify(stats)
    return jsonify({'error': 'File not found'}), 404

def _dispatch_subroute():
    """Dispatches requests when Vercel rewrites all paths to /api/index"""
    candidates = [
        request.args.get('__path__'),
        request.headers.get('x-matched-path'),
        request.headers.get('x-forwarded-uri'),
        request.environ.get('HTTP_X_MATCHED_PATH'),
        request.environ.get('PATH_INFO'),
        request.path
    ]
    target_path = ''
    for c in candidates:
        if c:
            c_clean = c.split('?')[0].rstrip('/')
            if any(endpoint in c_clean for endpoint in ['/api/', '/export/', '/predict']):
                target_path = c_clean
                break

    if not target_path:
        return None

    if target_path.endswith('/api/predict') or (request.is_json and '/api/' in target_path):
        return api_predict()
    elif target_path.endswith('/api/live') or target_path.endswith('/live'):
        return api_live()
    elif target_path.endswith('/api/stats') or target_path.endswith('/stats'):
        return api_stats()
    elif target_path.endswith('/api/tweets') or target_path.endswith('/tweets'):
        return api_tweets()
    elif target_path.endswith('/api/dataset') or target_path.endswith('/dataset'):
        return api_dataset()
    elif target_path.endswith('/api/model-info') or target_path.endswith('/model-info'):
        return api_model_info()
    elif target_path.endswith('/api/health') or target_path.endswith('/health'):
        return api_health()
    elif '/export/' in target_path:
        file_type = target_path.split('/export/')[-1]
        return export_file(file_type)

    return None

# Unified handler for all web interface routes
@app.route('/', methods=['GET', 'POST'])
@app.route('/predict', methods=['GET', 'POST'])
@app.route('/api', methods=['GET', 'POST'])
@app.route('/api/index', methods=['GET', 'POST'])
@app.route('/api/index.py', methods=['GET', 'POST'])
def index():
    # 1. First check if this was a rewritten sub-route call
    dispatched = _dispatch_subroute()
    if dispatched is not None:
        return dispatched

    # 2. Handle Web Form POST request
    if request.method == 'POST':
        ctx = _get_context_data()
        tweet = request.form.get('tweet', '')
        if not tweet.strip():
            return render_template('index.html',
                                   prediction="Please enter or select a tweet to classify.",
                                   tweet='',
                                   cleaned=None,
                                   is_request=None,
                                   resource=None,
                                   confidence=None,
                                   prob=None,
                                   **ctx)
        res = process_classification(tweet)
        return render_template('index.html',
                               prediction=res['prediction'],
                               tweet=res['tweet'],
                               cleaned=res['cleaned'],
                               is_request=res['is_request'],
                               resource=res['resource'],
                               confidence=res['confidence'],
                               prob=res['prob'],
                               **ctx)

    # 3. Default GET request -> Render main dashboard UI
    ctx = _get_context_data()
    return render_template('index.html',
                           prediction=None,
                           tweet='',
                           cleaned=None,
                           is_request=None,
                           resource=None,
                           confidence=None,
                           prob=None,
                           **ctx)

class VercelMiddleware:
    def __init__(self, wsgi_app):
        self.wsgi_app = wsgi_app

    def __call__(self, environ, start_response):
        matched = environ.get('HTTP_X_MATCHED_PATH') or environ.get('HTTP_X_FORWARDED_URI')
        if matched:
            path = matched.split('?')[0]
            if path and not path.endswith('/api/index') and not path.endswith('/api/index.py'):
                environ['PATH_INFO'] = path
        return self.wsgi_app(environ, start_response)

app.wsgi_app = VercelMiddleware(app.wsgi_app)
