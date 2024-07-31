from flask import Flask, render_template, request, jsonify, Response
import pubmed_scraper
import json

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/how-it-works')
def how_it_works():
    return render_template('how_it_works.html')

@app.route('/scrape', methods=['GET'])
def scrape():
    max_results = int(request.args.get('max_results', 100))
    protein_target = request.args.get('protein_target', '')
    max_publications = request.args.get('max_publications')
    max_publications = int(max_publications) if max_publications else None
    
    query = pubmed_scraper.generate_query(protein_target)
    
    def generate():
        try:
            for article, progress in pubmed_scraper.scrape_pubmed(query, protein_target, max_results=max_results, max_publications=max_publications):
                yield f"data: {json.dumps({'progress': progress, 'article': article})}\n\n"
            
            yield f"data: {json.dumps({'progress': 100, 'message': 'Search completed.'})}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'progress': 100, 'error': f'An error occurred: {str(e)}'})}\n\n"
    
    return Response(generate(), mimetype='text/event-stream')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)