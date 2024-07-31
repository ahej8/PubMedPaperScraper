document.addEventListener('DOMContentLoaded', function() {
    const form = document.getElementById('scraper-form');
    const progressBar = document.querySelector('.progress-bar');
    const progressContainer = document.querySelector('.progress');
    const resultsContainer = document.getElementById('results');
    const toggleAdvancedButton = document.getElementById('toggle-advanced');
    const advancedSearchSection = document.getElementById('advanced-search');
    const exportExcelBtn = document.getElementById('export-excel');
    let scrapedData = [];

    toggleAdvancedButton.addEventListener('click', function() {
        if (advancedSearchSection.style.display === 'none') {
            advancedSearchSection.style.display = 'block';
        } else {
            advancedSearchSection.style.display = 'none';
        }
    });

    form.addEventListener('submit', function(e) {
        e.preventDefault();
        
        const maxResults = document.getElementById('max-results').value;
        const proteinTarget = document.getElementById('protein-target').value;
        const maxPublications = document.getElementById('max-publications').value;
        
        progressContainer.style.display = 'block';
        progressBar.style.width = '0%';
        progressBar.textContent = '0%';
        resultsContainer.innerHTML = '';
        scrapedData = [];
        exportExcelBtn.style.display = 'none';
        
        const eventSource = new EventSource(`/scrape?max_results=${encodeURIComponent(maxResults)}&protein_target=${encodeURIComponent(proteinTarget)}&max_publications=${encodeURIComponent(maxPublications)}`);
        
        eventSource.onmessage = function(event) {
            const data = JSON.parse(event.data);
            progressBar.style.width = `${data.progress}%`;
            progressBar.textContent = `${Math.round(data.progress)}%`;
            
            if (data.article) {
                scrapedData.push(data.article);
                displayResults(scrapedData);
            } else if (data.message) {
                resultsContainer.innerHTML += `<p>${data.message}</p>`;
            }
            
            if (data.progress >= 100) {
                eventSource.close();
                exportExcelBtn.style.display = 'block';
            }
        };
        
        eventSource.onerror = function(error) {
            console.error('EventSource failed:', error);
            eventSource.close();
            resultsContainer.innerHTML = '<p>An error occurred while scraping PubMed.</p>';
        };
    });

    function displayResults(articles) {
        let html = '<h2>Results</h2>';
        if (articles.length > 0) {
            html += '<table class="results-table"><thead><tr><th>Title</th><th>Authors</th><th>Email</th><th>Summary</th><th>Lead Author Publications</th><th>Source Link</th></tr></thead><tbody>';
            
            articles.forEach(article => {
                html += `<tr>
                    <td class="title-col">${article.title}</td>
                    <td class="authors-col">${article.authors}</td>
                    <td class="email-col">${article.email}</td>
                    <td class="summary-col">${article.summary}</td>
                    <td class="publications-col">${article.first_author_publications}</td>
                    <td class="link-col"><a href="${article.source_link}" target="_blank">Link</a></td>
                </tr>`;
            });
            
            html += '</tbody></table>';
        } else {
            html += '<p>No results found.</p>';
        }
        resultsContainer.innerHTML = html;
    }

    exportExcelBtn.addEventListener('click', function() {
        const ws = XLSX.utils.json_to_sheet(scrapedData);
        const wb = XLSX.utils.book_new();
        XLSX.utils.book_append_sheet(wb, ws, "PubMed Results");
        XLSX.writeFile(wb, "pubmed_results.xlsx");
    });
});