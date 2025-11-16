const googleNews = require('google-news-json');
const readline = require('readline');

const rl = readline.createInterface({
  input: process.stdin,
  output: process.stdout,
});

const askQuestion = (query) =>
  new Promise((resolve) => {
    rl.question(query, (answer) => resolve(answer));
  });

const fetchGoogleNews = async () => {
  console.log('Google News fetcher ready. Press Ctrl+C to exit at any time.');

  const keyword = (await askQuestion('Enter a company name or keyword (e.g., stocks, AAPL): ')).trim();
  if (!keyword) {
    console.log('Keyword is required to search Google News. Exiting.');
    rl.close();
    return;
  }

  const regionInput = (await askQuestion('Enter locale (default en-US, format lang-country): ')).trim() || 'en-US';
  const [language, country] = regionInput.includes('-')
    ? regionInput.split('-')
    : [regionInput, 'US'];

  const maxArticles = parseInt((await askQuestion('How many articles would you like? (default 10): ')).trim(), 10) || 10;

  console.log('\nFetching Google News articles...', { keyword, locale: `${language}-${country}`, maxArticles });

  googleNews.getNews(googleNews.SEARCH, keyword, `${language}-${country}`, (error, response) => {
    if (error) {
      console.error('Error fetching Google News:', error.message || error);
      rl.close();
      return;
    }

    const articles = Array.isArray(response) ? response.slice(0, maxArticles) : [];

    if (articles.length === 0) {
      console.log('No articles returned by Google News for the given criteria.');
      rl.close();
      return;
    }

    articles.forEach((article, index) => {
      console.log(`\n[${index + 1}] ${article.title}`);
      console.log(`Source: ${article.publisher || article.publishers || 'Unknown'}`);
      console.log(`Published: ${article.time}`);
      if (article.snippet) {
        console.log(`Snippet: ${article.snippet}`);
      }
      console.log(`URL: ${article.link}`);
      console.log('---');
    });

    rl.close();
  });
};

fetchGoogleNews();