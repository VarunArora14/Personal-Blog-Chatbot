### Scraping Data

The first step we need for creating RAG application is to scrape the data from the blog.

For this we follow the following steps -

- Fetch all the urls of the blog using beautiful soup using file url_scraper.py
- Filter urls so they may not contain duplicate and not contain outsider links as well (happens when blogs themselves have reference urls)
- Store these urls in pkl file which will be used later to fetch data for RAG
