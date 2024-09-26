import requests
from bs4 import BeautifulSoup
import re
import time
from dotenv import load_dotenv
import pickle

load_dotenv()

def getBlogLinks(url = 'https://varunarora14.github.io/'):
    source_code = requests.get(url)
    soup = BeautifulSoup(source_code.content, 'lxml')
    data = []
    links = []

    def remove_duplicates(l): # remove duplicates and unURL string
        for item in l:
            match = re.search("(?P<url>https?://[^\s]+)", item)
            if match is not None:
                links.append((match.group("url")))

    for link in soup.find_all('a', href=True):
        data.append(str(link.get('href')))
    
    # print(data)    
    flag = True
    remove_duplicates(data)
    while flag:
        try:
            for link in links:
                for j in soup.find_all('a', href=True):
                    temp = []
                    source_code = requests.get(link)
                    soup = BeautifulSoup(source_code.content, 'lxml')
                    temp.append(str(j.get('href')))
                    remove_duplicates(temp)

                    # breaking loop in case links count very high for blogs leading to google sites
                    if len(links) > 162: # set limitation to number of URLs
                        break
                if len(links) > 162:
                    break
            if len(links) > 162:
                break
        except Exception as e:
            print(e)
            if len(links) > 162:
                break

    uniqueLinks = list(set(links))
    # extract only posts
    blogLinks = [link for link in uniqueLinks if "varunarora14.github.io/posts/" in link]
    with open('blog_urls.pkl', 'wb') as f:
        pickle.dump(blogLinks, f)

# getBlogLinks()

