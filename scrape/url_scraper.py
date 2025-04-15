import pickle
import re
import time

import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv

load_dotenv()


def getBlogLinks(url: str):
    source_code = requests.get(url)
    soup = BeautifulSoup(source_code.content, "lxml")
    data = []
    links = []

    def remove_duplicates(l):  # remove duplicates and unURL string
        for item in l:
            match = re.search("(?P<url>https?://[^\s]+)", item)
            if match is not None:
                links.append((match.group("url")))

    for link in soup.find_all("a", href=True):
        data.append(str(link.get("href")))

    # print(data)
    flag = True
    remove_duplicates(data)
    while flag:
        try:
            for link in links:
                for j in soup.find_all("a", href=True):
                    temp = []
                    source_code = requests.get(link)
                    soup = BeautifulSoup(source_code.content, "lxml")
                    temp.append(str(j.get("href")))
                    remove_duplicates(temp)

                    # breaking loop in case links count very high for blogs leading to google sites
                    if len(links) > 162:  # set limitation to number of URLs
                        break
                if len(links) > 162:
                    break
            if len(links) > 162:
                break
        except Exception as e:
            print(e)
            if len(links) > 162:
                break

    return links
    # extract only posts


def exportBlogUrls():
    unique_links = set()

    urls = ["https://varunarora14.github.io/"]
    i = 2
    isValidPage = True
    while isValidPage:
        page_url = f"https://varunarora14.github.io/page/{i}"
        print(page_url)
        response = requests.get(page_url)

        if response.status_code == 404:
            isValidPage = False

        else:
            i += 1
            urls.append(page_url)
            print(page_url)

    for url in urls:
        blog_urls = getBlogLinks(url=url)
        unique_links.update(blog_urls)

    blogLinks = [
        link for link in unique_links if "varunarora14.github.io/posts/" in link
    ]

    print(blogLinks)

    with open("blog_urls.pkl", "wb") as f:
        pickle.dump(blogLinks, f)


exportBlogUrls()
