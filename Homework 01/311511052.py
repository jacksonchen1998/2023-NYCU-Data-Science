import requests
from bs4 import BeautifulSoup
import tqdm
import os
import time
import tracemalloc

class BeautyCrawler:
    def __init__(self, index):
        self.index = index
        self.url = 'https://www.ptt.cc/bbs/Beauty/index'+ str(self.index) +'.html'
        self.headers = {'cookie': 'over18=1;'}
        self.soup = None
        self.articles = []
        self.imgs = []
        self.img_urls = []
        self.date = None
        self.vote = None

    def get_soup(self):
        r = requests.get(self.url, headers=self.headers)
        self.soup = BeautifulSoup(r.text, 'html.parser')
        return self.soup
    
    def get_articles(self):
        self.articles = self.soup.find_all('div', 'r-ent')
        return self.articles
    
    def get_article_url(self, article):
        # gap timer to avoid getting banned
        time.sleep(0.01)
        return 'https://www.ptt.cc' + article.find('a')['href']

    def get_article_title(self, article):
        return article.find('div', 'title').text.strip()
    
    def get_article_push(self, article):
        return article.find('div', 'nrec').text.strip()
    
    def get_article_date(self, article):
        # 01/01 to 0101 as format
        date = article.find('div', 'date').text.strip()
        if len(date) == 4:
            # add 0 if month is 1 digit
            date = '0' + date
            date = date.replace('/', '')
        if len(date) == 5:
            date = date.replace('/', '')
        return date
    
    def get_vote(self, article):
        return article.find('div', 'nrec').text.strip()
            
    def save_all_article_jsonl(self):
        # check if all_article.jsonl exists, if not, create it, otherwise, append
        if not os.path.exists('all_article.jsonl') and not os.path.isfile('all_popular.jsonl'):
            with open('all_article.jsonl', 'w') as f:
                f.write('')
            with open('all_popular.jsonl', 'w') as f:
                f.write('')
        for article in self.articles:
            article_data = {
                "date": self.get_article_date(article),
                "title": self.get_article_title(article),
                "url": self.get_article_url(article)
            }
            # remove articles with [公告] or Fw: [公告] in title
            if '公告' in article_data['title'] or 'Fw: [公告]' in article_data['title']:
                continue
            with open('all_article.jsonl', 'a') as f:
                f.write(str(article_data) + '\n')
            if self.get_vote(article) == '爆':
                with open('all_popular.jsonl', 'a') as f:
                    f.write(str(article_data) + '\n')

    def post_process_all_article(self):
        # delete_first_two_lines
        with open('all_article.jsonl', 'r') as f:
            lines = f.readlines()
        with open('all_article.jsonl', 'w') as f:
            f.writelines(lines[2:])
        # delete_last_four_lines
        with open('all_article.jsonl', 'r') as f:
            lines = f.readlines()
        with open('all_article.jsonl', 'w') as f:
            f.writelines(lines[:-4])
        # delete_last_one_line
        with open('all_popular.jsonl', 'r') as f:
            lines = f.readlines()
        with open('all_popular.jsonl', 'w') as f:
            f.writelines(lines[:-1])
        
if __name__ == '__main__':
    start = time.time() # record execution time
    tracemalloc.start() # record memory usage
    if os.path.exists('all_article.jsonl'): # delete all_article.jsonl if exists
        os.remove('all_article.jsonl')
    if os.path.exists('all_popular.jsonl'): # delete all_popular.jsonl if exists
        os.remove('all_popular.jsonl')
    for i in range(3647, 3956): # 3647, 3956
        crawler = BeautyCrawler(i)
        crawler.get_soup()
        crawler.get_articles()
        crawler.save_all_article_jsonl()
    crawler.post_process_all_article()
    end = time.time()
    current, peak = tracemalloc.get_traced_memory() # get memory usage
    print('Execution time: ', time.strftime("%H:%M:%S", time.gmtime(end - start))) # print execution time with format hh:mm:ss 
    print('Memory usage: ', f'{current / 10**6}MB')  # print memory usage with format MB