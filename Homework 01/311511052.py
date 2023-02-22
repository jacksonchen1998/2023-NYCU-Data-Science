import requests
from bs4 import BeautifulSoup
import tqdm
import os
import time
import tracemalloc
import sys
import json

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

    def get_article_inside_like_count_with_id(self, article_url):
        r = requests.get(article_url, headers=self.headers)
        soup = BeautifulSoup(r.text, 'html.parser')
        # get all push tags with class push-tag
        pushes = soup.find_all('div', class_='push')
        like_count, boo_count = 0, 0
        like_ids, boo_ids = {}, {}
        for push in pushes:
            push_tag = push.find("span", class_='push-tag').string.strip()
            push_id = push.find("span", class_='push-userid').string.strip()
            if push_tag == '推':
                like_count += 1
                if push_id in like_ids:
                    like_ids[push_id] += 1
                else:
                    like_ids[push_id] = 1
            elif push_tag == '噓':
                boo_count += 1
                if push_id in boo_ids:
                    boo_ids[push_id] += 1
                else:
                    boo_ids[push_id] = 1
        return like_count, boo_count, like_ids, boo_ids
        
if __name__ == '__main__':
    # python 311511052.py crawl to execute crawler 
    # python 311511052.py hello to print hello world
    # use sys.argv to get command line arguments
    if sys.argv[1] == 'crawl':
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
        tracemalloc.stop()
    elif sys.argv[1] == 'push' and sys.argv[2] != None and sys.argv[3] != None:
        start = time.time()
        tracemalloc.start()
        # set start date and end date
        start_date = str(sys.argv[2])
        end_date = str(sys.argv[3])
        # if exists push_start_date_end_date.jsonl, delete it
        if os.path.exists(f'push_{start_date}_{end_date}.json'):
            os.remove(f'push_{start_date}_{end_date}.json')
        # create jsonl file as push_start_date_end_date.jsonl
        with open(f'push_{start_date}_{end_date}.json', 'w') as f:
            f.write('')
        with open('all_article.jsonl', 'r') as f:
            res_like_count, res_boo_count = 0, 0
            res_like_ids, res_boo_ids = {}, {}
            total_like_ids, total_boo_ids = {}, {}
            for line in f:
                # convert string to dict
                article = eval(line)
                if int(start_date) <= int(article['date']) <= int(end_date):
                    # crawl push data
                    crawler = BeautyCrawler(0)
                    like_count, boo_count, res_like_ids, res_boo_ids = crawler.get_article_inside_like_count_with_id(article['url'])
                    res_like_count += like_count
                    res_boo_count += boo_count
                    # merge res_like_ids and res_boo_ids to total_like_ids and total_boo_ids
                    for key, value in res_like_ids.items():
                        if key in total_like_ids:
                            total_like_ids[key] += value
                        else:
                            total_like_ids[key] = value
                    for key, value in res_boo_ids.items():
                        if key in total_boo_ids:
                            total_boo_ids[key] += value
                        else:
                            total_boo_ids[key] = value
            # primary sort by value, secondary sort by key with dictioanry order
            total_boo_ids = {k: v for k, v in sorted(total_boo_ids.items(), key=lambda item: (-item[1], item[0]))}
            total_like_ids = {k: v for k, v in sorted(total_like_ids.items(), key=lambda item: (-item[1], item[0]))}
            # json format as {all_like: {res_like_count: int}, all_boo: {res_boo_count: int}, like 1: {id: str, count: int}, like 2: {id: str, count: int}, ...}
            res = {'all_like': {'res_like_count': res_like_count}, 'all_boo': {'res_boo_count': res_boo_count}}
            # pick top 10 like ids and boo ids
            for i, (key, value) in enumerate(total_like_ids.items()):
                if i == 10:
                    break
                res[f'like {i+1}'] = {'user_id': key, 'count': value}
            for i, (key, value) in enumerate(total_boo_ids.items()):
                if i == 10:
                    break
                res[f'boo {i+1}'] = {'user_id': key, 'count': value}
            # write to jsonl file
            with open(f'push_{start_date}_{end_date}.json', 'a') as f:
                f.write(json.dumps(res, indent=4, ensure_ascii=False))
        end = time.time()
        current, peak = tracemalloc.get_traced_memory()
        print('Execution time: ', time.strftime("%H:%M:%S", time.gmtime(end - start)))
        print('Memory usage: ', f'{current / 10**6}MB')
        tracemalloc.stop()
    elif sys.argv[1] == '-h':
        print('python 311511052.py crawl to execute crawler')
        print('python 311511052.py hello to print hello world')
        print('python 311511052.py push start_date end_date to get push data')
    else:
        print('Invalid command!, or you can use python 311511052.py -h to get help')