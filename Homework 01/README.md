# Data Science HW1

## Crawler

爬 PTT Beauty 板
* 2022一整年的文章
* 統計日期內推文跟噓文的數量
* 找出日期內最會推跟最會噓的人各前10名
* 統計日期內爆文的數量
* 抓取日期內爆文的所有圖片URL
* 關鍵字查詢

實作四種功能
* `python {student_id}.py crawl`
爬 2022	年一整年文章
* `python {student_id}.py push {start_date} {end_date}`
計算推文/噓文和找出前 10	名最會推跟噓的人
* `python {student_id}.py popular {start_date} {end_date}`
找爆文和圖片 URL
* `python {student_id}.py keyword {keyword} {start_date} {end_date}`
找內文中含有 {keyword}	的文章中的所有圖片

### Crawl (24%)

* 實作內容：
    * 爬2022年所有文章
    * 忽略分類為”[公告]”和”Fw: [公告]”的文章
    * 標題沒網址的可忽略
* 輸入: 無
* 輸出:
    * 兩個檔案：
        * `all_article.jsonl`: 包含所有文章
        * `all_popular.jsonl`: 包含所有爆文
    * 兩個檔案都是`jsonl`的格式，每個`json`的格式為:
    `{"date": {date}, "title": {標題}, 'url': {文章網址}}`

### Push (21%, 3 input cases, each case 7%)

* 實作內容：
找出在 start_date(含) 跟 end_date(含)	之間的：
    * 推文跟噓文的總數量
    * 推文數最多的前10名
    * 噓文數最多的前10名
* 輸入:
    * `{start_date}`
    * `{end_date}`
* 輸出:
    * 將結果輸出至檔案
    `push_{start_date}_{end_date}.json`
    * `json` format
        ```
        {
            "all_like": {總推文數量},
            "all_boo": {總噓文數量},
            "like 1": {"user_id": "{user id}", "count": {推文數}},
            "like 2": {"user_id": "{user id}", "count": {推文數}},
            ...
            "boo 1": {"user_id": "{user id}", "count": {推文數}},
            "boo 2": {"user_id": "{user id}", "count": {推文數}},
            ...
        }
        ```

### Popular (21%, 3 input cases, each case 7%)

* 實作內容：
找出在 `start_date`(含) 跟 `end_date`(含)之間的：
    * 爆文數量
    * 爆文的所有圖片網址 (包括推文和噓文的URL)
    * 開頭是http:// 或 https://，並且要以 jpg, jpeg, png, gif 為結尾，不限大小寫。
* 輸入:
    * `{start_date}`
    * `{end_date}`

* 輸出:
    * 將結果輸出至檔案
    `popular_{start_date}_{end_date}.json`
    * `json` format
        ```
        {
            "number_of_popular_articles": {爆文數量},
            "image_urls": [
                "{圖片的URL_1}",
                "{圖片的URL_2}",
                ...
            ]
        }
        ```

### Keyword (24%, 3 input cases, each case 8%)

* 實作內容：
    * 找出在 start_date(含) 跟 end_date (含) 之間且包含 keyword 的文章中所有圖片的 URL
    * URL包含在推文的圖片
    * keyword 只需考慮文章不需考慮推文 (不含空白字元)
* 輸入:
    * `{keyword}`
    * `{start_date}`
    * `{end_date}`
* 輸出：
    * 將結果輸出至檔案
        `keyword_{keyword}_{start_date}_{end_date}.json`
    * `json` format
        ```
        {
            "image_urls": [
                "{圖片的URL}",
                "{圖片的URL}",
                ...
            ]
        }
        ```

### Evaluation

`bash exe.sh` can run all the test cases and generate the result.

|task|command|time (hour:min:sec)|memory (MB)|
|:-|:-|:-|:-|
|crawl|`python 311511052.py crawl`|00:04:34|4.517032|
|push|`python 311511052.py push 0101 1231`|00:04:45|7.431676|
|popular|`python 311511052.py popular 0101 1231`|00:01:18|3.340951|
|keyword|`python 311511052.py keyword 正妹 0101 1231`|00:04:21|19.695956|

## Popularity Predictor (15%)

Goal: We want to build a binary classifier for classifying whether the image are in a popular article or not.
* Popular is 1 and unpopular is 0.
* [DEF] Popular is the article with more than 35 pushes.

`python {student_id}_pred.py {image_path_list}.txt`

* 輸入:
    * `{image_path_list}.txt` 的格式為一行一個絕
        對路徑，可直接讀取不需處理相對路徑：
        `/mnt/data/01.jpg`
        `/mnt/data/02.png`
        ...
* 輸出：
    * 輸出至 `{student_id}.txt` （當前資料夾下）
    * 格式為一行01字串，代表分類結果：
    `10111000…….01`

> Environment: 
> * Image: ubuntu: 22.04
> * Python	Version: Python 3.10
> * RAM: 8G

### Potentially	useful	materials
* [Face rating](https://github.com/HuyTu7/face_rating)
* [Facial Beauty Prediction](https://towardsdatascience.com/how-attractive-are-you-in-the-eyes-of-deep-neural-network-3d71c0755ccc)