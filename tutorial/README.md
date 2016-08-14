# Scrapy入门教程

## 一些命令

### 创建项目
```
scrapy startproject tutorial
```
该命令会创建下列内容的`tutorial`目录：　　
```
tutorial/
    scrapy.cfg
    tutorial/
        __init__.py
        items.py
        pipelines.py
        settings.py
        spiders/
            __init__.py
            ...
```
这些文件分别是：　　
* `scrapy.cfg`：项目配置文件
* `tutorial/`：存放该项目的python代码
* `tutorial/itmes.py`：定义items
* `tutorial/pipelines.py`：定义流程化的文件
* `tutorial/spiders/`：存放项目爬虫，可以自己自定义爬虫　　
代码具体如文档中操作，此处不多赘述。接下来说一些学习过程中的心得。  

### 定义Item
在定义Item时，我们需要编辑`tutorial/items.py`文件，需要声明Item的field,如下代码所示:

``` python
import scrapy

class DmozItem(scrapy.Item):
    title = scrapy.Field()
    link = scrapy.Field()
    desc = scrapy.Field()
```
### 定义spider
spider需要继承`scrpay.Spider`,它有几个属性和方法：
* `name`:爬虫的名字，可用于运行阶段指定爬虫
* `start_urls`：爬虫的起始起始网址
* `allowed_domains`：需要有，不然会报错空域名
* `parse()`：该方法主要用于爬虫爬到网页之后解析网页，可以定义一系列解析操作：保存，提取等

当然了，如果`items.py`中没有声明item的field，在spider中使用将会报出如下错误`KeyError: 'DmozItem does not support field: title'
`,而在爬虫定义文件中，item的使用可以直接使用`title`,`link`,`desc`,也可以使用字典的形式，如下代码：

```python
item = DmozItem()
item['title'] = sel.xpath('a/text()').extract()
item['link'] = sel.xpath('a/@href').extract()
item['desc'] = sel.xpath('text()').extract()
```  
### Crawling
使用命令`scrapy crawl dmoz`,dmoz为定义爬虫名字

### 提取Items
`Selectors`有两种：`xpath`和`css`，推荐使用`xpath`。下面主要讲解`xpath`用法　　

对于html标记来说，如`/html/head/title`分级标记，使用`xpath`如下代码提取其包含的内容:
```
response.xpath('/html/head/title')
response.xpath('//title')
```
此类方法属于对象`response`。上述代码提取得到的为网页标记语言,如`[<Selector xpath='//title' data=u'<title>Open Directory - Computers: Progr'>]`,
若想提取文本内容，只需使用 
```
response.xpath('//title/text()')
```
而对于html标记语言中的此类格式，如`<div class="fb-follow" data-href="https://www.facebook.com/dmoz" data-layout="button" data-show-faces="false"></div>`
提取class后内容需要使用@：
```
response.xpath('//div/@class')
```
而如果想提取至列表，则使用`extract()`方法：
```
response.xpath('//div/@class').extract()
```
即`response`对象有如下集中基础方法：
* `xpath()`
* `css()`
* `extract()`：直接提取内容
* `re()`: 正则表达式提取信息

### 跟随链接爬取
贴出代码:
```python
import scrapy

from tutorial.items import DmozItem

class DmozSpider(scrapy.Spider):
    name = "dmoz"
    allowed_domains = ["dmoz.org"]
    start_urls = [
        "http://www.dmoz.org/Computers/Programming/Languages/Python/",
    ]

    def parse(self, response):
        for href in response.css("ul.directory.dir-col > li > a::attr('href')"):
            url = response.urljoin(href.extract())
            yield scrapy.Request(url, callback=self.parse_dir_contents)

    def parse_dir_contents(self, response):
        for sel in response.xpath('//ul/li'):
            item = DmozItem()
            item['title'] = sel.xpath('a/text()').extract()
            item['link'] = sel.xpath('a/@href').extract()
            item['desc'] = sel.xpath('text()').extract()
            yield item
```

利用`scrapy.Request()`方法实现跟随链接并调用对应方法。

### 保存爬取结果
命令`scrapy crawl dmoz -o items.json`,当然这是简单保存方法，复杂可使用`pipelines.py`定义。


### Tips
在爬去新浪微博，github时，遇到了robot的阻止，z这是个需要解决的问题。