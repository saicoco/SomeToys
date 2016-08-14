# -*- coding: utf-8 -*-
# author = sai

import scrapy
from tutorial.items import DmozItem

class DmozSpider(scrapy.Spider):
    name = 'dmoz'
    allowed_domains = ['dmoz.org']
    start_urls = [
        "http://www.dmoz.org/Computers/Programming/Languages/Python/Books/",
        "http://www.dmoz.org/Computers/Programming/Languages/Python/Resources/"
    ]
    # start_urls = [
    #     "http://weibo.com/?sudaref=www.baidu.com&retcode=6102"
    # ]
    # 发现被robots.txt禁止爬虫了
    def parse(self, response):
        for href in response.xpath("//ul/li/a/@href"):
            url = response.urljoin(href.extract())
            yield scrapy.Request(url, callback=self.parse_dir_content)

    def parse_dir_content(self, response):
        # filename = response.url.split("/")[-2] + '.html'
        # with open(filename, 'wb') as f:
        #     f.write(response.body)
        for sel in response.xpath('//ul/li'):
            # 简单写法
            # title = sel.xpath('a/text()').extract()
            # link = sel.xpath('a/@href').extract()
            # desc = sel.xpath('text()').extract()
            # print title, link, desc

            # 字典写法
            item = DmozItem()
            item['title'] = sel.xpath('a/text()').extract()
            item['link'] = sel.xpath('a/@href').extract()
            item['desc'] = sel.xpath('text()').extract()
            yield item