# -*- coding: utf-8 -*-

# Define your item pipelines here
#
# Don't forget to add your pipeline to the ITEM_PIPELINES setting
# See: https://doc.scrapy.org/en/latest/topics/item-pipeline.html

import csv

class TumorPipeline(object):
    def open_spider(self, spider):
        self.table = []

    def close_spider(self, spider):
        with open('tumor.csv', mode='w') as fout:
            wrt = csv.writer(fout, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            for row in self.table:
                wrt.writerow(row)
        with open('tumor_links.txt', mode='w') as fout:
            for row in self.table:
                fout.write(f'{row[-1]}\n')

    def process_item(self, item, spider):
        spider.log(f'Processing: {item}')
        self.table.append(item['cells'])
        return item
