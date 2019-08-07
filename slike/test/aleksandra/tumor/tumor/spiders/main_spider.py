import scrapy

class MainSpider(scrapy.Spider):
    name = "main"
    allowed_domains = ['tumorlibrary.com']

    def start_requests(self):
        for page in range(0, 7 + 1):
            url = f"http://www.tumorlibrary.com/case/list.jsp?pager.offset={page * 25}&category_sub_type=&category=Bone+Tumors&image_type=X-ray&treatment=&order=diagnosis+ASC&sub_location=&image_id=&diagnosis=&sex=&filter.y=16&location=Distal+femur&filter.x=113&age=&category_type=&case_id="
            yield scrapy.Request(url=url, callback=self.parse)

    def parse(self, response):
        first = True
        for row in response.xpath('/html/body/table/tr[@class="vatop"]/td[2]/div/center/table/tr'):
            if first:
                first = False
                continue
            cells = [cell.css('::text').get() for cell in row.css('td')]
            image_page = f'detail.jsp?image_id={cells[1]}'
            req = response.follow(image_page, callback=self.parse_image)
            req.meta['cells'] = cells
            yield req

    def parse_image(self, response):
        cells = response.meta['cells']
#        x = response.xpath('/html/body/table/tr[1]/td/table/tr[1]/td[3]/ul/li[12]/text()[1]').get().strip().split(':')[1].strip().split(',')[0]
        x = response.xpath('/html/body/table/tr[1]/td/table/tr[1]/td[3]/ul/li')[-1].xpath('text()[1]').get().strip().split(':')[1].strip().split(',')[0]
        self.log(f'image: {response.url} x={x}')
        intx = -1
        try:
            intx = int(x)
        except:
            pass

        if intx >= 0 and intx == int(cells[1]):
            #image_url = response.xpath('/html/body/table/tr[1]/td/table/tr[1]/td[1]/a[2]/@href').get()
            image_url = f'/case/images/{int(cells[1])}.jpg'
            image_url = response.urljoin(image_url)
            cells.append(image_url)
            cells_dict = {'cells':cells}
            yield(cells_dict)

