import scrapy
import urllib.request

class QuotesSpider(scrapy.Spider):
    name = "templates_spider"

    def start_requests(self):
        urls = []

        for i in range(1,22):
            urls.append('https://imgflip.com/memetemplates?page={}'.format(i))
        
        for url in urls:
            yield scrapy.Request(url=url, callback=self.parse)


    def readimg(self, url):
        req = urllib.request.Request(url)
        req.add_header('User-Agent', 'Mozilla/5.0 (Windows NT 6.1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/41.0.2228.0 Safari/537.3')
        with urllib.request.urlopen(req) as response:
            return response.read()

    def downloadimg(self, imgurl, imgfilename):
        with open('../data/TEMPLATES/' + imgfilename, 'wb') as f:
            f.write(self.readimg(imgurl))

    def parse(self, response):
        images = response.css('a img')
        for image in images:
            template_name = '_'.join(image.attrib['alt'].lower().split(' ')[:-2])
            if len(template_name) == 0:
                continue
            template_source = image.attrib['src']
            self.downloadimg('https:' + template_source, template_name + '.jpg')