import tornado.web


class RestClass(tornado.web.RequestHandler):

    @classmethod
    def make_api(self, config):
        self.config = config
        return self