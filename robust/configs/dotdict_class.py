class DotDict(dict):
        #Custom dictionary class that allows attribute access using the dot operator.
        def __getattr__(self, item):
            return self.get(item)

        def __setattr__(self, key, value):
            self[key] = value
