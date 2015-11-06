import abc

class abcModels(object):
    __metaclass__ = abc.ABCMeta
    
    @abc.abstractmethod
    def foo(self):
        return

    @abc.abstractmethod
    def bar(self):
        return
    
class bla(object):
    def __init__(self):
        return
    
    def blup(self):
        print 'bla'
    
    def blabla(self):
        pass
    
abcModels.register(bla)

def main():
    print 'Subclass:', issubclass(bla, abcModels)
    print 'Instance:', isinstance(bla(), abcModels)
    c = bla
    c.blup
    
if __name__ == '__main__':
    main()