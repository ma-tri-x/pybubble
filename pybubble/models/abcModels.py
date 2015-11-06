import abc


class abcModels(object):
    __metaclass__=abc.ABCMeta
    
    @abc.abstractmethod
    def foo(self):
        return

    @abc.abstractmethod
    def bar(self):
        return
    
class bla(abcModels):
    def __init__(self):
        return
    
    def bar(self):
        print 'bla'
    
    def foo(self):
        pass
    

def main():
    print 'Subclass:', issubclass(bla, abcModels)
    print 'Instance:', isinstance(bla(), abcModels)
    c = bla
    c.bar()
    
if __name__ == '__main__':
    main()