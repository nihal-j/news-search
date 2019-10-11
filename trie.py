class Node:
    
    def __init__(self):
        
        self.prefixes = 0
        self.words = 0
        self.children = [None for i in range (256)]
        
        
    def add(self, suffix, currentIndex):
        
        print (suffix)
        
        if currentIndex == len(suffix):
            self.words = self.words + 1
            self.prefixes = self.prefixes + 1
        
        else:
            print(ord(suffix[currentIndex]))
            self.prefixes = self.prefixes + 1
            firstChar = suffix[currentIndex]
            if self.children[ord(firstChar)] == None:
                newNode = Node()
                self.children[ord(firstChar)] = newNode
            currentIndex += 1
            self.children[ord(firstChar)].add(suffix, currentIndex)
                
    
    def count_words(self, suffix, currentIndex):
        
        if currentIndex == len(suffix):
            return self.words
        
        else:
            firstChar = suffix[currentIndex]
            if self.children[ord(firstChar)] == None:
                return 0
            else:
                currentIndex += 1
                return self.children[ord(firstChar)].count_words(suffix, currentIndex)
            
    def count_prefixes(self, suffix, currentIndex):
        
        if currentIndex == len(suffix):
            return self.prefixes
        
        else:
            firstChar = suffix[currentIndex]
            if self.children[ord(firstChar)] == None:
                return 0
            else:
                currentIndex += 1
                return self.children[ord(firstChar)].count_prefixes(suffix, currentIndex)
            
class CollectionNode(Node):
    
    def __init__(self):
        super(CollectionNode,self).__init__()
        self.documents = set()
        
    def add_(self, suffix, currentIndex, docID):
        
#         print (suffix)
        
        if currentIndex == len(suffix):
            self.words = self.words + 1
            self.prefixes = self.prefixes + 1
            self.documents.add(docID)
        
        else:
#             print(ord(suffix[currentIndex]))
            self.prefixes = self.prefixes + 1
            firstChar = suffix[currentIndex]
            if self.children[ord(firstChar)] == None:
                newNode = CollectionNode()
                self.children[ord(firstChar)] = newNode
            currentIndex += 1
            self.children[ord(firstChar)].add_(suffix, currentIndex, docID)
            
    def get_doc_list(self, suffix, currentIndex):
        
        if currentIndex == len(suffix):
            return self.documents
        
        else:
            firstChar = suffix[currentIndex]
            if self.children[ord(firstChar)] == None:
                return []
            else:
                currentIndex += 1
                return self.children[ord(firstChar)].get_doc_list(suffix, currentIndex) 

## Test code

# root = Node()
# root.add('hello', 0)
# root.add('hell', 0)
# root.add('adc', 0)
# root.add('heafth', 0)
# print(
# root.count_prefixes('hel', 0),
# root.count_prefixes('hea', 0),
# root.count_words('hello',0),
# root.count_words('hell', 0),
# root.count_words('adc', 0),
# # root.count_words('hellodd',0))
print('dd')
