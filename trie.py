class Node:
    
    def __init__(self):
        
        self.prefixes = 0
        self.words = 0
        self.children = [None for i in range (256)]
        
        
    def add(self, suffix, currentIndex):
        
        if currentIndex == len(suffix):
            self.words = self.words + 1
            self.prefixes = self.prefixes + 1
        
        else:
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
        self.titles = set()

    def add_title(self, suffix, currentIndex, docID):

        if currentIndex == len(suffix):
            self.words = self.words + 1
            self.prefixes = self.prefixes + 1
            self.titles.add(docID)

        else:
            self.prefixes = self.prefixes + 1
            firstChar = suffix[currentIndex]
            if self.children[ord(firstChar)] == None:
                newNode = CollectionNode()
                self.children[ord(firstChar)] = newNode
            currentIndex += 1
            self.children[ord(firstChar)].add_title(suffix, currentIndex, docID)

        
    def add_document(self, suffix, currentIndex, docID):
        
        if currentIndex == len(suffix):
            self.words = self.words + 1
            self.prefixes = self.prefixes + 1
            self.documents.add(docID)
        
        else:
            self.prefixes = self.prefixes + 1
            firstChar = suffix[currentIndex]
            if self.children[ord(firstChar)] == None:
                newNode = CollectionNode()
                self.children[ord(firstChar)] = newNode
            currentIndex += 1
            self.children[ord(firstChar)].add_document(suffix, currentIndex, docID)
            
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

    def get_title_list(self, suffix, currentIndex):
        
        if currentIndex == len(suffix):
            return self.titles
        
        else:
            firstChar = suffix[currentIndex]
            if self.children[ord(firstChar)] == None:
                return []
            else:
                currentIndex += 1
                return self.children[ord(firstChar)].get_title_list(suffix, currentIndex)