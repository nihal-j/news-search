class Node:
    
    def __init__(self):
        
        self.prefixes = 0
        self.words = 0
        # self.children = [None for i in range (128)]
        self.children = {}
        
    def add(self, suffix, currentIndex):

        '''Adds suffix[currenIndex : ] to self'''

        if currentIndex == len(suffix):
            self.words = self.words + 1
            self.prefixes = self.prefixes + 1
        
        else:
            self.prefixes = self.prefixes + 1
            firstChar = suffix[currentIndex]
            # if self.children[ord(firstChar)] == None:
            #     newNode = Node()
            #     self.children[ord(firstChar)] = newNode
            if ord(firstChar) not in self.children:
                newNode = Node()
                self.children[ord(firstChar)] = newNode
            currentIndex += 1
            self.children[ord(firstChar)].add(suffix, currentIndex)
                
    
    def count_words(self, suffix, currentIndex):

        '''Returns the count of words ending with suffix[currentIndex]'''
        
        if currentIndex == len(suffix):
            return self.words
        
        else:
            firstChar = suffix[currentIndex]
            # if self.children[ord(firstChar)] == None:
            #     return 0
            if ord(firstChar) not in self.children:
                return 0
            else:
                currentIndex += 1
                return self.children[ord(firstChar)].count_words(suffix, currentIndex)
            
    def count_prefixes(self, suffix, currentIndex):

        '''Returns the count of prefixes ending with suffix[currentIndex]'''
        
        if currentIndex == len(suffix):
            return self.prefixes
        
        else:
            firstChar = suffix[currentIndex]
            # if self.children[ord(firstChar)] == None:
            #     return 0
            if ord(firstChar) not in self.children:
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

        '''Adds suffix[currenIndex : ] to self and appends docID to self.titles
        at the end of suffix'''

        if currentIndex == len(suffix):
            self.words = self.words + 1
            self.prefixes = self.prefixes + 1
            self.titles.add(docID)

        else:
            self.prefixes = self.prefixes + 1
            firstChar = suffix[currentIndex]
            # if self.children[ord(firstChar)] == None:
            #     newNode = CollectionNode()
            #     self.children[ord(firstChar)] = newNode
            if ord(firstChar) not in self.children:
                newNode = CollectionNode()
                self.children[ord(firstChar)] = newNode
            currentIndex += 1
            self.children[ord(firstChar)].add_title(suffix, currentIndex, docID)

        
    def add_document(self, suffix, currentIndex, docID):

        '''Adds docID to self.documents at the end of suffix'''
        
        if currentIndex == len(suffix):
            self.words = self.words + 1
            self.prefixes = self.prefixes + 1
            self.documents.add(docID)
        
        else:
            self.prefixes = self.prefixes + 1
            firstChar = suffix[currentIndex]
            # if self.children[ord(firstChar)] == None:
            #     newNode = CollectionNode()
            #     self.children[ord(firstChar)] = newNode
            if ord(firstChar) not in self.children:
                newNode = CollectionNode()
                self.children[ord(firstChar)] = newNode
            currentIndex += 1
            self.children[ord(firstChar)].add_document(suffix, currentIndex, docID)
            
    def get_doc_list(self, suffix, currentIndex):

        '''Returns list of documents containing suffix'''
        
        if currentIndex == len(suffix):
            return self.documents
        
        else:
            firstChar = suffix[currentIndex]
            # if self.children[ord(firstChar)] == None:
            #     return []
            if ord(firstChar) not in self.children:
                return []
            else:
                currentIndex += 1
                return self.children[ord(firstChar)].get_doc_list(suffix, currentIndex)

    def get_title_list(self, suffix, currentIndex):

        '''Returns list of titles that contain suffix'''
        
        if currentIndex == len(suffix):
            return self.titles
        
        else:
            firstChar = suffix[currentIndex]
            # if self.children[ord(firstChar)] == None:
            #     return []
            if ord(firstChar) not in self.children:
                return []
            else:
                currentIndex += 1
                return self.children[ord(firstChar)].get_title_list(suffix, currentIndex)