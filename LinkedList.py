
class LinkedListNode:
    def __init__(self,value,prevNode=None):
        self.value = value

        self.nextNode = None
        self.prevNode = prevNode
    
class LinkedList:
    def __init__(self,arr=[],reverseIteration=False):
        self.size = 0
        self.headNode = None
        self.tailNode = None
        self.reverseIteration = reverseIteration

        for e in arr:
            self.append(e)

    def append(self,newVal):
        newNode = LinkedListNode(newVal,self.tailNode)

        if self.headNode is None:
            self.headNode = newNode
            self.tailNode = newNode
        else:
            self.tailNode.nextNode = newNode
            self.tailNode = newNode
        
        self.size += 1

    def prepend(self,newVal):
        newNode = LinkedListNode(newVal)

        if self.headNode is None:
            self.headNode = newNode
            self.tailNode = newNode
        else:
            newNode.nextNode = self.headNode
            self.headNode.prevNode = newNode
            self.headNode = newNode
        
        self.size += 1

    def extend(self,otherList):
        assert isinstance(otherList,LinkedList), f"Error! \"extend\" can only be used for other linked lists, not \"{type(otherList)}\""
        self.tailNode.nextNode = otherList.headNode
        otherList.headNode.prevNode = self.tailNode

        self.tailNode = otherList.tailNode
        self.size += otherList.size

    def remove(self,node:LinkedListNode):
        if node.nextNode is not None:
            node.nextNode.prevNode = node.prevNode
        if node.prevNode is not None:
            node.prevNode.nextNode = node.nextNode

        if self.headNode == node:
            self.headNode = node.nextNode
        if self.tailNode == node:
            self.tailNode = node.prevNode

        self.size -= 1

    def __iter__(self):
        if not self.reverseIteration:
            nodei = self.headNode
            while nodei is not None:
                yield nodei.value
                nodei = nodei.nextNode
        else:
            nodei = self.tailNode
            while nodei is not None:
                yield nodei.value
                nodei = nodei.prevNode

    def __len__(self):
        return self.size
    
class SortedLinkedList(LinkedList):
    """
    A LinkedList that automatically sorts added entries from lowest to greatest value based on a provided "key" function. The key function should take in an object from the LinkedList and return a number indicating the value to sort the element by within the array.
    """
    def __init__(self, arr=[], key=None, reverseIteration=False):
        if key is None:
            key = lambda x: x
        self.key = key
        self.size = 0
        self.headNode = None
        self.tailNode = None
        self.reverseIteration = reverseIteration

        for e in arr:
            self.insert(e)

        

    def append(self, newVal):
        raise NotImplementedError("\"append\" is intentionally not implemented for a SortedLinkedList. Please use \"insert\" instead.")
    
    def prepend(self, newVal):
        raise NotImplementedError("\"prepend\" is intentionally not implemented for a SortedLinkedList. Please use \"insert\" instead.")
    
    def insert(self,newVal):
        """
        A function to insert a new value to this list in the appropriate position based on it's value produced from "key"

        Arguments
        ---------
        newVal: object
            The value you'd like to insert
        """
        if len(self) == 0:
            super().append(newVal)
            return
        
        minVal = self.key(self.headNode.value)
        maxVal = self.key(self.tailNode.value)
        thisVal = self.key(newVal)

        if thisVal <= minVal:
            super().prepend(newVal)
        elif thisVal >= maxVal:
            super().append(newVal)
        else:
            #Iterate over the list to find the appropriate position for this new value.
            #   Start iterating with whichever end of the list is closer to this value.
            minDist = thisVal - minVal
            maxDist = maxVal - thisVal
            if minDist <= maxDist:
                #Start iterating at the head node.
                nodei = self.headNode
                while nodei is not None:
                    vali = self.key(nodei.value)
                    if thisVal <= vali:
                        newNode = LinkedListNode(newVal,nodei.prevNode)
                        if nodei.prevNode is not None:
                            nodei.prevNode.nextNode = newNode

                        newNode.nextNode = nodei
                        nodei.prevNode = newNode

                        break
                    nodei = nodei.nextNode
            else:
                #Start iterating at the tail node.
                nodei = self.tailNode
                while nodei is not None:
                    vali = self.key(nodei.value)
                    if thisVal >= vali:
                        newNode = LinkedListNode(newVal)
                        nodei.nextNode.prevNode = newNode
                        newNode.nextNode = nodei.nextNode
                        
                        nodei.nextNode = newNode
                        newNode.prevNode = nodei

                        break
                    nodei = nodei.prevNode
            self.size += 1


