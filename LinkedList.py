
class LinkedListNode:
    """
    A node in a doubly-linked list structure.
    
    Each node contains a value and references to the previous and next nodes
    in the linked list, enabling bidirectional traversal.
    
    Attributes:
        value: The data stored in this node.
        nextNode (LinkedListNode): Reference to the next node in the list.
        prevNode (LinkedListNode): Reference to the previous node in the list.
    """
    def __init__(self,value,prevNode=None):
        """
        Initialize a new linked list node.
        
        Args:
            value: The data to store in this node.
            prevNode (LinkedListNode, optional): The previous node in the list. Defaults to None.
        """
        self.value = value

        self.nextNode = None
        self.prevNode = prevNode
    
class LinkedList:
    """
    A doubly-linked list implementation with bidirectional traversal support.
    
    Provides efficient insertion and deletion operations at both ends of the list.
    Supports iteration in both forward and reverse directions.
    
    Attributes:
        size (int): Number of elements in the list.
        headNode (LinkedListNode): First node in the list.
        tailNode (LinkedListNode): Last node in the list.
        reverseIteration (bool): Whether iteration should be in reverse order.
    """
    def __init__(self,arr=[],reverseIteration=False):
        """
        Initialize a new linked list.
        
        Args:
            arr (list, optional): Initial elements to populate the list. Defaults to [].
            reverseIteration (bool, optional): Whether to iterate in reverse order. Defaults to False.
        """
        self.size = 0
        self.headNode = None
        self.tailNode = None
        self.reverseIteration = reverseIteration

        for e in arr:
            self.append(e)

    def append(self,newVal):
        """
        Add a new element to the end of the list.
        
        Args:
            newVal: The value to append to the list.
        """
        newNode = LinkedListNode(newVal,self.tailNode)

        if self.headNode is None:
            self.headNode = newNode
            self.tailNode = newNode
        else:
            self.tailNode.nextNode = newNode
            self.tailNode = newNode
        
        self.size += 1

    def prepend(self,newVal):
        """
        Add a new element to the beginning of the list.
        
        Args:
            newVal: The value to prepend to the list.
        """
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
        """
        Extend this list by appending all elements from another LinkedList.
        
        Args:
            otherList (LinkedList): The linked list to append to this list.
            
        Raises:
            AssertionError: If otherList is not a LinkedList instance.
        """
        assert isinstance(otherList,LinkedList), f"Error! \"extend\" can only be used for other linked lists, not \"{type(otherList)}\""
        self.tailNode.nextNode = otherList.headNode
        otherList.headNode.prevNode = self.tailNode

        self.tailNode = otherList.tailNode
        self.size += otherList.size

    def remove(self,node:LinkedListNode):
        """
        Remove a specific node from the list.
        
        Args:
            node (LinkedListNode): The node to remove from the list.
        """
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
        """
        Make the LinkedList iterable.
        
        Returns:
            iterator: An iterator over the list values in forward or reverse order
                     depending on the reverseIteration setting.
        """
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
        """
        Return the number of elements in the list.
        
        Returns:
            int: The size of the linked list.
        """
        return self.size
    
class SortedLinkedList(LinkedList):
    """
    A LinkedList that maintains elements in sorted order based on a key function.
    
    Automatically sorts elements from lowest to greatest value when inserting.
    The key function determines the sort order by extracting a comparable value
    from each element.
    
    Attributes:
        key (callable): Function that takes an element and returns a sortable value.
        (Inherits all other attributes from LinkedList)
    """
    def __init__(self, arr=[], key=None, reverseIteration=False):
        """
        Initialize a sorted linked list.
        
        Args:
            arr (list, optional): Initial elements to add (will be sorted). Defaults to [].
            key (callable, optional): Function to extract sort key from elements.
                If None, elements are compared directly. Defaults to None.
            reverseIteration (bool, optional): Whether to iterate in reverse. Defaults to False.
        """
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
        """
        Append operation is not supported for SortedLinkedList.
        
        Raises:
            NotImplementedError: Always, since append would violate sort order.
        """
        raise NotImplementedError("\"append\" is intentionally not implemented for a SortedLinkedList. Please use \"insert\" instead.")
    
    def prepend(self, newVal):
        """
        Prepend operation is not supported for SortedLinkedList.
        
        Raises:
            NotImplementedError: Always, since prepend would violate sort order.
        """
        raise NotImplementedError("\"prepend\" is intentionally not implemented for a SortedLinkedList. Please use \"insert\" instead.")
    
    def insert(self,newVal):
        """
        Insert a new value in the appropriate sorted position.
        
        Finds the correct position to maintain sort order and inserts the element.
        The position is determined by comparing key(newVal) with existing elements.
        
        Args:
            newVal: The value to insert into the sorted list.
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


