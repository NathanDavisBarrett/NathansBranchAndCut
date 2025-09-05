from LinkedList import SortedLinkedList


def test_Sorted():
    data = [10, 17, 3, 32, 100, 17, 35, 0]
    llist = SortedLinkedList(data)
    result = [e for e in llist]

    expected = [0, 3, 10, 17, 17, 32, 35, 100]

    assert len(result) == len(expected)
    for i in range(len(result)):
        assert result[i] == expected[i]


if __name__ == "__main__":
    test_Sorted()
