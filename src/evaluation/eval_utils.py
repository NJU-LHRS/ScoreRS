from typing import Any, Callable, Iterable, List, Optional, Tuple, Union, Iterator
import collections



class Collator:
    """
    A class for reordering and batching elements of an array.

    This class allows for sorting an array based on a provided sorting function, grouping elements based on a grouping function, and generating batches from the sorted and grouped data.
    """

    def __init__(
        self,
        arr: List,
        sort_fn: Callable,
        group_fn: Callable = lambda x: x[1],
        grouping: bool = False,
    ) -> None:
        self.grouping = grouping
        self.fn = sort_fn
        self.group_fn = lambda x: group_fn(x[1])  # first index are enumerated indices
        self.reorder_indices: List = []
        self.size = len(arr)
        self.arr_with_indices: Iterable[Any] = tuple(enumerate(arr))  # [indices, (arr)]
        if self.grouping is True:
            self.group_by_index()

    def group_by_index(self) -> None:
        self.arr_with_indices = self.group(self.arr_with_indices, fn=self.group_fn, values=False)

    def get_batched(self, n: int = 1, batch_fn: Optional[Callable] = None) -> Iterator:
        """
        Generates and yields batches from the reordered array.

        Parameters:
        - n (int): The size of each batch. Defaults to 1.
        - batch_fn (Optional[Callable[[int, Iterable], int]]): A function to determine the size of each batch. Defaults to None.

        Yields:
        Iterator: An iterator over batches of reordered elements.
        """
        if self.grouping:
            for (
                key,
                values,
            ) in self.arr_with_indices.items():  # type: ignore
                values = self._reorder(values)
                batch = self.get_chunks(values, n=n, fn=batch_fn)
                yield from batch
        else:
            values = self._reorder(self.arr_with_indices)  # type: ignore
            batch = self.get_chunks(values, n=n, fn=batch_fn)
            yield from batch

    def _reorder(self, arr: Union[List, Tuple[Tuple[int, Any], ...]]) -> Iterator:
        """
        Reorders the elements in the array based on the sorting function.

        Parameters:
        - arr (Union[List, Tuple[Tuple[int, Any], ...]]): The array or iterable to be reordered.

        Yields:
        List: Yields reordered elements one by one.
        """
        arr = sorted(arr, key=lambda x: self.fn(x[1]))
        self.reorder_indices.extend([x[0] for x in arr])
        yield from [x[1] for x in arr]

    def get_original(self, newarr: List) -> List:
        """
        Restores the original order of elements from the reordered list.

        Parameters:
        - newarr (List): The reordered array.

        Returns:
        List: The array with elements restored to their original order.
        """
        res = [None] * self.size
        cov = [False] * self.size

        for ind, v in zip(self.reorder_indices, newarr):
            res[ind] = v
            cov[ind] = True

        assert all(cov)

        return res

    def __len__(self):
        return self.size

    @staticmethod
    def group(arr: Iterable, fn: Callable, values: bool = False) -> Iterable:
        """
        Groups elements of an iterable based on a provided function.

        Parameters:
        - arr (Iterable): The iterable to be grouped.
        - fn (Callable): The function to determine the grouping.
        - values (bool): If True, returns the values of the group. Defaults to False.

        Returns:
        Iterable: An iterable of grouped elements.
        """
        res = collections.defaultdict(list)
        for ob in arr:
            try:
                hashable_dict = tuple(
                    (
                        key,
                        tuple(value) if isinstance(value, collections.abc.Iterable) else value,
                    )
                    for key, value in sorted(fn(ob).items())
                )
                res[hashable_dict].append(ob)
            except TypeError:
                res[fn(ob)].append(ob)
        if not values:
            return res
        return res.values()

    @staticmethod
    def get_chunks(_iter, n: int = 0, fn=None):
        """
        Divides an iterable into chunks of specified size or based on a given function.
        Useful for batching

        Parameters:
        - iter: The input iterable to be divided into chunks.
        - n: An integer representing the size of each chunk. Default is 0.
        - fn: A function that takes the current index and the iterable as arguments and returns the size of the chunk. Default is None.

        Returns:
        An iterator that yields chunks of the input iterable.

        Example usage:
        ```
        data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        for chunk in chunks(data, 3):
            print(chunk)
        ```
        Output:
        ```
        [1, 2, 3]
        [4, 5, 6]
        [7, 8, 9]
        [10]
        ```
        """
        arr = []
        _iter = tuple(_iter)
        for i, x in enumerate(_iter):
            arr.append(x)
            if len(arr) == (fn(i, _iter) if fn else n):
                yield arr
                arr = []

        if arr:
            yield arr