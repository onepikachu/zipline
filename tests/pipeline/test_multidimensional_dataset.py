from collections import OrderedDict
import itertools

from nose_parameterized import parameterized
import numpy as np

from zipline.pipeline.data import (
    Column,
    MultiDimensionalDataSet,
    MultiDimensionalDataSetSlice,
)
from zipline.testing import ZiplineTestCase
from zipline.testing.predicates import (
    assert_equal,
    assert_is,
    assert_is_not,
    assert_is_not_subclass,
    assert_is_subclass,
    assert_less,
    assert_raises_str,
)


class TestMultiDimensionalDataSet(ZiplineTestCase):
    def test_cache(self):
        class MD1(MultiDimensionalDataSet):
            free_axes = [('axis_0', ['a', 'b', 'c'])]

        class MD2(MultiDimensionalDataSet):
            free_axes = [('axis_0', ['a', 'b', 'c'])]

        MD1Slice = MD1.slice(axis_0='a')
        MD2Slice = MD2.slice(axis_0='a')

        assert_equal(MD1Slice.free_axes_labels, MD2Slice.free_axes_labels)
        assert_is_not(MD1Slice, MD2Slice)

    def test_empty_free_axes(self):
        expected_msg = (
            'MultiDimensionalDataSet must be defined with non-empty free_axes'
        )
        with assert_raises_str(ValueError, expected_msg):
            class MD(MultiDimensionalDataSet):
                free_axes = []

    def spec(*cs):
        return (cs,)

    @parameterized.expand([
        spec(
            ('axis_0', range(10))
        ),
        spec(
            ('axis_0', range(10)),
            ('axis_1', range(10, 15)),
        ),
        spec(
            ('axis_0', range(10)),
            ('axis_1', range(10, 15)),
            ('axis_2', range(5, 15)),
        ),
        spec(
            ('axis_0', range(6)),
            ('axis_1', {'a', 'b', 'c'}),
            ('axis_2', range(5, 15)),
            ('axis_3', {'b', 'c', 'e'}),
        ),
    ])
    def test_valid_slice(self, axes_spec):
        class MD(MultiDimensionalDataSet):
            free_axes = axes_spec

            f8 = Column('f8')
            i8 = Column('i8', missing_value=0)
            ob = Column('O')
            M8 = Column('M8[ns]')
            boolean = Column('?')

        expected_axes = OrderedDict([(k, frozenset(v)) for k, v in axes_spec])
        assert_equal(MD.free_axes, expected_axes)

        for valid_combination in itertools.product(*expected_axes.values()):
            Slice = MD.slice(*valid_combination)
            alternate_constructions = [
                # all positional
                MD.slice(*valid_combination),
                # all keyword
                MD.slice(**dict(zip(expected_axes.keys(), valid_combination))),
                # mix keyword/positional
                MD.slice(
                    *valid_combination[:len(valid_combination) // 2],
                    **dict(
                        list(zip(expected_axes.keys(), valid_combination))[
                            len(valid_combination) // 2:
                        ],
                    )
                ),
            ]
            for alt in alternate_constructions:
                assert_is(Slice, alt, msg='Slices are not properly memoized')

            expected_labels = OrderedDict(
                zip(expected_axes, valid_combination),
            )
            assert_equal(Slice.free_axes_labels, expected_labels)

            assert_is(Slice.parent_multidimensional_dataset, MD)

            assert_is_subclass(Slice, MultiDimensionalDataSetSlice)

            expected_columns = {
                ('f8', np.dtype('f8'), Slice),
                ('i8', np.dtype('i8'), Slice),
                ('ob', np.dtype('O'), Slice),
                ('M8', np.dtype('M8[ns]'), Slice),
                ('boolean', np.dtype('?'), Slice),
            }
            actual_columns = {
                (c.name, c.dtype, c.dataset) for c in Slice.columns
            }
            assert_equal(actual_columns, expected_columns)

    del spec

    def test_slice_unknown_axes(self):
        class MD(MultiDimensionalDataSet):
            free_axes = [
                ('axis_0', {'a', 'b', 'c'}),
                ('axis_1', {'c', 'd', 'e'}),
            ]

        def expect_slice_fails(*args, **kwargs):
            expected_msg = kwargs.pop('expected_msg')

            with assert_raises_str(TypeError, expected_msg):
                MD.slice(*args, **kwargs)

        # insufficient positional
        expect_slice_fails(
            expected_msg=(
                'no label provided for the following axes: axis_0, axis_1'
            ),
        )
        expect_slice_fails(
            'a',
            expected_msg='no label provided for the following axis: axis_1',
        )

        # too many positional
        expect_slice_fails(
            'a', 'b', 'c',
            expected_msg='MD has 2 free axes but 3 were given',
        )

        # mismatched keys
        expect_slice_fails(
            axis_2='??',
            expected_msg='MD does not have the following axis: axis_2',
        )
        expect_slice_fails(
            axis_1='??', axis_2='??',
            expected_msg='MD does not have the following axis: axis_2',
        )
        expect_slice_fails(
            axis_0='??', axis_1='??', axis_2='??',
            expected_msg='MD does not have the following axis: axis_2',
        )

        # the extra keyword axes should be sorted
        expect_slice_fails(
            axis_3='??', axis_2='??',
            expected_msg='MD does not have the following axes: axis_2, axis_3',
        )

    def test_slice_unknown_axis_label(self):
        class MD(MultiDimensionalDataSet):
            free_axes = [
                ('axis_0', {'a', 'b', 'c'}),
                ('axis_1', {'c', 'd', 'e'}),
            ]

        def expect_slice_fails(*args, **kwargs):
            expected_msg = kwargs.pop('expected_msg')

            with assert_raises_str(ValueError, expected_msg):
                MD.slice(*args, **kwargs)

        expect_slice_fails(
            'not-in-0', 'c',
            expected_msg="'not-in-0' is not a value along the axis_0 axis",
        )
        expect_slice_fails(
            axis_0='not-in-0', axis_1='c',
            expected_msg="'not-in-0' is not a value along the axis_0 axis",
        )

        expect_slice_fails(
            'a', 'not-in-1',
            expected_msg="'not-in-1' is not a value along the axis_1 axis",
        )
        expect_slice_fails(
            axis_0='a', axis_1='not-in-1',
            expected_msg="'not-in-1' is not a value along the axis_1 axis",
        )

    def test_inheritence(self):
        class Parent(MultiDimensionalDataSet):
            free_axes = [
                ('axis_0', {'a', 'b', 'c'}),
                ('axis_1', {'d', 'e', 'f'}),
            ]

            column_0 = Column('f8')
            column_1 = Column('?')

        class Child(Parent):
            column_2 = Column('O')
            column_3 = Column('i8', -1)

        assert_is_subclass(Child, Parent)
        assert_equal(Child.free_axes, Parent.free_axes)

        ParentSlice = Parent.slice(axis_0='a', axis_1='d')
        ChildSlice = Child.slice(axis_0='a', axis_1='d')

        assert_is_not_subclass(ChildSlice, ParentSlice)

        expected_child_slice_columns = frozenset({
            ChildSlice.column_0,
            ChildSlice.column_1,
            ChildSlice.column_2,
            ChildSlice.column_3,
        })
        assert_equal(ChildSlice.columns, expected_child_slice_columns)
