#emacs: -*- mode: python-mode; py-indent-offset: 4; indent-tabs-mode: nil -*-
#ex: set sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Unit tests for PyMVPA Parameter class."""

import unittest, copy

import numpy as N
from sets import Set

from mvpa.misc.state import Stateful, StateVariable
from mvpa.misc.param import Parameter, KernelParameter

class BlankClass(Stateful):
    pass

class SimpleClass(Stateful):
    C = Parameter(1.0, min=0, doc="C parameter")

class MixedClass(Stateful):
    C = Parameter(1.0, min=0, doc="C parameter")
    D = Parameter(3.0, min=0, doc="D parameter")
    state1 = StateVariable(doc="bogus")

class ParamsTests(unittest.TestCase):

    def testBlank(self):
        blank  = BlankClass()

        self.failUnlessRaises(AttributeError, blank.__getattribute__, 'states')
        self.failUnlessRaises(AttributeError, blank.__getattribute__, '')

    def testSimple(self):
        simple  = SimpleClass()

        self.failUnlessEqual(len(simple.params.items), 1)
        self.failUnlessRaises(AttributeError, simple.__getattribute__, 'dummy')
        self.failUnlessRaises(AttributeError, simple.__getattribute__, '')

        self.failUnlessEqual(simple.C, 1.0)
        self.failUnlessEqual(simple.params.isSet("C"), False)
        self.failUnlessEqual(simple.params.isSet(), False)
        simple.C = 10.0
        self.failUnlessEqual(simple.params.isSet("C"), True)
        self.failUnlessEqual(simple.params.isSet(), True)

        self.failUnlessEqual(simple.C, 10.0)
        simple.params["C"].resetvalue()
        self.failUnlessEqual(simple.params.isSet("C"), True)
        # TODO: Test if we 'train' a classifier f we get isSet to false
        self.failUnlessEqual(simple.C, 1.0)
        self.failUnlessRaises(AttributeError, simple.params.__getattribute__, 'B')

    def testMixed(self):
        mixed  = MixedClass()

        self.failUnlessEqual(len(mixed.params.items), 2)
        self.failUnlessEqual(len(mixed.states.items), 1)
        self.failUnlessRaises(AttributeError, mixed.__getattribute__, 'kernel_params')

        self.failUnlessEqual(mixed.C, 1.0)
        self.failUnlessEqual(mixed.params.isSet("C"), False)
        self.failUnlessEqual(mixed.params.isSet(), False)
        mixed.C = 10.0
        self.failUnlessEqual(mixed.params.isSet("C"), True)
        self.failUnlessEqual(mixed.params.isSet("D"), False)
        self.failUnlessEqual(mixed.params.isSet(), True)
        self.failUnlessEqual(mixed.D, 3.0)


def suite():
    return unittest.makeSuite(ParamsTests)


if __name__ == '__main__':
    import runner

