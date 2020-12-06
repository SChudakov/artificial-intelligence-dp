#!/usr/bin/env python3
import sys


class PartialParse(object):
    def __init__(self, sentence):
        self.sentence = sentence

        self.stack = ['ROOT']
        self.buffer = sentence[:]
        self.dependencies = list()

    def parse_step(self, transition):
        if transition == 'S':
            self.stack.append(self.buffer.pop(0))
        elif transition == 'LA':
            self.dependencies.append((self.stack[-1], self.stack.pop(-2)))
        elif transition == 'RA':
            self.dependencies.append((self.stack[-2], self.stack.pop(-1)))

    def parse(self, transitions):
        for transition in transitions:
            self.parse_step(transition)
        return self.dependencies


def mini_batch_parse(sentences, model, batch_size):
    partial_parses = [PartialParse(s) for s in sentences]
    unfinished_parses = partial_parses[:]  # shallow copy

    while unfinished_parses:
        mini_batch = unfinished_parses[:batch_size]
        next_trans = model.predict(mini_batch)
        for pp, t in zip(mini_batch, next_trans):  # pp: each partial parse, t: next step(transition) from model on pp
            pp.parse_step(t)
            if len(pp.stack) == 1 and len(pp.buffer) == 0:
                unfinished_parses.remove(pp)

    dependencies = [pp.dependencies for pp in partial_parses]

    return dependencies


def test_step(name, transition, stack, buf, deps, ex_stack, ex_buf, ex_deps):
    pp = PartialParse([])
    pp.stack, pp.buffer, pp.dependencies = stack, buf, deps

    pp.parse_step(transition)
    stack, buf, deps = (tuple(pp.stack), tuple(pp.buffer), tuple(sorted(pp.dependencies)))
    assert stack == ex_stack, \
        "{:} test resulted in stack {:}, expected {:}".format(name, stack, ex_stack)
    assert buf == ex_buf, \
        "{:} test resulted in buffer {:}, expected {:}".format(name, buf, ex_buf)
    assert deps == ex_deps, \
        "{:} test resulted in dependency list {:}, expected {:}".format(name, deps, ex_deps)
    print("{:} test passed!".format(name))


def test_parse_step():
    test_step("SHIFT", "S", ["ROOT", "the"], ["cat", "sat"], [],
              ("ROOT", "the", "cat"), ("sat",), ())
    test_step("LEFT-ARC", "LA", ["ROOT", "the", "cat"], ["sat"], [],
              ("ROOT", "cat",), ("sat",), (("cat", "the"),))
    test_step("RIGHT-ARC", "RA", ["ROOT", "run", "fast"], [], [],
              ("ROOT", "run",), (), (("run", "fast"),))


def test_parse():
    sentence = ["parse", "this", "sentence"]
    dependencies = PartialParse(sentence).parse(["S", "S", "S", "LA", "RA", "RA"])
    dependencies = tuple(sorted(dependencies))
    expected = (('ROOT', 'parse'), ('parse', 'sentence'), ('sentence', 'this'))
    assert dependencies == expected, \
        "parse test resulted in dependencies {:}, expected {:}".format(dependencies, expected)
    assert tuple(sentence) == ("parse", "this", "sentence"), \
        "parse test failed: the input sentence should not be modified"
    print("parse test passed!")


class DummyModel(object):
    def predict(self, partial_parses):
        return [("RA" if pp.stack[1] is "right" else "LA") if len(pp.buffer) == 0 else "S"
                for pp in partial_parses]


def test_dependencies(name, deps, ex_deps):
    deps = tuple(sorted(deps))
    assert deps == ex_deps, \
        "{:} test resulted in dependency list {:}, expected {:}".format(name, deps, ex_deps)


def test_mini_batch_parse():
    sentences = [["right", "arcs", "only"],
                 ["right", "arcs", "only", "again"],
                 ["left", "arcs", "only"],
                 ["left", "arcs", "only", "again"]]
    deps = mini_batch_parse(sentences, DummyModel(), 2)
    test_dependencies("minibatch_parse", deps[0],
                      (('ROOT', 'right'), ('arcs', 'only'), ('right', 'arcs')))
    test_dependencies("minibatch_parse", deps[1],
                      (('ROOT', 'right'), ('arcs', 'only'), ('only', 'again'), ('right', 'arcs')))
    test_dependencies("minibatch_parse", deps[2],
                      (('only', 'ROOT'), ('only', 'arcs'), ('only', 'left')))
    test_dependencies("minibatch_parse", deps[3],
                      (('again', 'ROOT'), ('again', 'arcs'), ('again', 'left'), ('again', 'only')))
    print("minibatch_parse test passed!")


if __name__ == '__main__':
    args = sys.argv
    if len(args) != 2:
        raise Exception(
            "You did not provide a valid keyword. Either provide 'part_c' or 'part_d', when executing this script")
    elif args[1] == "part_c":
        test_parse_step()
        test_parse()
    elif args[1] == "part_d":
        test_mini_batch_parse()
    else:
        raise Exception(
            "You did not provide a valid keyword. Either provide 'part_c' or 'part_d', when executing this script")
